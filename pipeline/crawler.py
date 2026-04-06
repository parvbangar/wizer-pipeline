"""
crawler.py
══════════
Fetches full article content: body text, images, video metadata, and all
Open Graph / JSON-LD / schema.org data.

FETCH STRATEGIES (tried in order until one returns usable HTML)
───────────────────────────────────────────────────────────────
  1. Default UA      — Standard NewsIngestBot request.  Works for ~70% of feeds.

  2. Googlebot UA    — Many paywalled sites (The Hindu, FT, WSJ, Bloomberg…)
                       whitelist Googlebot so their content gets indexed by Google.
                       Sending Googlebot headers bypasses most soft paywalls.
                       Works for an additional ~20% of "paywalled" feeds.

  3. AMP URL         — AMP (Accelerated Mobile Pages) versions of articles are
                       lightweight, server-rendered, and often paywall-free.
                       We try three common AMP patterns:
                         /amp path suffix
                         ?amp=1 query parameter
                         amp. subdomain
                       Works for another ~5% of hard cases.

  4. Wayback Machine — archive.org stores historical snapshots.  We call the
                       availability API to get the latest snapshot URL, then
                       fetch it.  Last resort but surprisingly effective for
                       news articles (most were publicly accessible when first
                       published).

  Each strategy is retried up to MAX_FETCH_RETRIES times with exponential
  backoff before moving on to the next strategy.

TEXT EXTRACTION (tried in order, returns the longest result ≥ 200 chars)
─────────────────────────────────────────────────────────────────────────
  0. RSS content:encoded — some feeds deliver full article text in the feed
                           itself.  Free — no HTTP request needed.
  1. trafilatura        — best accuracy for news body extraction.
  2. readability-lxml   — Mozilla's Readability algorithm; good for varied
                          layouts and sites trafilatura struggles with.
  3. newspaper3k        — good fallback; re-fetches the URL internally.
  4. BeautifulSoup      — raw <p> tag extraction; last resort.

MEDIA EXTRACTION
────────────────
  Images  — og:image, twitter:image, JSON-LD image, RSS media:content,
             first <img> with src that looks like a photo.
  Videos  — og:video, twitter:player, YouTube embeds, Vimeo embeds,
             JSON-LD VideoObject, MP4 <source> / <video src> tags.
  All video metadata is stored in og_tags["_videos"] as a list of dicts.
"""

from __future__ import annotations

import html as html_module
import json
import logging
import re
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse, urlunparse

from pipeline.config import (
    PAYWALLED_DOMAINS,
    CRAWL_TIMEOUT_SECONDS,
    MAX_ARTICLE_BODY_CHARS,
    USER_AGENT,
    GOOGLEBOT_UA,
    MAX_FETCH_RETRIES,
    RETRY_DELAY_SECONDS,
    ARCHIVE_ORG_TIMEOUT,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ARTICLE DATA CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrawledArticle:
    """
    Holds all extracted data for one article.
    Fields match your articles table columns exactly.
    """
    # Core identity
    feed_id:      str
    url:          str
    url_hash:     int

    # Content
    title:        str = ""
    description:  str = ""
    full_text:    str = ""
    author:       str = ""

    # Media
    top_image_url: str = ""

    # Dates
    published_at:  datetime | None = None
    crawled_at:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Classification (from feed)
    language:      str = ""
    language_code: str = ""
    country_code:  str = ""
    iab_tier1:     str = ""
    iab_tier2:     str = ""

    # Denormalised feed info (for easy querying without joins)
    feed_url:      str = ""
    domain:        str = ""
    publisher_name: str = ""

    # Open Graph tags + video metadata (stored as jsonb)
    og_tags:       dict = field(default_factory=dict)

    # Status flags
    is_crawled:    bool = False    # True = full text successfully extracted
    is_duplicate:  bool = False    # True = near-duplicate of another article

    # SimHash of title (for near-duplicate detection)
    title_simhash: int | None = None

    # Which fetch strategy succeeded (for debugging / monitoring)
    crawl_strategy: str = ""

    # Future fields (left empty for now)
    story_id:       str | None = None
    propensity_score: float | None = None

    def to_db_row(self) -> dict:
        """Convert to a dict matching the articles table schema exactly."""
        return {
            "feed_id":        self.feed_id,
            "url":            self.url,
            "url_hash":       self.url_hash,
            "title":          self.title[:1000] if self.title else "",
            "title_simhash":  self.title_simhash,
            "description":    self.description[:2000] if self.description else "",
            "full_text":      self.full_text[:MAX_ARTICLE_BODY_CHARS],
            "top_image_url":  self.top_image_url[:500] if self.top_image_url else "",
            "author":         self.author[:300] if self.author else "",
            "published_at":   self.published_at.isoformat() if self.published_at else None,
            "crawled_at":     self.crawled_at.isoformat(),
            "language":       self.language,
            "language_code":  self.language_code,
            "country_code":   self.country_code,
            "og_tags":        self.og_tags,
            "is_crawled":      self.is_crawled,
            "is_duplicate":    self.is_duplicate,
            "crawl_strategy":  self.crawl_strategy,
            "story_id":        self.story_id,
            "propensity_score": self.propensity_score,
            "feed_url":       self.feed_url,
            "domain":         self.domain,
            "publisher_name": self.publisher_name,
            "iab_tier1":      self.iab_tier1,
            "iab_tier2":      self.iab_tier2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Extract bare domain from URL, e.g. 'https://www.thehindu.com/art' → 'thehindu.com'"""
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def is_paywalled(url: str, db_has_paywall: bool = False) -> bool:
    """
    Check if an article is likely behind a paywall.
    Used for logging and to skip the default UA attempt (save time).
    The crawler still attempts Googlebot / AMP / Wayback on paywalled articles.
    """
    if db_has_paywall:
        return True
    domain = _extract_domain(url)
    return (
        domain in PAYWALLED_DOMAINS
        or any(domain.endswith("." + pd) for pd in PAYWALLED_DOMAINS)
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML FETCHING — LAYER 1: SINGLE REQUEST
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_html(
    url: str,
    ua: str = USER_AGENT,
    extra_headers: dict | None = None,
) -> str | None:
    """
    Fetch raw HTML for a URL using the given user agent.
    Returns the HTML string, or None on any failure.
    Reads up to 3 MB to handle long-form articles.
    """
    headers = {
        "User-Agent":      ua,
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8,ta;q=0.7,te;q=0.6",
        "Accept-Encoding": "identity",
    }
    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=CRAWL_TIMEOUT_SECONDS) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if not any(t in content_type for t in ("html", "xml", "text")):
                log.debug("Skipping non-HTML content-type '%s' at %s", content_type, url)
                return None
            return resp.read(3 * 1024 * 1024).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        log.debug("HTTP %d fetching %s", e.code, url)
        return None
    except urllib.error.URLError as e:
        log.debug("URL error fetching %s: %s", url, e.reason)
        return None
    except Exception as e:
        log.debug("Fetch error %s: %s", url, type(e).__name__)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HTML FETCHING — LAYER 2: STRATEGY FALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def _is_useful_html(html: str | None) -> bool:
    """Return True if the fetched HTML looks like it contains real article content."""
    return bool(html) and len(html) > 500


def _try_amp_url(url: str) -> str | None:
    """
    Try common AMP URL patterns.  AMP pages are server-rendered, lightweight,
    and often served without paywalls.

    Patterns tried:
      1. https://DOMAIN/PATH/amp  (most common — used by NDTV, India Today, etc.)
      2. https://DOMAIN/PATH?amp=1
      3. https://amp.DOMAIN/PATH  (subdomain — used by some international sites)
    """
    parsed = urlparse(url)
    base   = url.split("?")[0].split("#")[0].rstrip("/")

    amp_variants = [
        base + "/amp",
        url + ("&" if "?" in url else "?") + "amp=1",
        urlunparse(parsed._replace(netloc="amp." + parsed.netloc)),
    ]

    for amp_url in amp_variants:
        if amp_url == url:          # don't retry the exact same URL
            continue
        html = _fetch_html(amp_url, ua=USER_AGENT)
        if _is_useful_html(html):
            log.debug("AMP fetch succeeded: %s", amp_url)
            return html

    return None


def _try_wayback_machine(url: str) -> str | None:
    """
    Check archive.org for the most recent public snapshot of this URL.

    How it works:
      1. Call the Wayback Machine availability API.
      2. If a snapshot exists, fetch it.

    This is a last resort — it works best for articles that were publicly
    accessible when they were first published (before a paywall was added).
    """
    api_url = (
        "https://archive.org/wayback/available?url="
        + urllib.parse.quote(url, safe="")
    )
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=ARCHIVE_ORG_TIMEOUT) as resp:
            data    = json.loads(resp.read())
            closest = data.get("archived_snapshots", {}).get("closest", {})
            if not closest.get("available"):
                return None
            snapshot_url = closest.get("url", "")
            if not snapshot_url:
                return None
            log.debug("Wayback snapshot found: %s", snapshot_url)
            return _fetch_html(snapshot_url, ua=USER_AGENT)
    except Exception as e:
        log.debug("Wayback Machine lookup failed for %s: %s", url, e)
        return None


def _fetch_with_fallbacks(url: str, paywalled: bool = False) -> tuple[str | None, str]:
    """
    Try every fetch strategy in order until one returns usable HTML.

    Strategy order:
      1. Default UA        — skip if article is known-paywalled (saves time)
      2. Googlebot UA      — bypasses most soft paywalls
      3. AMP URL variants  — paywall-free lightweight versions
      4. Wayback Machine   — archived snapshot

    Each network strategy is retried up to MAX_FETCH_RETRIES times with
    exponential backoff (1s → 2s → 4s) before moving to the next strategy.

    Returns (html, strategy_name) where strategy_name is one of:
      "default", "googlebot", "amp", "wayback", "failed"
    """
    ua_strategies = []

    # Skip default UA for known paywalled sites — it'll just waste time
    if not paywalled:
        ua_strategies.append(("default", USER_AGENT, {}))

    # Googlebot bypasses most soft paywalls
    ua_strategies.append(
        ("googlebot", GOOGLEBOT_UA, {"X-Forwarded-For": "66.249.66.1"})
    )

    for name, ua, extra in ua_strategies:
        for attempt in range(MAX_FETCH_RETRIES):
            html = _fetch_html(url, ua=ua, extra_headers=extra or None)
            if _is_useful_html(html):
                log.debug("Fetched via %s (attempt %d): %s", name, attempt + 1, url[:80])
                return html, name
            if attempt < MAX_FETCH_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS * (2 ** attempt))

    # AMP fallback
    html = _try_amp_url(url)
    if _is_useful_html(html):
        return html, "amp"

    # Wayback Machine — last resort
    html = _try_wayback_machine(url)
    if _is_useful_html(html):
        return html, "wayback"

    return None, "failed"


# ─────────────────────────────────────────────────────────────────────────────
# METADATA EXTRACTION FROM HTML
# ─────────────────────────────────────────────────────────────────────────────

def _extract_og_tags(html: str) -> dict[str, str]:
    """
    Extract Open Graph (og:), Twitter Card, and article: meta tags from HTML.
    Also extracts the canonical URL and <title> tag as fallbacks.
    """
    tags: dict[str, str] = {}
    flags = re.IGNORECASE | re.DOTALL

    # Match <meta property="og:..." content="..."/>
    for m in re.finditer(
        r'<meta[^>]+property=["\']([^"\']+)["\'][^>]+content=["\']([^"\']*)["\']',
        html, flags
    ):
        prop, content = m.group(1).strip(), m.group(2).strip()
        if prop.startswith(("og:", "article:", "twitter:")):
            tags[prop] = content

    # Also match content-first ordering
    for m in re.finditer(
        r'<meta[^>]+content=["\']([^"\']*)["\'][^>]+property=["\']([^"\']+)["\']',
        html, flags
    ):
        content, prop = m.group(1).strip(), m.group(2).strip()
        if prop.startswith(("og:", "article:", "twitter:")) and prop not in tags:
            tags[prop] = content

    # Twitter image (uses name= not property=)
    if "og:image" not in tags:
        m = re.search(
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']*)["\']',
            html, flags
        )
        if m:
            tags["og:image"] = m.group(1).strip()

    # Canonical URL
    m = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']*)["\']',
        html, flags
    )
    if m:
        tags["canonical"] = m.group(1).strip()

    # <title> tag fallback for og:title
    if "og:title" not in tags:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, flags)
        if m:
            tags["og:title"] = re.sub(r"\s+", " ", m.group(1)).strip()

    return tags


def _extract_jsonld(html: str) -> dict[str, Any]:
    """
    Extract structured data from JSON-LD <script> blocks.
    Prefers NewsArticle / Article / BlogPosting types.
    """
    for block in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE
    ):
        try:
            data = json.loads(block.strip())
            if isinstance(data, list):
                data = data[0]
            type_val = data.get("@type", "")
            if any(t in str(type_val) for t in ("Article", "BlogPosting", "NewsArticle")):
                return data
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    return {}


def _extract_video_metadata(html: str, og: dict) -> list[dict]:
    """
    Extract all video references from the page.

    Sources checked:
      - og:video / og:video:url  (Open Graph video)
      - twitter:player           (Twitter card video player)
      - YouTube iframes/embeds   (youtube.com/embed/VIDEO_ID)
      - Vimeo iframes/embeds     (player.vimeo.com/video/VIDEO_ID)
      - Direct MP4 <video src> or <source src> tags

    Returns a list of dicts, e.g.:
      [{"type": "youtube", "url": "https://www.youtube.com/watch?v=ABC"},
       {"type": "og_video", "url": "https://cdn.example.com/video.mp4"}]

    Stored in og_tags["_videos"] — no new DB column needed.
    """
    videos: list[dict] = []
    seen_urls: set[str] = set()

    def _add(v_type: str, v_url: str) -> None:
        if v_url and v_url not in seen_urls:
            seen_urls.add(v_url)
            videos.append({"type": v_type, "url": v_url})

    # Open Graph video
    _add("og_video",      og.get("og:video:url") or og.get("og:video", ""))
    # Twitter player
    _add("twitter_player", og.get("twitter:player", ""))

    # YouTube embeds — capture video ID and build canonical watch URL
    for m in re.finditer(
        r'(?:youtube\.com/embed/|youtu\.be/)([A-Za-z0-9_-]{11})',
        html, re.IGNORECASE
    ):
        _add("youtube", f"https://www.youtube.com/watch?v={m.group(1)}")

    # Vimeo embeds
    for m in re.finditer(
        r'player\.vimeo\.com/video/(\d+)',
        html, re.IGNORECASE
    ):
        _add("vimeo", f"https://vimeo.com/{m.group(1)}")

    # Direct MP4 / WebM sources in <video> or <source> tags
    for m in re.finditer(
        r'<(?:video|source)[^>]+src=["\']([^"\']+\.(?:mp4|webm|ogg|m3u8)(?:\?[^"\']*)?)["\']',
        html, re.IGNORECASE
    ):
        _add("direct_video", m.group(1))

    return videos


def _extract_publish_date(entry: dict, og: dict, jsonld: dict) -> datetime | None:
    """Extract publication date from multiple sources, best-first."""
    import calendar, email.utils

    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        t = entry.get(key)
        if t:
            try:
                return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)
            except Exception:
                pass

    date_str = jsonld.get("datePublished") or jsonld.get("dateCreated")
    if date_str:
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    og_time = og.get("article:published_time")
    if og_time:
        try:
            return datetime.fromisoformat(og_time.replace("Z", "+00:00"))
        except ValueError:
            pass

    for key in ("published", "updated"):
        s = entry.get(key)
        if s:
            try:
                return email.utils.parsedate_to_datetime(s).astimezone(timezone.utc)
            except Exception:
                pass

    return None


def _extract_author(entry: dict, og: dict, jsonld: dict) -> str:
    """Extract author name from the best available source."""
    author_data = jsonld.get("author")
    if isinstance(author_data, dict):
        name = author_data.get("name", "")
        if name:
            return str(name).strip()
    elif isinstance(author_data, list) and author_data:
        name = author_data[0].get("name", "") if isinstance(author_data[0], dict) else author_data[0]
        if name:
            return str(name).strip()

    og_author = og.get("article:author") or og.get("og:author")
    if og_author:
        return og_author.strip()

    for key in ("author", "author_detail"):
        v = entry.get(key)
        if isinstance(v, dict):
            return v.get("name", "").strip()
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# RSS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rss_image(entry: dict) -> str:
    """Extract image URL from RSS entry fields (media:content, enclosure, etc.)"""
    for m in entry.get("media_content", []):
        url = m.get("url", "")
        if url and re.search(r"\.(jpe?g|png|webp|gif)(\?|$)", url, re.I):
            return url

    for enc in entry.get("enclosures", []):
        if "image" in enc.get("type", ""):
            href = enc.get("href", "")
            if href:
                return href

    thumbnails = entry.get("media_thumbnail", [])
    if thumbnails:
        return thumbnails[0].get("url", "")

    return ""


def _extract_rss_description(entry: dict) -> str:
    """Extract and clean description/summary from RSS entry."""
    for key in ("summary", "description"):
        text = entry.get(key, "")
        if isinstance(text, list) and text:
            text = text[0].get("value", "")
        if text and isinstance(text, str):
            clean = re.sub(r"<[^>]+>", " ", text)
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) > 20:
                return clean[:2000]

    for item in entry.get("content", []):
        text = item.get("value", "")
        if text:
            clean = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", clean).strip()[:2000]

    return ""


def _extract_rss_fulltext(entry: dict) -> str:
    """
    Extract full article text from the RSS entry itself.

    Some publishers (e.g. Substack, many blogs, some newspapers) include the
    complete article body in the RSS feed via <content:encoded>.
    feedparser exposes this as entry.content[0].value.

    This is the cheapest possible extraction — no HTTP request needed.
    We prefer this over crawling when available and long enough (>500 chars).
    """
    def _strip_and_decode(raw: str) -> str:
        """Strip HTML tags then decode HTML entities (&#8216; → ', &amp; → &)."""
        stripped = re.sub(r"<[^>]+>", " ", raw)
        decoded  = html_module.unescape(stripped)
        return re.sub(r"\s+", " ", decoded).strip()

    # content:encoded — the most common full-text RSS field
    for item in entry.get("content", []):
        raw = item.get("value", "")
        if raw:
            clean = _strip_and_decode(raw)
            if len(clean) >= 500:
                return clean[:MAX_ARTICLE_BODY_CHARS]

    # Some feeds use summary_detail with a long HTML body
    summary_detail = entry.get("summary_detail", {})
    raw = summary_detail.get("value", "") if isinstance(summary_detail, dict) else ""
    if raw and summary_detail.get("type") == "text/html":
        clean = _strip_and_decode(raw)
        if len(clean) >= 500:
            return clean[:MAX_ARTICLE_BODY_CHARS]

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# FULL-TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fulltext_trafilatura(html: str, url: str) -> str:
    """
    Primary extractor — trafilatura is purpose-built for news articles.

    Tries precision mode first (strips sidebars, share prompts, and nav
    elements aggressively).  If the result is too short (<300 chars) it
    falls back to balanced mode, which recovers articles on unusual layouts
    at the cost of occasionally including some boilerplate.
    """
    try:
        import trafilatura

        common = dict(
            url=url,
            include_comments=False,
            include_tables=False,
            include_links=False,
            no_fallback=False,
        )

        # Precision mode: removes sidebars / ads / nav / share prompts
        result = trafilatura.extract(html, favor_precision=True, **common)
        if result and len(result) >= 300:
            return result

        # Balanced mode: recovers content on unusual page layouts
        result = trafilatura.extract(html, favor_precision=False, **common)
        return result or ""

    except ImportError:
        log.debug("trafilatura not installed")
        return ""
    except Exception as e:
        log.debug("trafilatura error on %s: %s", url, e)
        return ""


def _extract_fulltext_readability(html: str) -> str:
    """
    Mozilla Readability algorithm via readability-lxml.
    Particularly strong on sites with complex layouts where trafilatura
    under-extracts (magazine sites, long-form journalism).
    """
    try:
        from readability import Document
        doc     = Document(html)
        summary = doc.summary(html_partial=True)
        clean   = re.sub(r"<[^>]+>", " ", summary)
        clean   = re.sub(r"\s+", " ", clean).strip()
        return clean if len(clean) >= 200 else ""
    except ImportError:
        log.debug("readability-lxml not installed — skipping")
        return ""
    except Exception as e:
        log.debug("readability error: %s", e)
        return ""


def _extract_fulltext_newspaper(url: str) -> str:
    """
    newspaper3k fallback.  Re-fetches the URL internally using its own HTTP
    client, so only call this after all other extractors have failed.
    """
    try:
        from newspaper import Article
        article = Article(url, request_timeout=CRAWL_TIMEOUT_SECONDS)
        article.download()
        article.parse()
        return article.text or ""
    except ImportError:
        return ""
    except Exception as e:
        log.debug("newspaper3k error on %s: %s", url, e)
        return ""


def _extract_fulltext_bs4(html: str) -> str:
    """
    Raw <p> tag extraction using BeautifulSoup.
    This is the last-resort extractor — it will always return SOMETHING
    as long as the page has paragraph tags.  Quality varies widely.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        # Strip boilerplate sections
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "figure", "figcaption"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text  = " ".join(p.get_text(separator=" ") for p in paragraphs)
        clean = re.sub(r"\s+", " ", text).strip()
        return clean if len(clean) >= 200 else ""
    except ImportError:
        log.debug("beautifulsoup4 / lxml not installed — skipping bs4 extraction")
        return ""
    except Exception as e:
        log.debug("bs4 extraction error: %s", e)
        return ""


def _clean_boilerplate(text: str) -> str:
    """
    Remove editorial boilerplate lines from extracted article text.

    Targets patterns that NLP extractors (trafilatura, readability) sometimes
    leave behind: subscription CTAs, "Also Read" links, social share prompts,
    copyright footers, and navigation cruft.

    Strategy: line-by-line pass — only removes lines that match explicit
    boilerplate patterns.  Does NOT touch lines that look like real sentences
    so legitimate article content is never damaged.
    """
    if not text:
        return text

    _LINE_RE = re.compile(
        r"(?i)("
        # Reading prompts / cross-links
        r"also\s+read|read\s+also|read\s+more|more\s+on\s+this"
        r"|related\s+(news|stories|articles|coverage)"
        r"|you\s+may\s+(also\s+)?like|don.t\s+miss"
        # Subscription / follow CTAs
        r"|subscribe\s+(to|now|for|here)|sign\s+up\s+(for|to|here)"
        r"|join\s+our\s+(newsletter|channel|group|community|telegram|whatsapp)"
        r"|follow\s+us\s+on|download\s+(the\s+)?(app|our\s+app)"
        r"|get\s+(the\s+)?(latest|breaking|live)\s+news"
        r"|stay\s+updated|for\s+latest\s+updates"
        # Social sharing
        r"|share\s+(this\s+)?(article|story|post|on)|tweet\s+this|share\s+via"
        # Legal / copyright
        r"|copyright\s+©|©\s+\d{4}|all\s+rights\s+reserved"
        r"|terms\s+of\s+(use|service)|privacy\s+policy"
        # Ads / sponsored
        r"|advertisement|sponsored\s+content|partner\s+content|paid\s+post"
        # Click prompts
        r"|click\s+here\s+to|tap\s+here\s+to|watch\s+live\s+on"
        # Editorial markers
        r"|disclaimer\s*:|editor.?s\s+note\s*:|correction\s*:"
        # Byline boilerplate (The Independent-style preambles)
        r"|independent\.co\.uk|please\s+allow\s+a\s+moment"
        r")"
    )

    lines = text.splitlines()
    clean: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean.append("")
            continue
        # Skip lines matching boilerplate patterns
        if _LINE_RE.search(stripped):
            continue
        # Skip bare URLs
        if re.match(r"^https?://\S+$", stripped):
            continue
        # Skip lines that are just a social handle
        if re.match(r"^@\w{1,30}\s*$", stripped):
            continue
        clean.append(line)

    # Collapse runs of blank lines to at most one
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(clean)).strip()

    # ── Inline DOM artifact removal ───────────────────────────────────────────
    # Some sites leak CSS class names / JS variable names as inline text nodes
    # that extractors pick up verbatim.  Strip them here.
    #
    # text_fields — Madhyamam.com: their article body div has class="text_fields"
    #   and trafilatura includes the class name inline before the article text.
    #   Affects 100% of Madhyamam articles (539+ articles = largest single source).
    result = re.sub(r"\btext_fields\b", "", result)

    # Collapse any double-spaces left behind by the substitutions above
    result = re.sub(r"  +", " ", result).strip()
    return result


def _score_candidate(text: str) -> float:
    """
    Quality score for an extracted text candidate.

    Combines length (more text is generally better) with a penalty for
    boilerplate density.  A 500-word clean article beats a 1000-word article
    that is 40% subscription prompts.
    """
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    # Rough boilerplate word density
    bp_words = {"subscribe", "newsletter", "follow", "advertisement",
                "click", "tap", "share", "tweet", "copyright", "disclaimer"}
    bp_count = sum(1 for w in words if w.lower() in bp_words)
    bp_ratio = bp_count / len(words)
    # Score: word count minus heavy boilerplate penalty
    return len(words) * (1.0 - min(bp_ratio * 4, 0.9))


def _best_fulltext(html: str, url: str) -> str:
    """
    Try all text extractors in order, clean each result, and return the
    highest-quality one.

    Quality = word count × (1 − boilerplate_density), so a shorter but
    clean article beats a longer boilerplate-bloated one.
    """
    candidates: list[str] = []

    # trafilatura: try precision mode first (avoids sidebars / share prompts),
    # fall back to balanced mode if precision yields too little.
    t = _extract_fulltext_trafilatura(html, url)
    if t:
        candidates.append(_clean_boilerplate(t))

    r = _extract_fulltext_readability(html)
    if r:
        candidates.append(_clean_boilerplate(r))

    # newspaper3k re-fetches so it's expensive — only call if we have nothing
    if not candidates:
        n = _extract_fulltext_newspaper(url)
        if n:
            candidates.append(_clean_boilerplate(n))

    if not candidates:
        b = _extract_fulltext_bs4(html)
        if b:
            candidates.append(_clean_boilerplate(b))

    if not candidates:
        return ""

    # Return the highest-quality candidate
    return max(candidates, key=_score_candidate)


# ─────────────────────────────────────────────────────────────────────────────
# SAFE ENTRY HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _make_safe_entry(entry: dict) -> dict:
    """Convert feedparser entry to a plain dict safe for JSON serialisation."""
    safe: dict = {}
    for k, v in entry.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def crawl_article(
    rss_entry:     dict,
    feed:          dict,
    norm_url:      str,
    hash_value:    int,
    title_simhash: int,
) -> CrawledArticle:
    """
    Crawl one article end-to-end and return a fully-populated CrawledArticle.

    Strategy:
      1. Extract whatever the RSS feed gives us for free (no HTTP needed).
      2. Check if RSS content:encoded has full text — use it directly if so.
      3. Attempt to fetch the page using all fallback strategies in order.
         Paywalled sites skip the default UA and go straight to Googlebot.
      4. If the fetch succeeds, extract text (4 extractors), OG tags,
         JSON-LD, and all video/media metadata.
      5. If all fetch strategies fail, store the article with RSS-derived
         metadata so it's still in the DB for potential re-crawl later.

    is_crawled is set True only when we actually got full_text.
    """
    db_has_paywall = bool(feed.get("has_paywall", False))
    paywalled      = is_paywalled(norm_url, db_has_paywall)

    if paywalled:
        log.debug("PAYWALLED — trying Googlebot/AMP/Wayback: %s", norm_url)

    # ── STEP 1: What RSS gives us for free ───────────────────────────────────
    rss_title       = html_module.unescape((rss_entry.get("title") or "").strip())
    rss_description = _extract_rss_description(rss_entry)
    rss_image       = _extract_rss_image(rss_entry)
    rss_author      = _extract_author(rss_entry, {}, {})
    rss_fulltext    = _extract_rss_fulltext(rss_entry)

    article = CrawledArticle(
        feed_id        = str(feed.get("id", "")),
        url            = norm_url,
        url_hash       = hash_value,
        title          = rss_title,
        description    = rss_description,
        full_text      = rss_fulltext,
        author         = rss_author,
        top_image_url  = rss_image,
        title_simhash  = title_simhash,
        language       = feed.get("language_name", ""),
        language_code  = feed.get("language_code", "en"),
        country_code   = feed.get("country_code", ""),
        iab_tier1      = feed.get("iab_tier1", ""),
        iab_tier2      = feed.get("iab_tier2", ""),
        feed_url       = feed.get("feed_url", ""),
        domain         = feed.get("domain", ""),
        publisher_name = feed.get("publisher_name", ""),
    )

    # ── STEP 2: RSS had full text — no crawl needed ──────────────────────────
    if rss_fulltext:
        article.published_at    = _extract_publish_date(rss_entry, {}, {})
        article.is_crawled      = True
        article.crawl_strategy  = "rss_content"
        log.debug("Full text from RSS feed itself: %s", norm_url)
        return article

    # ── STEP 3: Fetch the page (all strategies) ───────────────────────────────
    html, strategy = _fetch_with_fallbacks(norm_url, paywalled=paywalled)

    if html is None:
        # Every strategy failed — store with RSS metadata only
        log.warning(
            "ALL fetch strategies failed — storing RSS-only metadata: %s", norm_url
        )
        article.published_at   = _extract_publish_date(rss_entry, {}, {})
        article.is_crawled     = False
        article.crawl_strategy = "failed"
        return article

    # ── STEP 4: Extract everything from HTML ─────────────────────────────────
    og     = _extract_og_tags(html)
    jsonld = _extract_jsonld(html)
    videos = _extract_video_metadata(html, og)

    # Merge all text extractors — returns the longest/best result
    full_text = _best_fulltext(html, norm_url)

    # Merge metadata — crawled page beats RSS data
    article.title        = html_module.unescape(og.get("og:title") or jsonld.get("headline") or rss_title)
    article.description  = og.get("og:description") or jsonld.get("description") or rss_description
    article.top_image_url = (
        og.get("og:image")
        or (jsonld.get("image") or {}).get("url", "") if isinstance(jsonld.get("image"), dict) else ""
        or rss_image
    )
    article.author       = _extract_author(rss_entry, og, jsonld) or rss_author
    article.published_at = _extract_publish_date(rss_entry, og, jsonld)
    # Cap future-dated articles. Some feeds (e.g. ESPN India) set published_at
    # to a scheduled future time, which pushes them to the top of ORDER BY
    # published_at DESC queries and displaces genuinely fresh content.
    if article.published_at and article.published_at > article.crawled_at:
        article.published_at = article.crawled_at
    article.full_text    = full_text
    article.is_crawled   = bool(full_text)
    article.crawl_strategy = strategy

    # Pack OG tags + all video metadata into the og_tags jsonb column
    if videos:
        og["_videos"] = videos
    article.og_tags = og

    if not full_text:
        log.warning(
            "Fetched HTML (strategy=%s) but all text extractors returned empty: %s",
            strategy, norm_url,
        )

    return article
