"""
crawler.py
══════════
Fetches full article content: body text, images, metadata, Open Graph tags.

WHAT DOES THIS MODULE DO?
  When the RSS feed gives us an article URL, it usually provides:
    - title (often truncated)
    - a short description/summary
    - publication date
    - author (sometimes)
  
  But it rarely gives us:
    - the full article body
    - high-quality images
    - Open Graph metadata (og:title, og:image, og:description)
    - structured metadata (JSON-LD)
  
  This module fetches the actual article page and extracts all of that.

THE EXTRACTION PIPELINE:
  1. Fetch the HTML page (with retries, timeout, proper User-Agent)
  2. Parse Open Graph tags (og:title, og:image, og:description, etc.)
  3. Parse JSON-LD structured data (used by Google News, news sites)
  4. Extract full article text with trafilatura (primary) or newspaper3k (fallback)
  5. Package everything into a clean dict matching your articles table schema

PAYWALL HANDLING:
  Paywalled sites (thehindu.com, ft.com, etc.) block crawlers.
  When we detect a paywall:
    - We still store the article (title, description, URL from RSS)
    - is_crawled = False (we know we didn't get the full text)
    - full_text = "" (empty)
    - top_image_url = from RSS or og:image (often available even on paywall pages)
  
  WORKAROUNDS FOR PAYWALLED CONTENT:
    Option A: Google Cache — sometimes available, often blocked too
    Option B: archive.org — works for older articles, not real-time
    Option C: 12ft.io — a proxy service that removes paywalls (unreliable)
    Option D: Diffbot / Firecrawl — paid services that handle paywalls properly
    Option E: Use the RSS summary (what we do here — free and reliable)
  
  For a zero-budget startup, Option E is the right choice.
  You can add Option D later when you have revenue.

METADATA WE EXTRACT FROM OG TAGS:
  og:title       — article title (usually better than RSS title)
  og:description — short description
  og:image       — main article image (what you see when shared on WhatsApp)
  og:url         — canonical URL
  og:type        — "article" for news articles
  article:published_time — ISO 8601 publication timestamp
  article:author         — author name
  article:section        — category (e.g. "Sports", "Business")
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from pipeline.config import (
    PAYWALLED_DOMAINS,
    CRAWL_TIMEOUT_SECONDS,
    MAX_ARTICLE_BODY_CHARS,
    USER_AGENT,
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

    # Open Graph tags (full dump as dict → stored as jsonb)
    og_tags:       dict = field(default_factory=dict)

    # Status flags
    is_crawled:    bool = False    # True = full text successfully extracted
    is_duplicate:  bool = False    # True = near-duplicate of another article

    # SimHash of title (for near-duplicate detection)
    title_simhash: int | None = None

    # Future fields (left empty for now)
    story_id:       str | None = None
    propensity_score: float | None = None

    def to_db_row(self) -> dict:
        """
        Convert to a dict matching your articles table schema exactly.
        All field names here MUST match your database column names.
        """
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
            "og_tags":        self.og_tags,          # jsonb — dict goes in as-is
            "is_crawled":     self.is_crawled,
            "is_duplicate":   self.is_duplicate,
            "story_id":       self.story_id,
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
    Check if an article is behind a paywall.

    Two sources (belt + braces):
      1. db_has_paywall: the has_paywall column in your feeds table
      2. Domain check: is the domain in our PAYWALLED_DOMAINS config list?

    Either source triggers paywall mode.
    """
    if db_has_paywall:
        return True
    domain = _extract_domain(url)
    # Check exact match + subdomain match (e.g. epaper.thehindu.com)
    return (
        domain in PAYWALLED_DOMAINS
        or any(domain.endswith("." + pd) for pd in PAYWALLED_DOMAINS)
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_html(url: str) -> str | None:
    """
    Fetch the raw HTML of an article page.

    Uses Python's built-in urllib (no extra dependencies).
    
    Headers we send:
      User-Agent: Identifies us as a bot (ethical — we're not pretending to be a browser)
      Accept: We prefer HTML
      Accept-Language: English first, then Indian languages
    
    Returns the HTML as a string, or None if the fetch fails.
    Max 2MB downloaded — prevents hanging on huge pages.
    """
    request = urllib.request.Request(url, headers={
        "User-Agent":      USER_AGENT,
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8,ta;q=0.7,te;q=0.6",
        "Accept-Encoding": "identity",   # don't compress — simpler to handle
    })
    try:
        with urllib.request.urlopen(request, timeout=CRAWL_TIMEOUT_SECONDS) as resp:
            content_type = resp.headers.get("Content-Type", "")
            # Skip non-HTML responses (PDFs, images, etc.)
            if not any(t in content_type for t in ("html", "xml", "text")):
                log.debug("Skipping non-HTML content-type '%s' at %s", content_type, url)
                return None
            # Read up to 2MB
            return resp.read(2 * 1024 * 1024).decode("utf-8", errors="replace")
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
# METADATA EXTRACTION FROM HTML
# ─────────────────────────────────────────────────────────────────────────────

def _extract_og_tags(html: str) -> dict[str, str]:
    """
    Extract Open Graph (og:) and Twitter Card meta tags from HTML.

    These tags are placed by publishers specifically for social sharing
    and news aggregators — they're the most reliable source of metadata.

    Example HTML these patterns match:
      <meta property="og:title" content="Budget 2024: FM announces..."/>
      <meta property="og:image" content="https://cdn.thehindu.com/img.jpg"/>
      <meta name="twitter:image" content="https://cdn.thehindu.com/img.jpg"/>
      <link rel="canonical" href="https://thehindu.com/news/art.html"/>

    Returns a dict with all found og: and article: tags.
    Stored in og_tags column as jsonb.
    """
    tags: dict[str, str] = {}
    flags = re.IGNORECASE | re.DOTALL

    # Match <meta property="og:..." content="..."/> patterns
    for m in re.finditer(
        r'<meta[^>]+property=["\']([^"\']+)["\'][^>]+content=["\']([^"\']*)["\']',
        html, flags
    ):
        prop, content = m.group(1).strip(), m.group(2).strip()
        if prop.startswith(("og:", "article:", "twitter:")):
            tags[prop] = content

    # Also match content-first ordering: <meta content="..." property="..."/>
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

    # <title> tag (fallback for og:title)
    if "og:title" not in tags:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, flags)
        if m:
            tags["og:title"] = re.sub(r"\s+", " ", m.group(1)).strip()

    return tags


def _extract_jsonld(html: str) -> dict[str, Any]:
    """
    Extract structured data from JSON-LD <script> blocks.

    News sites use JSON-LD (JSON Linked Data) to tell Google News about
    their articles.  It contains the most structured, reliable metadata.

    Example:
      <script type="application/ld+json">
      {
        "@type": "NewsArticle",
        "headline": "Budget 2024 analysis",
        "author": {"name": "John Doe"},
        "datePublished": "2024-02-01T10:00:00+05:30",
        "image": {"url": "https://cdn.example.com/img.jpg"}
      }
      </script>
    """
    for block in re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, re.DOTALL | re.IGNORECASE
    ):
        try:
            data = json.loads(block.strip())
            # JSON-LD can be a list or a single object
            if isinstance(data, list):
                data = data[0]
            # We want Article, NewsArticle, BlogPosting etc.
            type_val = data.get("@type", "")
            if any(t in str(type_val) for t in ("Article", "BlogPosting", "NewsArticle")):
                return data
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    return {}


def _extract_publish_date(entry: dict, og: dict, jsonld: dict) -> datetime | None:
    """
    Extract publication date from multiple sources, best-first.

    Priority:
      1. feedparser parsed_time (most reliable — already structured)
      2. JSON-LD datePublished (structured, from the page)
      3. og:article:published_time (from Open Graph)
      4. feedparser date strings (less reliable but widely available)
    """
    import calendar, email.utils

    # 1. feedparser's already-parsed time struct
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        t = entry.get(key)
        if t:
            try:
                return datetime.fromtimestamp(calendar.timegm(t), tz=timezone.utc)
            except Exception:
                pass

    # 2. JSON-LD datePublished
    date_str = jsonld.get("datePublished") or jsonld.get("dateCreated")
    if date_str:
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    # 3. Open Graph article:published_time
    og_time = og.get("article:published_time")
    if og_time:
        try:
            return datetime.fromisoformat(og_time.replace("Z", "+00:00"))
        except ValueError:
            pass

    # 4. RSS date strings
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
    # JSON-LD author (most structured)
    author_data = jsonld.get("author")
    if isinstance(author_data, dict):
        name = author_data.get("name", "")
        if name:
            return str(name).strip()
    elif isinstance(author_data, list) and author_data:
        name = author_data[0].get("name", "") if isinstance(author_data[0], dict) else author_data[0]
        if name:
            return str(name).strip()

    # Open Graph
    og_author = og.get("article:author") or og.get("og:author")
    if og_author:
        return og_author.strip()

    # feedparser entry
    for key in ("author", "author_detail"):
        v = entry.get(key)
        if isinstance(v, dict):
            return v.get("name", "").strip()
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def _extract_rss_image(entry: dict) -> str:
    """Extract image URL from RSS entry fields (media:content, enclosure, etc.)"""
    # media:content (most common in news RSS)
    for m in entry.get("media_content", []):
        url = m.get("url", "")
        if url and re.search(r"\.(jpe?g|png|webp|gif)(\?|$)", url, re.I):
            return url

    # enclosure (used in podcast-style RSS)
    for enc in entry.get("enclosures", []):
        if "image" in enc.get("type", ""):
            href = enc.get("href", "")
            if href:
                return href

    # media:thumbnail
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
            # Remove HTML tags
            clean = re.sub(r"<[^>]+>", " ", text)
            # Collapse whitespace
            clean = re.sub(r"\s+", " ", clean).strip()
            if len(clean) > 20:
                return clean[:2000]

    # Try content field
    for item in entry.get("content", []):
        text = item.get("value", "")
        if text:
            clean = re.sub(r"<[^>]+>", " ", text)
            return re.sub(r"\s+", " ", clean).strip()[:2000]

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# FULL-TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_fulltext_trafilatura(html: str, url: str) -> str:
    """
    Extract article body text using trafilatura.

    Trafilatura is a Python library specifically designed for news article
    extraction.  It removes navigation, ads, footers, and other boilerplate
    and returns only the main article text.

    It's the best free tool available for this task.
    """
    try:
        import trafilatura
        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            include_links=False,
            no_fallback=False,           # use fallback methods if main method fails
            favor_precision=False,        # favor recall (get more text, even if noisy)
        )
        return result or ""
    except ImportError:
        log.debug("trafilatura not installed — skipping full-text extraction")
        return ""
    except Exception as e:
        log.debug("trafilatura error on %s: %s", url, e)
        return ""


def _extract_fulltext_newspaper(url: str) -> str:
    """
    Fallback full-text extraction using newspaper3k.

    newspaper3k is another library for article extraction.
    It re-fetches the URL (slow — extra network request) but is more
    robust on some sites where trafilatura struggles.
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


def _make_safe_entry(entry: dict) -> dict:
    """
    Convert feedparser entry to a plain dict safe for JSON serialisation.
    feedparser uses struct_time objects which aren't JSON-serialisable.
    """
    safe: dict = {}
    for k, v in entry.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)   # test if it's JSON-serialisable
                safe[k] = v
            except (TypeError, ValueError):
                safe[k] = str(v)   # convert to string if not
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def crawl_article(
    rss_entry:     dict,    # feedparser entry dict
    feed:          dict,    # feed row from your feeds table
    norm_url:      str,     # normalised article URL
    hash_value:    int,     # url_hash integer
    title_simhash: int,     # SimHash of title for near-dedup
) -> CrawledArticle:
    """
    Main entry point for article crawling.
    Given an RSS entry, returns a CrawledArticle with all available metadata.

    The result is the same whether the article is paywalled or not —
    but paywalled articles will have is_crawled=False and full_text="".

    Args:
      rss_entry:     The raw feedparser entry dict for this article
      feed:          The feed's database row (for domain, publisher, etc.)
      norm_url:      The normalised URL (already processed by dedup.normalise_url)
      hash_value:    The url_hash integer (already computed)
      title_simhash: The SimHash of the title (already computed)
    """
    db_has_paywall = bool(feed.get("has_paywall", False))
    paywalled      = is_paywalled(norm_url, db_has_paywall)

    # --- Extract what RSS gives us for free ---
    rss_title       = (rss_entry.get("title") or "").strip()
    rss_description = _extract_rss_description(rss_entry)
    rss_image       = _extract_rss_image(rss_entry)
    rss_author      = _extract_author(rss_entry, {}, {})

    # Build the result object with feed metadata pre-filled
    article = CrawledArticle(
        feed_id        = str(feed.get("id", "")),
        url            = norm_url,
        url_hash       = hash_value,
        title          = rss_title,
        description    = rss_description,
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

    # ── PAYWALLED ARTICLE PATH ────────────────────────────────────────────────
    if paywalled:
        # Store what we have from RSS, but don't crawl the page
        log.debug("PAYWALL — storing RSS metadata only: %s", norm_url)
        article.is_crawled   = False
        article.published_at = _extract_publish_date(rss_entry, {}, {})
        # Try to get og:image with a minimal fetch (many paywall pages
        # still serve OG tags without requiring a subscription)
        html = _fetch_html(norm_url)
        if html:
            og = _extract_og_tags(html)
            article.og_tags       = og
            article.top_image_url = og.get("og:image") or rss_image
            if not article.title:
                article.title = og.get("og:title", "")
            if not article.description:
                article.description = og.get("og:description", "")
        return article

    # ── NON-PAYWALLED ARTICLE PATH ────────────────────────────────────────────
    html = _fetch_html(norm_url)

    if html is None:
        # Network failure — store RSS metadata, mark as not crawled
        log.debug("FETCH FAILED — storing RSS metadata: %s", norm_url)
        article.is_crawled   = False
        article.published_at = _extract_publish_date(rss_entry, {}, {})
        return article

    # Parse all metadata from HTML
    og     = _extract_og_tags(html)
    jsonld = _extract_jsonld(html)

    # Full text extraction: trafilatura first, newspaper3k fallback
    full_text = _extract_fulltext_trafilatura(html, norm_url)
    if not full_text:
        log.debug("trafilatura returned empty — trying newspaper3k for %s", norm_url)
        full_text = _extract_fulltext_newspaper(norm_url)

    # Merge metadata — crawled data beats RSS data
    article.title        = og.get("og:title") or jsonld.get("headline") or rss_title
    article.description  = og.get("og:description") or jsonld.get("description") or rss_description
    article.top_image_url = (
        og.get("og:image")
        or (jsonld.get("image", {}) or {}).get("url", "")
        or rss_image
    )
    article.author       = _extract_author(rss_entry, og, jsonld) or rss_author
    article.published_at = _extract_publish_date(rss_entry, og, jsonld)
    article.og_tags      = og          # full dump into jsonb column
    article.full_text    = full_text
    article.is_crawled   = bool(full_text)   # True only if we got real text

    return article
