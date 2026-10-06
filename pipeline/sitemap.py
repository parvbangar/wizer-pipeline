"""
pipeline/sitemap.py
═══════════════════
News sitemaps (Google News sitemap protocol, `news:news`) as an ingestion
source.

WHY:
  A publisher's news sitemap lists EVERY article it published in the last ~48 h,
  with URL, title, publication date and language. Measured 2026-10-06 on 50
  major Indian publishers: sitemaps listed ~28,500 articles/day, their RSS
  feeds ~3,400 (RSS covered 11.7 % of the union, sitemaps 96 %). Jagran: 1,066
  vs 2; Navbharat Times: 479 vs 0. Sitemaps are the completeness source; RSS
  stays for publishers without one.

HOW IT PLUGS IN:
  A sitemap is a row in `feeds` with feed_type = 'news_sitemap'. The poller
  fetches it with fetch_news_sitemap() instead of feedparser; the entries come
  back shaped like feedparser entries (link, title, published_parsed, language,
  media_thumbnail), so dedup, discovery, insert, hand-off and feed state are
  the same code path as RSS.

  Sitemap INDEXES (e.g. Dainik Bhaskar's 28 edition sitemaps) are followed one
  level, newest children first. Only entries published within
  SITEMAP_LOOKBACK_HOURS are returned — and, once the feed has been polled
  successfully, only those newer than its last success minus an overlap —
  so an hourly poll checks a few hundred URLs, not 48 h of them every time.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape

log = logging.getLogger(__name__)

FEED_TYPE_NEWS_SITEMAP = "news_sitemap"

SITEMAP_LOOKBACK_HOURS = 48          # never look further back than this
SITEMAP_OVERLAP_HOURS = 3            # re-check this much before the last success (late lastmods)
SITEMAP_MAX_CHILDREN = 40            # children of an index fetched per poll
SITEMAP_MAX_BYTES = 30 * 1024 * 1024
_IST = timezone(timedelta(hours=5, minutes=30))

_LOC = re.compile(r"<loc>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</loc>", re.S | re.I)
_LASTMOD = re.compile(r"<lastmod>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</lastmod>", re.S | re.I)
_BLOCK_SITEMAP = re.compile(r"<sitemap\b[^>]*>(.*?)</sitemap>", re.S | re.I)
_BLOCK_URL = re.compile(r"<url\b[^>]*>(.*?)</url>", re.S | re.I)


def _tag(name: str) -> re.Pattern:
    return re.compile(rf"<(?:\w+:)?{name}\b[^>]*>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</(?:\w+:)?{name}>",
                      re.S | re.I)


_PUB_DATE = _tag("publication_date")
_TITLE = _tag("title")
_LANGUAGE = _tag("language")
_KEYWORDS = _tag("keywords")
_IMAGE_LOC = re.compile(r"<image:loc>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</image:loc>", re.S | re.I)


def parse_date(text: str | None) -> datetime | None:
    """ISO 8601 (W3C) or RFC 822. A naive value is taken as IST (Indian publishers' local time)."""
    if not text:
        return None
    s = unescape(text).strip()
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        try:
            d = parsedate_to_datetime(s)
        except (TypeError, ValueError, IndexError):
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=_IST)
    return d.astimezone(timezone.utc)


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", unescape(unescape(text or ""))).strip()


def parse_sitemap(xml: str) -> tuple[str, list[dict]]:
    """
    ("index", [{"loc", "lastmod"}]) for a sitemap index, or
    ("urlset", [{"loc", "published", "title", "language", "keywords", "image", "news"}]).
    Tolerant of namespaces, CDATA and malformed surroundings (regex, not a strict parser:
    publishers' XML is frequently invalid).
    """
    head = xml[:4000].lower()
    if "<sitemapindex" in head:
        out = []
        for blk in _BLOCK_SITEMAP.findall(xml):
            m = _LOC.search(blk)
            if m:
                lm = _LASTMOD.search(blk)
                out.append({"loc": _clean(m.group(1)), "lastmod": parse_date(lm.group(1)) if lm else None})
        return "index", out
    out = []
    for blk in _BLOCK_URL.findall(xml):
        m = _LOC.search(blk)
        if not m:
            continue
        is_news = "publication_date" in blk.lower()
        pd = _PUB_DATE.search(blk)
        lm = _LASTMOD.search(blk)
        title = _TITLE.search(blk) if is_news else None
        lang = _LANGUAGE.search(blk) if is_news else None
        kw = _KEYWORDS.search(blk) if is_news else None
        img = _IMAGE_LOC.search(blk)
        out.append({
            "loc": _clean(m.group(1)),
            "published": parse_date(pd.group(1)) if pd else (parse_date(lm.group(1)) if lm else None),
            "title": _clean(title.group(1)) if title else "",
            "language": _clean(lang.group(1)).lower() if lang else "",
            "keywords": _clean(kw.group(1)) if kw else "",
            "image": _clean(img.group(1)) if img else "",
            "news": is_news,
        })
    return "urlset", out


# news:language should be ISO 639-1, but Indian publishers emit regional or
# misspelt codes (2026-10-06 discovery: Tamil as "tn").
_LANG_FIXES = {"tn": "ta", "od": "or", "ori": "or", "hin": "hi", "eng": "en", "tam": "ta",
               "tel": "te", "mal": "ml", "kan": "kn", "mar": "mr", "ben": "bn", "guj": "gu",
               "pan": "pa", "urd": "ur", "asm": "as", "hindi": "hi", "english": "en",
               "tamil": "ta", "telugu": "te", "marathi": "mr", "bengali": "bn", "bangla": "bn",
               "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "punjabi": "pa",
               "urdu": "ur", "odia": "or", "oriya": "or", "assamese": "as"}


def normalize_language(code: str | None) -> str:
    """ISO 639-1 where possible ('hi', 'en-IN' → 'en', 'tn' → 'ta'); '' if unknown."""
    c = (code or "").strip().lower().replace("_", "-")
    if not c:
        return ""
    base = c.split("-")[0]
    return _LANG_FIXES.get(c, _LANG_FIXES.get(base, base if len(base) in (2, 3) and base.isalpha() else ""))


def _as_entry(item: dict) -> dict:
    """A feedparser-shaped entry (what discover_article / the poller expect)."""
    e = {"link": item["loc"], "id": item["loc"], "title": item.get("title") or ""}
    if item.get("published"):
        e["published"] = item["published"].isoformat()
        e["published_parsed"] = item["published"].utctimetuple()
    lang = normalize_language(item.get("language"))
    if lang:
        e["language"] = lang
    if item.get("keywords"):
        e["tags"] = [{"term": k.strip()} for k in item["keywords"].split(",") if k.strip()]
    if item.get("image"):
        e["media_thumbnail"] = [{"url": item["image"]}]
    return e


def fetch_news_sitemap(url: str, download, since: datetime | None = None,
                       now: datetime | None = None) -> tuple[list[dict], dict, str | None]:
    """
    Fetch a news sitemap (or index of them) and return (entries, meta, error).

    download(url) -> (bytes, headers, final_url) and raises on failure —
    the poller passes its bounded downloader.
    since: only entries published after this (default: now - SITEMAP_LOOKBACK_HOURS).
    Undated entries are kept only from sitemaps where dated ones are rare (a
    sitemap that dates its entries is trusted to date all of them).
    """
    now = now or datetime.now(timezone.utc)
    floor = now - timedelta(hours=SITEMAP_LOOKBACK_HOURS)
    since = max(since, floor) if since else floor

    def get(u: str) -> str:
        body, _headers, _final = download(u)
        return body.decode("utf-8", errors="replace")

    try:
        kind, items = parse_sitemap(get(url))
    except Exception as e:
        return [], {}, f"fetch failed: {e}"

    meta = {"kind": kind, "children": 0}
    if kind == "index":
        children = items
        dated = [c for c in children if c["lastmod"]]
        if dated:
            children = sorted((c for c in children if c["lastmod"] and c["lastmod"] >= since - timedelta(hours=24)),
                              key=lambda c: c["lastmod"], reverse=True)
        children = children[:SITEMAP_MAX_CHILDREN]
        meta["children"] = len(children)
        items = []
        errors = 0
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="sitemap") as ex:
            for res in ex.map(lambda c: _safe_child(get, c["loc"]), children):
                if res is None:
                    errors += 1
                else:
                    items.extend(res)
        if children and errors == len(children):
            return [], meta, "every child sitemap failed"
        meta["child_errors"] = errors

    dated = sum(1 for i in items if i["published"])
    trust_dates = dated >= max(1, len(items) // 2)
    seen: set[str] = set()
    entries = []
    for it in items:
        if it["loc"] in seen:
            continue
        seen.add(it["loc"])
        if it["published"]:
            if it["published"] < since or it["published"] > now + timedelta(hours=1):
                continue
        elif trust_dates:
            continue
        entries.append(_as_entry(it))
    meta.update({"items": len(items), "entries": len(entries)})
    return entries, meta, None


def _safe_child(get, loc: str) -> list[dict] | None:
    try:
        kind, items = parse_sitemap(get(loc))
        return items if kind == "urlset" else []
    except Exception as e:
        log.debug("child sitemap %s failed: %s", loc, e)
        return None


def since_for_feed(feed: dict, now: datetime | None = None) -> datetime | None:
    """Incremental window: from the feed's last success minus the overlap."""
    last = feed.get("last_success_at")
    if not last:
        return None
    try:
        t = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t - timedelta(hours=SITEMAP_OVERLAP_HOURS)

