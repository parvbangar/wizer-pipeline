#!/usr/bin/env python3
"""
tools/cluster_eval/fetch_headlines.py
═════════════════════════════════════
Pull a live sample of Indian news (English + Hindi) straight from publisher
RSS feeds — the same kind of input Layer 1 stores — so the clustering
thresholds can be calibrated on real, same-day coverage instead of guesses.

Output: tools/cluster_eval/data/headlines.json  (git-ignored; regenerate any time)
  [{"id": 0, "source": "thehindu.com", "lang": "en", "title": "...",
    "description": "...", "published_at": "2026-10-03T05:12:00+00:00"}, ...]

USAGE:
  python tools/cluster_eval/fetch_headlines.py --max-age-hours 36
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser

# Several sections per large publisher on purpose: the more outlets that cover
# the same day's events, the more true "same story" pairs the sample contains.
FEEDS: list[tuple[str, str]] = [
    # ── English ──────────────────────────────────────────────────────────────
    ("en", "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("en", "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms"),   # India
    ("en", "https://timesofindia.indiatimes.com/rssfeeds/1898055.cms"),       # Business
    ("en", "https://www.thehindu.com/news/national/feeder/default.rss"),
    ("en", "https://www.thehindu.com/business/feeder/default.rss"),
    ("en", "https://www.thehindu.com/sport/feeder/default.rss"),
    ("en", "https://indianexpress.com/section/india/feed/"),
    ("en", "https://indianexpress.com/section/business/feed/"),
    ("en", "https://indianexpress.com/section/sports/feed/"),
    ("en", "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml"),
    ("en", "https://www.hindustantimes.com/feeds/rss/business/rssfeed.xml"),
    ("en", "https://www.hindustantimes.com/feeds/rss/cricket/rssfeed.xml"),
    ("en", "https://feeds.feedburner.com/ndtvnews-top-stories"),
    ("en", "https://feeds.feedburner.com/ndtvnews-india-news"),
    ("en", "https://www.livemint.com/rss/news"),
    ("en", "https://www.livemint.com/rss/markets"),
    ("en", "https://economictimes.indiatimes.com/rssfeedstopstories.cms"),
    ("en", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("en", "https://www.business-standard.com/rss/latest.rss"),
    ("en", "https://www.indiatoday.in/rss/home"),
    ("en", "https://www.indiatoday.in/rss/1206514"),
    ("en", "https://www.news18.com/rss/india.xml"),
    ("en", "https://zeenews.india.com/rss/india-national-news.xml"),
    ("en", "https://www.thehindubusinessline.com/feeder/default.rss"),
    ("en", "https://www.moneycontrol.com/rss/latestnews.xml"),
    ("en", "https://theprint.in/feed/"),
    ("en", "https://feeds.feedburner.com/ScrollinArticles.rss"),
    ("en", "https://www.newindianexpress.com/Nation/rssfeed/?id=170&getXmlFeed=true"),
    ("en", "https://www.firstpost.com/commonfeeds/v1/mfp/rss/india.xml"),
    ("en", "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml"),
    # ── Hindi ────────────────────────────────────────────────────────────────
    ("hi", "https://feeds.bbci.co.uk/hindi/rss.xml"),
    ("hi", "https://www.amarujala.com/rss/breaking-news.xml"),
    ("hi", "https://feeds.feedburner.com/ndtvkhabar"),
    ("hi", "https://www.jagran.com/rss/news/national.xml"),
    ("hi", "https://navbharattimes.indiatimes.com/rssfeedsdefault.cms"),
    ("hi", "https://www.livehindustan.com/rss/national"),
    ("hi", "https://hindi.news18.com/rss/khabar/nation/nation.xml"),
    ("hi", "https://www.aajtak.in/rssfeeds/?id=home"),
    ("hi", "https://www.abplive.com/home/feed"),
    ("hi", "https://www.bhaskar.com/rss-v1--category-1061.xml"),
]

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_UA = "Mozilla/5.0 (compatible; WizerClusterEval/1.0)"


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


def _published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            return datetime(*st[:6], tzinfo=timezone.utc)
    return None


def _fetch(lang_url: tuple[str, str]) -> tuple[str, list[dict], str | None]:
    lang, url = lang_url
    try:
        parsed = feedparser.parse(url, agent=_UA, request_headers={"Accept": "*/*"})
    except Exception as e:  # feedparser rarely raises, but network stacks can
        return url, [], f"{type(e).__name__}: {e}"
    rows = []
    for e in parsed.entries:
        title = _clean(e.get("title", ""))
        if len(title) < 15:
            continue
        rows.append({
            "lang": lang,
            "source": (urlparse(url).hostname or "").removeprefix("www.").removeprefix("feeds."),
            "title": title,
            "description": _clean(e.get("summary", ""))[:600],
            "published": _published(e),
            "link": e.get("link", ""),
        })
    err = None if rows else f"0 entries (bozo={parsed.get('bozo')}, status={parsed.get('status')})"
    return url, rows, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-age-hours", type=int, default=36)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data" / "headlines.json"))
    args = ap.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.max_age_hours)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(_fetch, FEEDS))

    seen_titles: set[str] = set()
    rows: list[dict] = []
    for url, entries, err in results:
        print(f"{len(entries):4d}  {url}" + (f"   [{err}]" if err else ""), file=sys.stderr)
        for r in entries:
            if r["published"] is not None and r["published"] < cutoff:
                continue
            key = r["title"].lower()
            if key in seen_titles:          # same feed item listed in two sections
                continue
            seen_titles.add(key)
            r["published_at"] = r.pop("published").isoformat() if r["published"] else None
            rows.append(r)

    for i, r in enumerate(rows):
        r["id"] = i
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    by_lang = {lang: sum(r["lang"] == lang for r in rows) for lang in ("en", "hi")}
    print(f"\n{len(rows)} headlines ({by_lang}) from "
          f"{len({r['source'] for r in rows})} sources → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
