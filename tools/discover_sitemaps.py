#!/usr/bin/env python3
"""
tools/discover_sitemaps.py
══════════════════════════
Find, measure and register the news sitemaps of Indian publishers
(pipeline/sitemap.py explains why sitemaps are the completeness source).

For each domain:
  1. candidates = `Sitemap:` lines of robots.txt that look like news
     (news / today / latest / google) + well-known paths;
  2. each candidate is fetched with the production reader
     (pipeline.sitemap.fetch_news_sitemap) and measured: entries published in
     the last 24 h, share carrying the news:news protocol, languages;
  3. keep a candidate only if it is a real news sitemap: ≥ MIN_24H fresh
     entries, at least half carrying the news:news protocol (a per-article
     publication date + title — a bare <lastmod> proves nothing), and few
     query-string URLs (2026-10-06: liveindia.tv served ~1,700 random
     "/?h=<n>" URLs under nine sitemap names). Drop one whose fresh URLs are
     ≥ 70 % covered by a bigger kept one (the same list under two names);
  4. --register upserts the kept sitemaps into `feeds` as
     feed_type='news_sitemap', update_cadence='breaking_news' (polled hourly),
     with the majority news:language as language_code.

USAGE:
  python tools/discover_sitemaps.py                      # data/india_news_domains.txt, report only
  python tools/discover_sitemaps.py --from-db            # + every Indian domain in feeds
  python tools/discover_sitemaps.py --from-db --register # write the feeds rows
  python tools/discover_sitemaps.py --domains a.com b.in --json out.json

Re-runnable: registration upserts on feed_url, so running it again refreshes
metadata and adds newly found sitemaps; it never deactivates anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.poller import FeedFetchError, _download_feed                     # noqa: E402
from pipeline.sitemap import (FEED_TYPE_NEWS_SITEMAP, SITEMAP_MAX_BYTES,       # noqa: E402
                              fetch_news_sitemap)

log = logging.getLogger("discover_sitemaps")

DEFAULT_LIST = Path(__file__).resolve().parents[1] / "data" / "india_news_domains.txt"
MIN_24H = 5
OVERLAP_DROP = 0.70
_NEWSY = re.compile(r"news|today|latest|google|recent|48h|daily", re.I)
_SKIP = re.compile(r"video|photo|gallery|image|amp|static|archive|author|tag|topic|category|section|liveblog|web-?stor", re.I)
WELL_KNOWN = ("/sitemap/today", "/sitemap/today.xml", "/news-sitemap.xml", "/sitemap-news.xml",
              "/sitemap/news.xml", "/googlenews.xml", "/google-news-sitemap.xml", "/sitemap_latest.xml",
              "/sitemaps/news.xml", "/sitemap/googlenews/all/all.xml")

# India's Indic-language codes: a domain whose feeds use one is Indian.
INDIC = {"hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "ur", "or", "as", "sa", "ne", "kok", "mai"}


def host_of(entry: str) -> str:
    entry = entry.strip()
    if "://" not in entry:
        entry = "https://" + entry
    return urlparse(entry).netloc.lower()


def load_domains(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(host_of(line))
    return out


def domains_from_db() -> list[str]:
    """Indian / Indian-language domains of every feed (active or not)."""
    from pipeline.db import get_client
    rows, start = [], 0
    while True:
        page = (get_client().table("feeds").select("domain, country_code, language_code, feed_type")
                .range(start, start + 999).execute().data or [])
        rows.extend(page)
        if len(page) < 1000:
            break
        start += 1000
    hosts = set()
    for r in rows:
        if (r.get("feed_type") or "") == FEED_TYPE_NEWS_SITEMAP or not r.get("domain"):
            continue
        cc = (r.get("country_code") or "").upper()
        lang = (r.get("language_code") or "").split("-")[0].lower()
        if cc in ("IN", "IND") or lang in INDIC:
            hosts.add(host_of(r["domain"]))
    return sorted(hosts)


def _download(url: str):
    return _download_feed(url, max_bytes=SITEMAP_MAX_BYTES, deadline=60)


def robots_sitemaps(host: str) -> list[str]:
    for h in dict.fromkeys([host, host if host.startswith("www.") else "www." + host]):
        try:
            body, _, _ = _download_feed(f"https://{h}/robots.txt", deadline=20)
        except FeedFetchError:
            continue
        found = re.findall(r"(?im)^\s*sitemap\s*:\s*(\S+)", body.decode("utf-8", "replace"))
        if found:
            return list(dict.fromkeys(found))
    return []


def candidates_for(host: str) -> list[str]:
    from_robots = robots_sitemaps(host)
    newsy = [u for u in from_robots if _NEWSY.search(u.split("/", 3)[-1]) and not _SKIP.search(u.split("/", 3)[-1])]
    known = [f"https://{host}{p}" for p in WELL_KNOWN]
    return list(dict.fromkeys(newsy + known))


def measure(url: str, now: datetime) -> dict | None:
    entries, meta, err = fetch_news_sitemap(url, _download, since=now - timedelta(hours=24), now=now)
    if err or not entries:
        return None
    langs = Counter((e.get("language") or "").split("-")[0] or "?" for e in entries)
    return {"url": url, "fresh_24h": len(entries), "kind": meta.get("kind"),
            "children": meta.get("children", 0),
            "news_protocol": sum(1 for e in entries if e.get("title")),
            "languages": dict(langs.most_common(5)),
            "_urls": {e["link"] for e in entries}}


def discover(host: str, now: datetime) -> dict:
    result = {"domain": host, "kept": [], "rejected": [], "candidates": 0}
    try:
        cands = candidates_for(host)
    except Exception as e:                       # one broken site never stops the sweep
        result["error"] = str(e)[:200]
        return result
    result["candidates"] = len(cands)
    measured = []
    for url in cands:
        try:
            m = measure(url, now)
        except Exception as e:
            log.debug("%s: %s", url, e)
            m = None
        if not m or m["fresh_24h"] < MIN_24H:
            continue
        if m["news_protocol"] < 0.5 * m["fresh_24h"]:
            result["rejected"].append({"url": url, "reason": "not news-sitemap protocol"})
            continue
        if sum("?" in u for u in m["_urls"]) > 0.3 * len(m["_urls"]):
            result["rejected"].append({"url": url, "reason": "query-string URLs"})
            continue
        measured.append(m)
    measured.sort(key=lambda m: (m["news_protocol"] > 0, m["fresh_24h"]), reverse=True)
    for m in measured:
        if any(len(m["_urls"] & k["_urls"]) >= OVERLAP_DROP * len(m["_urls"]) for k in result["kept"]):
            result["rejected"].append({"url": m["url"], "reason": "covered by a kept sitemap"})
            continue
        result["kept"].append(m)
    for m in result["kept"]:
        m.pop("_urls", None)
    return result


def register(results: list[dict], publisher_names: dict[str, str]) -> int:
    from pipeline.db import get_client
    rows, seen = [], set()
    for r in results:
        for m in r["kept"]:
            if m["url"] in seen:            # the same sitemap reached from two host spellings
                continue
            seen.add(m["url"])
            lang = next((k for k in m["languages"] if k != "?"), None)
            rows.append({
                "feed_url": m["url"],
                "domain": r["domain"],
                "publisher_name": publisher_names.get(r["domain"]) or r["domain"],
                "feed_type": FEED_TYPE_NEWS_SITEMAP,
                "feed_format": "xml",
                "update_cadence": "breaking_news",
                "poll_interval_mins": 60,
                "language_code": lang,
                "country_code": "IND",
                "is_active": True,
                "validation_tier": "tier1",
                "metadata_source": "sitemap_discovery",
                "avg_items_per_day": m["fresh_24h"],
                "item_count": m["fresh_24h"],
            })
    for i in range(0, len(rows), 200):
        get_client().table("feeds").upsert(rows[i:i + 200], on_conflict="feed_url").execute()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="*", help="domains to probe (default: data/india_news_domains.txt)")
    ap.add_argument("--from-db", action="store_true", help="also probe every Indian domain in feeds")
    ap.add_argument("--register", action="store_true", help="upsert kept sitemaps into feeds")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    hosts = [host_of(d) for d in args.domains] if args.domains else load_domains(DEFAULT_LIST)
    if args.from_db:
        hosts += domains_from_db()
    hosts = list(dict.fromkeys(h.removeprefix("m.") for h in hosts))
    log.info("Probing %d domains", len(hosts))
    now = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda h: discover(h, now), hosts))

    kept, seen_urls = [], set()
    for r in results:
        for m in r["kept"]:
            if m["url"] not in seen_urls:          # one sitemap reached via several host spellings
                seen_urls.add(m["url"])
                kept.append((r["domain"], m))
    total = sum(m["fresh_24h"] for _, m in kept)
    for d, m in sorted(kept, key=lambda x: -x[1]["fresh_24h"]):
        print(f"  {m['fresh_24h']:6d}/24h  {d:40s} {m['url']}  {m['languages']}")
    print(f"\n{len(kept)} sitemaps on {len({d for d, _ in kept})} of {len(hosts)} domains — "
          f"{total:,} articles in the last 24 h")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    if args.register:
        names = {}
        n = register(results, names)
        print(f"Registered / refreshed {n} news sitemaps in feeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
