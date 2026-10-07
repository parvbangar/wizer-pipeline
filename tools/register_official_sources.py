#!/usr/bin/env python3
"""
tools/register_official_sources.py
══════════════════════════════════
The official Indian sources (docs/OFFICIAL_SOURCES.md) as rows of `feeds`,
polled every 15 minutes by ingest-official.yml.

  python tools/register_official_sources.py            # check every source live, report
  python tools/register_official_sources.py --register # check, then upsert the ones that answer

Each source is fetched with the production code path (pipeline.official /
feedparser) before registration; a source that fails is reported and NOT
registered. Re-runnable: upsert on feed_url, nothing is deactivated.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import official                                   # noqa: E402
from pipeline.poller import _fetch_rss_blocking                 # noqa: E402
from pipeline.language_scope import LANGUAGES  # noqa: E402

PIB = "https://www.pib.gov.in/allRel.aspx?reg={reg}&lang={lang}"

# PIB: (office reg id, language id, language code, office) — verified 2026-10-06.
# English for every office; regional languages where PIB publishes them; Delhi
# (national) in English, Hindi and Urdu. Hindi listings of the Hindi-belt
# regional offices have no separate page (they are in Delhi's Hindi listing).
PIB_LISTINGS = [
    ("3", "1", "en", "Delhi"), ("3", "2", "hi", "Delhi"), ("3", "3", "ur", "Delhi"),
    ("19", "4", "bn", "Kolkata"), ("17", "6", "pa", "Chandigarh"), ("20", "8", "kn", "Bengaluru"),
    ("1", "9", "mr", "Mumbai"), ("23", "10", "as", "Guwahati"), ("6", "11", "ta", "Chennai"),
    ("22", "13", "gu", "Ahmedabad"), ("30", "14", "mni", "Imphal"), ("24", "15", "ml", "Thiruvananthapuram"),
    ("5", "16", "te", "Hyderabad"), ("21", "18", "or", "Bhubaneswar"),
] + [(reg, "1", "en", f"office {reg}") for reg in
     ["1", "5", "6", "17", "19", "20", "21", "22", "23", "24"] + [str(r) for r in range(30, 47)]]


def sources() -> list[dict]:
    rows = []
    for reg, lang, code, office in PIB_LISTINGS:
        rows.append({"feed_url": PIB.format(reg=reg, lang=lang), "feed_type": "pib_daily",
                     "publisher_name": f"PIB {office}", "domain": "pib.gov.in", "language_code": code,
                     "publisher_type": "government"})
    rows += [
        {"feed_url": "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml",
         "feed_type": "nse_announcements", "publisher_name": "NSE corporate announcements",
         "domain": "nseindia.com", "language_code": "en", "publisher_type": "exchange"},
        {"feed_url": "https://www.bseindia.com/data/xml/announcements.xml",
         "feed_type": "bse_announcements", "publisher_name": "BSE corporate announcements",
         "domain": "bseindia.com", "language_code": "en", "publisher_type": "exchange"},
        {"feed_url": official.SEBI_LIST_AJAX, "feed_type": "sebi_listing",
         "publisher_name": "SEBI (all news listing)", "domain": "sebi.gov.in", "language_code": "en",
         "publisher_type": "regulator"},
    ]
    plain_rss = [
        ("https://rbi.org.in/pressreleases_rss.xml", "RBI press releases", "rbi.org.in", "regulator"),
        ("https://rbi.org.in/notifications_rss.xml", "RBI notifications", "rbi.org.in", "regulator"),
        ("https://rbi.org.in/speeches_rss.xml", "RBI speeches", "rbi.org.in", "regulator"),
        ("https://rbi.org.in/Publication_rss.xml", "RBI publications", "rbi.org.in", "regulator"),
        ("https://rbi.org.in/Bulletin_rss.xml", "RBI bulletin", "rbi.org.in", "regulator"),
        ("https://nsearchives.nseindia.com/content/RSS/Circulars.xml", "NSE circulars", "nseindia.com", "exchange"),
        ("https://nsearchives.nseindia.com/content/RSS/Board_Meetings.xml", "NSE board meetings", "nseindia.com", "exchange"),
        ("https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml", "NSE corporate actions", "nseindia.com", "exchange"),
        ("https://nsearchives.nseindia.com/content/RSS/Offer_Documents.xml", "NSE offer documents", "nseindia.com", "exchange"),
        ("https://www.bseindia.com/data/xml/notices.xml", "BSE notices", "bseindia.com", "exchange"),
        ("https://www.trai.gov.in/rss.xml", "TRAI", "trai.gov.in", "regulator"),
        ("https://mahasamvad.in/feed/", "Maharashtra DGIPR (Mahasamvad)", "mahasamvad.in", "government"),
        ("https://mpinfo.org/RSSFeed/RSSFeed_EngNews.xml", "MP Jansampark (English)", "mpinfo.org", "government"),
        ("https://mpinfo.org/RSSFeed/RSSFeed_News.xml", "MP Jansampark (Hindi)", "mpinfo.org", "government"),
        ("https://mpinfo.org/RSSFeed/RSSFeed_Cabinet_Decision.xml", "MP cabinet decisions", "mpinfo.org", "government"),
    ]
    for url, name, domain, ptype in plain_rss:
        rows.append({"feed_url": url, "feed_type": "rss", "publisher_name": name, "domain": domain,
                     "language_code": "hi" if url.endswith("RSSFeed_News.xml") else
                     ("mr" if "mahasamvad" in url else "en"), "publisher_type": ptype})
    for r in rows:
        r.update({"update_cadence": "official", "poll_interval_mins": 15, "country_code": "IND",
                  "is_active": True, "validation_tier": "tier1", "metadata_source": "official_registry"})
    # English + Hindi only (2026-10-07): regional-language listings are not registered.
    return [r for r in rows if r["language_code"] in LANGUAGES]


def check(row: dict) -> tuple[dict, int, str | None]:
    if row["feed_type"] == "rss":
        entries, _meta, err = _fetch_rss_blocking(row["feed_url"])
    else:
        entries, _meta, err = official.fetch_official(row["feed_type"], row["feed_url"])
    return row, len(entries), err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    rows = sources()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check, rows))
    ok = []
    for row, n, err in results:
        status = f"ERROR {err}" if err else f"{n} items"
        print(f"  {row['feed_type']:18s} {row['publisher_name']:34s} {status}")
        if not err:
            ok.append(row)
    print(f"\n{len(ok)} of {len(rows)} sources answer")
    if args.register and ok:
        from pipeline.db import get_client
        get_client().table("feeds").upsert(ok, on_conflict="feed_url").execute()
        print(f"Registered / refreshed {len(ok)} official sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
