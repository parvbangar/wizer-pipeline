#!/usr/bin/env python3
"""
tools/coverage_audit.py
═══════════════════════
How complete is WIZER? Measured daily, independently, per publisher
(.github/workflows/coverage.yml; results in coverage_daily, see
docs/coverage_migration.sql; method in docs/COVERAGE.md).

For day D (default: yesterday, UTC):

  1. SITEMAP MISS RATE — every registered news sitemap is the publisher's own
     list of what it published. URLs dated D that we have not stored are
     misses. (Run early: sitemaps hold ~48 h, so D is still listed.)
  2. GDELT CAPTURE–RECAPTURE — GDELT discovers articles independently of us.
     For each Indian domain: reference = GDELT's URLs from D, both = those we
     also stored. Our miss rate on GDELT's sample estimates our miss rate
     overall, and Lincoln–Petersen N = ours × reference / both estimates the
     domain's true output. Indian hosts GDELT saw that have no source in
     `feeds` are listed for adding.
  3. YIELD — our own stored count per domain; a domain whose count falls by
     more than half against its previous measurement has a dead source.

Existence checks use the ingestion's own URL hashing (pipeline.dedup), so a
"miss" means exactly "the pipeline would not dedup this URL as known".

USAGE:
  python tools/coverage_audit.py                 # yesterday, sitemaps + GDELT, write results
  python tools/coverage_audit.py --day 2026-10-06 --no-gdelt --dry-run
  python tools/coverage_audit.py --gdelt-every 2  # every 2nd 15-min GDELT file (default 2)
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("coverage_audit")

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"
INDIC = {"hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "ur", "or", "as", "sa", "ne", "kok", "mai", "mni"}
ALERT_MISS_RATE = 0.02         # a top publisher missing more than 2 % of its sitemap
ALERT_YIELD_DROP = 0.5         # a domain's stored count halving against its last measurement
TOP_N = 100


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def host_key(url_or_host: str) -> str:
    """'https://www.Jagran.com/x' / 'm.jagran.com' → 'jagran.com' (comparison key, not registered domain)."""
    h = urlparse(url_or_host).netloc if "://" in url_or_host else url_or_host
    h = h.lower().split(":")[0]
    for p in ("www.", "m.", "amp."):
        if h.startswith(p):
            h = h[len(p):]
    return h


def lincoln_petersen(ours: int, reference: int, both: int) -> float | None:
    """Chapman's bias-corrected Lincoln–Petersen estimate of the population size."""
    if reference <= 0 or ours <= 0:
        return None
    return (ours + 1) * (reference + 1) / (both + 1) - 1


def miss_row(day: date, domain: str, reference: str, ref_urls: set[str], stored: set[str],
             ours_total: int | None = None) -> dict:
    missing = sorted(ref_urls - stored)
    both = len(ref_urls) - len(missing)
    row = {"day": day.isoformat(), "domain": domain, "reference": reference,
           "ref_count": len(ref_urls), "ours": both, "missing": len(missing),
           "miss_rate": round(len(missing) / len(ref_urls), 4) if ref_urls else None,
           "est_universe": None, "sample_missing": missing[:10]}
    if reference == "gdelt" and ours_total is not None:
        est = lincoln_petersen(ours_total, len(ref_urls), both)
        row["est_universe"] = round(est, 1) if est is not None else None
        row["ours"] = ours_total
    return row


def gkg_urls(lines, wanted_hosts: set[str], tld_in: bool = True):
    """(host, url) for GKG rows whose document URL is on an Indian host (field 5, V2DOCUMENTIDENTIFIER)."""
    for line in lines:
        parts = line.split("\t", 5)
        if len(parts) < 5:
            continue
        url = parts[4].strip()
        if not url.startswith("http"):
            continue
        h = host_key(url)
        if h in wanted_hosts or (tld_in and (h.endswith(".in") or h.endswith(".co.in"))):
            yield h, url


def alerts(rows: list[dict], previous_yield: dict[str, int], top_domains: set[str]) -> list[str]:
    out = []
    for r in rows:
        if r["reference"] == "sitemap" and r["domain"] in top_domains and r["ref_count"] >= 20 \
                and (r["miss_rate"] or 0) > ALERT_MISS_RATE:
            out.append(f"{r['domain']}: missed {r['missing']} of {r['ref_count']} sitemap URLs "
                       f"({r['miss_rate']:.1%}), e.g. {r['sample_missing'][:2]}")
        if r["reference"] == "ingest":
            prev = previous_yield.get(r["domain"])
            if prev and prev >= 20 and r["ours"] < prev * (1 - ALERT_YIELD_DROP):
                out.append(f"{r['domain']}: stored {r['ours']} vs {prev} at the last measurement — dead source?")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def _stored_urls(urls: set[str]) -> set[str]:
    """The subset of `urls` the database already holds (ingestion's own hashing)."""
    from pipeline import db
    from pipeline.dedup import url_hash
    by_hash = defaultdict(set)
    for u in urls:
        by_hash[url_hash(u)].add(u)
    hashes = list(by_hash)
    found: set[int] = set()
    for i in range(0, len(hashes), 1500):
        found |= db.existing_hashes(hashes[i:i + 1500])
    return {u for h in found for u in by_hash.get(h, ())}


def _feeds() -> list[dict]:
    from pipeline.db import get_client
    rows, start = [], 0
    while True:
        page = (get_client().table("feeds").select("feed_url, domain, feed_type, is_active, country_code, language_code")
                .range(start, start + 999).execute().data or [])
        rows.extend(page)
        if len(page) < 1000:
            return rows
        start += 1000


def sitemap_reference(feed_url: str, day: date) -> set[str]:
    from pipeline.poller import _download_feed
    from pipeline.sitemap import SITEMAP_MAX_BYTES, fetch_news_sitemap
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    entries, _, err = fetch_news_sitemap(feed_url, lambda u: _download_feed(u, max_bytes=SITEMAP_MAX_BYTES, deadline=90),
                                         since=start, now=start + timedelta(days=1))
    if err:
        log.warning("sitemap %s: %s", feed_url, err)
    return {e["link"] for e in entries}


def gdelt_day(day: date, wanted_hosts: set[str], every: int = 2) -> dict[str, set[str]]:
    """GDELT English + translingual GKG URLs of Indian hosts for `day` (every n-th 15-min file)."""
    import httpx
    slots = [datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
             for i in range(96)][::max(every, 1)]
    names = [f"{t:%Y%m%d%H%M%S}.gkg.csv.zip" for t in slots] + \
            [f"{t:%Y%m%d%H%M%S}.translation.gkg.csv.zip" for t in slots]

    def one(name: str) -> list[tuple[str, str]]:
        try:
            r = httpx.get(f"{GDELT_BASE}/{name}", timeout=120)
            if r.status_code != 200:
                return []
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
            return list(gkg_urls(raw.splitlines(), wanted_hosts))
        except Exception as e:
            log.debug("GDELT %s: %s", name, e)
            return []

    out: dict[str, set[str]] = defaultdict(set)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pairs in ex.map(one, names):
            for h, u in pairs:
                out[h].add(u)
    log.info("GDELT: %d files, %d Indian URLs on %d hosts", len(names), sum(map(len, out.values())), len(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", help="YYYY-MM-DD (default: yesterday UTC)")
    ap.add_argument("--no-gdelt", action="store_true")
    ap.add_argument("--gdelt-every", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true", help="measure, do not write coverage_daily")
    ap.add_argument("--summary", help="write a markdown summary here (GITHUB_STEP_SUMMARY)")
    ap.add_argument("--alerts", help="write alert lines here (one per line)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    from pipeline.db import get_client
    day = date.fromisoformat(args.day) if args.day else (datetime.now(timezone.utc) - timedelta(days=1)).date()
    feeds = _feeds()
    sitemaps = [f for f in feeds if f.get("is_active") and f.get("feed_type") == "news_sitemap"]
    indian_hosts = {host_key(f["domain"]) for f in feeds if f.get("domain") and (
        (f.get("country_code") or "").upper() in ("IN", "IND")
        or (f.get("language_code") or "").split("-")[0].lower() in INDIC)}
    known_hosts = {host_key(f["domain"]) for f in feeds if f.get("domain") and f.get("is_active")}
    ours_by_host: dict[str, int] = defaultdict(int)
    for r in get_client().rpc("wizer_domain_counts", {"p_day": day.isoformat()}).execute().data or []:
        ours_by_host[host_key(r["domain"])] += int(r["n"])

    rows: list[dict] = []
    # 1. sitemaps
    with ThreadPoolExecutor(max_workers=8) as ex:
        refs = list(ex.map(lambda f: (f, sitemap_reference(f["feed_url"], day)), sitemaps))
    for f, urls in refs:
        if urls:
            rows.append(miss_row(day, host_key(f["domain"]), "sitemap", urls, _stored_urls(urls)))
    # merge several sitemaps of one domain into one row (keep the largest reference)
    merged: dict[str, dict] = {}
    for r in rows:
        if r["domain"] not in merged or r["ref_count"] > merged[r["domain"]]["ref_count"]:
            merged[r["domain"]] = r
    rows = list(merged.values())

    # 2. GDELT
    discover: list[tuple[str, int]] = []
    if not args.no_gdelt:
        g = gdelt_day(day, indian_hosts, args.gdelt_every)
        for h, urls in g.items():
            if h in known_hosts:
                rows.append(miss_row(day, h, "gdelt", urls, _stored_urls(urls), ours_total=ours_by_host.get(h, 0)))
            elif len(urls) >= 5:
                discover.append((h, len(urls)))
        discover.sort(key=lambda x: -x[1])

    # 3. yield
    for h, n in ours_by_host.items():
        rows.append({"day": day.isoformat(), "domain": h, "reference": "ingest", "ref_count": n, "ours": n,
                     "missing": 0, "miss_rate": None, "est_universe": None, "sample_missing": []})

    prev_rows = (get_client().table("coverage_daily").select("domain, ours, day").eq("reference", "ingest")
                 .lt("day", day.isoformat()).order("day", desc=True).limit(5000).execute().data or [])
    previous_yield: dict[str, int] = {}
    for r in prev_rows:
        previous_yield.setdefault(r["domain"], int(r["ours"]))
    top = {r["domain"] for r in sorted((r for r in rows if r["reference"] == "sitemap"),
                                       key=lambda r: -r["ref_count"])[:TOP_N]}
    alert_lines = alerts(rows, previous_yield, top)

    sm = [r for r in rows if r["reference"] == "sitemap"]
    gd = [r for r in rows if r["reference"] == "gdelt"]
    sm_ref, sm_miss = sum(r["ref_count"] for r in sm), sum(r["missing"] for r in sm)
    gd_ref, gd_miss = sum(r["ref_count"] for r in gd), sum(r["missing"] for r in gd)
    lines = [f"## WIZER coverage — {day}",
             f"- **Sitemaps:** {len(sm)} publishers, {sm_ref:,} URLs listed, {sm_miss:,} missed "
             f"(**{(sm_miss / sm_ref if sm_ref else 0):.2%}**)",
             f"- **GDELT sample:** {len(gd)} Indian domains, {gd_ref:,} URLs, {gd_miss:,} missed "
             f"(**{(gd_miss / gd_ref if gd_ref else 0):.2%}**)",
             f"- **Stored:** {sum(ours_by_host.values()):,} articles from {len(ours_by_host)} domains",
             "", "### Worst sitemap misses"]
    for r in sorted(sm, key=lambda r: -r["missing"])[:15]:
        if r["missing"]:
            lines.append(f"- {r['domain']}: {r['missing']}/{r['ref_count']} ({r['miss_rate']:.1%})")
    if discover:
        lines += ["", "### Indian hosts GDELT saw that have no source"]
        lines += [f"- {h}: {n} URLs" for h, n in discover[:25]]
    if alert_lines:
        lines += ["", "### Alerts"] + [f"- {a}" for a in alert_lines]
    report = "\n".join(lines)
    print(report)
    if args.summary:
        Path(args.summary).write_text(report + "\n", encoding="utf-8")
    if args.alerts:
        Path(args.alerts).write_text("\n".join(alert_lines), encoding="utf-8")

    if not args.dry_run and rows:
        payload = [dict(r, sample_missing=r["sample_missing"]) for r in rows]
        for i in range(0, len(payload), 500):
            get_client().table("coverage_daily").upsert(payload[i:i + 500],
                                                         on_conflict="day,domain,reference").execute()
        log.info("Wrote %d coverage rows for %s", len(payload), day)
    return 0


if __name__ == "__main__":
    sys.exit(main())
