#!/usr/bin/env python3
"""
push_feeds.py
═════════════
Reads a CSV of RSS feeds, deduplicates them against each other and against
the live Supabase feeds table, then inserts only genuinely new feeds.

DEDUPLICATION — THREE LAYERS
─────────────────────────────
Layer 1 — Within-CSV exact duplicates (same feed_url string)
  Catches copy-paste / merge errors in the source file.

Layer 2 — Within-CSV normalised URL duplicates
  Both feed_url and final_url are normalised:
    • lowercase domain, strip www.
    • remove tracking query params (utm_*, fbclid, gclid …)
    • collapse // in path, strip trailing slash
    • sort remaining query params
  Two rows that normalise to the same final_url are treated as the same feed.
  When duplicates compete, the higher-quality row wins (freshness_score →
  days_since_update → priority_score → metadata_confidence).

Layer 3 — Against Supabase
  All (feed_url, final_url) values from the live feeds table are loaded,
  normalised, and stored in a set.  Any CSV row whose normalised URL is
  already in that set is skipped.

USAGE
─────
    # Dry run — shows counts, inserts nothing
    python push_feeds.py --csv /path/to/feeds.csv --dry-run

    # Live run — inserts new feeds in batches of 500
    python push_feeds.py --csv /path/to/feeds.csv

    # Custom batch size
    python push_feeds.py --csv /path/to/feeds.csv --batch-size 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse, urlencode, parse_qsl

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env.local")
load_dotenv()  # fallback to .env if .env.local not found

log = logging.getLogger("push_feeds")

# ─────────────────────────────────────────────────────────────────────────────
# URL NORMALISATION  (mirrors pipeline/dedup.py — kept self-contained so this
#                     script can be run from anywhere without the full package)
# ─────────────────────────────────────────────────────────────────────────────

_STRIP_PARAMS: frozenset = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term",
    "utm_content", "utm_id", "utm_referrer",
    "gclid", "gclsrc", "gbraid", "wbraid",
    "fbclid", "fb_action_ids", "fb_source",
    "twclid", "msclkid",
    "hsa_acc", "hsa_cam", "hsa_grp", "hsa_ad",
    "mc_cid", "mc_eid",
    "_ga", "igshid", "s_cid", "ncid", "cmpid", "mbid",
    "ref", "referrer", "source", "origin",
    "sid", "session_id", "share", "sharesource",
})


def _normalise(url: str) -> str:
    url = url.strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        p = urlparse(url)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/{2,}", "/", p.path)
        if len(path) > 1:
            path = path.rstrip("/")
        clean = sorted([
            (k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
            if k.lower() not in _STRIP_PARAMS
        ])
        return urlunparse((p.scheme.lower(), host, path, "", urlencode(clean), ""))
    except Exception:
        return url


# ─────────────────────────────────────────────────────────────────────────────
# VALUE PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def _bool(val: str) -> bool:
    return str(val).lower().strip() in ("true", "1", "yes")


def _float(val: str) -> float | None:
    try:
        return float(val) if val and val.strip() else None
    except (ValueError, TypeError):
        return None


def _str(val: str) -> str | None:
    v = (val or "").strip()
    return v if v else None


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY KEY — used to pick the best row when two CSV rows are duplicates
# ─────────────────────────────────────────────────────────────────────────────

def _quality_key(row: dict) -> tuple:
    """
    Higher tuple = better quality row.
    Priority order:
      1. freshness_score (0–100) — higher is fresher
      2. days_since_update       — lower is more recent (negated so higher wins)
      3. priority_score          — higher is more important
      4. metadata_confidence     — higher = more reliable metadata
    """
    return (
        _float(row.get("freshness_score", "")) or 0,
        -(_float(row.get("days_since_update", "")) or 9999),
        _float(row.get("priority_score", "")) or 0,
        _float(row.get("metadata_confidence", "")) or 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSV → FEED ROW MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Full set of columns we want to push.  The script will auto-detect which of
# these actually exist in the Supabase table and only include those.
_ALL_MAPPED: dict[str, str] = {
    # CSV column          : feeds table column
    "feed_url":             "feed_url",
    "final_url":            "final_url",
    "domain":               "domain",
    "publisher_name":       "publisher_name",
    "publisher_type":       "publisher_type",
    "country_code":         "country_code",
    "country_name":         "country_name",
    "language_code":        "language_code",
    "language_name":        "language_name",
    "iab_tier1":            "iab_tier1",
    "iab_tier2":            "iab_tier2",
    "update_cadence":       "update_cadence",
    "validation_tier":      "validation_tier",
    "political_lean":       "political_lean",
    "audience_type":        "audience_type",
    "feed_format":          "feed_format",
    "feed_type":            "feed_type",
}

_BOOL_COLS = {"has_paywall", "is_satire"}
_FLOAT_COLS = {"priority_score", "freshness_score", "avg_items_per_day", "metadata_confidence"}

# These are always set by the script (not from CSV)
_PIPELINE_DEFAULTS = {
    "is_active":     True,
    "fail_count":    0,
    "articles_found": 0,
}


def _csv_row_to_feed(row: dict, table_cols: set[str]) -> dict:
    """
    Map one CSV row to a dict ready for Supabase insert.
    Only includes columns that actually exist in the table.
    """
    feed: dict = {}

    # String columns
    for csv_col, db_col in _ALL_MAPPED.items():
        if db_col not in table_cols:
            continue
        val = _str(row.get(csv_col, ""))
        if val:
            feed[db_col] = val

    # Boolean columns
    for col in _BOOL_COLS:
        if col in table_cols:
            feed[col] = _bool(row.get(col, "false"))

    # Numeric columns
    for col in _FLOAT_COLS:
        if col in table_cols:
            v = _float(row.get(col, ""))
            if v is not None:
                feed[col] = v

    # Pipeline defaults (always included if column exists)
    for col, default in _PIPELINE_DEFAULTS.items():
        if col in table_cols:
            feed[col] = default

    # Ensure feed_url is always present (required)
    if "feed_url" not in feed:
        feed["feed_url"] = row.get("feed_url", "").strip()

    return feed


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        log.error(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY. "
            "Set them in your .env file or environment."
        )
        sys.exit(1)
    from supabase import create_client
    return create_client(url, key)


def _discover_table_columns(client) -> set[str]:
    """
    Ask Supabase what columns the feeds table actually has by reading one row.
    Falls back to a guaranteed-safe minimal set if the table is empty.
    """
    try:
        resp = client.table("feeds").select("*").limit(1).execute()
        if resp.data:
            cols = set(resp.data[0].keys())
            log.info("Discovered %d columns in feeds table: %s", len(cols), sorted(cols))
            return cols
    except Exception as e:
        log.warning("Could not auto-discover columns: %s — using minimal safe set", e)

    # Minimal safe set (columns confirmed by config.py)
    return {
        "feed_url", "final_url", "domain", "publisher_name",
        "country_code", "language_code", "iab_tier1", "iab_tier2",
        "has_paywall", "update_cadence", "validation_tier", "priority_score",
        "is_active", "fail_count", "articles_found",
    }


def _load_existing_normalised(client) -> set[str]:
    """
    Load all normalised feed URLs already in Supabase.
    Uses limit=1000 pages (Supabase PostgREST default max) and paginates
    until fewer rows than page_size are returned.
    Returns a set of normalised URL strings for O(1) lookup.
    """
    existing: set[str] = set()
    page_size = 1000   # PostgREST hard cap per request
    offset = 0
    total_rows = 0

    log.info("Loading existing feeds from Supabase...")
    while True:
        resp = (
            client.table("feeds")
            .select("feed_url, final_url")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        total_rows += len(rows)
        for row in rows:
            for col in ("feed_url", "final_url"):
                url = (row.get(col) or "").strip()
                if url:
                    existing.add(_normalise(url))
        if len(rows) < page_size:
            break
        offset += page_size

    log.info("Loaded %d normalised URLs from %d existing feed rows",
             len(existing), total_rows)
    return existing


def _insert_batch(client, batch: list[dict], dry_run: bool) -> tuple[int, int]:
    """
    Upsert a batch of feed rows, ignoring conflicts on feed_url.
    Returns (inserted, failed).  No row-by-row fallback needed because
    ON CONFLICT DO NOTHING handles duplicates at the DB level in bulk.
    """
    if dry_run:
        return len(batch), 0

    try:
        resp = (
            client.table("feeds")
            .upsert(batch, on_conflict="feed_url", ignore_duplicates=True)
            .execute()
        )
        return len(resp.data or []), 0
    except Exception as e:
        log.error("Batch upsert failed: %s", e)
        return 0, len(batch)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Push new RSS feeds from CSV to Supabase")
    parser.add_argument("--csv",        required=True,  help="Path to the feeds CSV file")
    parser.add_argument("--dry-run",    action="store_true", help="Count and report only, insert nothing")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per Supabase insert call (default 500)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    # ── STEP 1: Read CSV ─────────────────────────────────────────────────────
    log.info("Reading %s …", csv_path)
    raw_rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
    log.info("  %d rows loaded from CSV", len(raw_rows))

    # ── STEP 2: Layer 1 dedup — exact feed_url duplicates within CSV ─────────
    seen_exact: set[str] = set()
    after_exact: list[dict] = []
    exact_dups = 0
    for row in raw_rows:
        fu = row.get("feed_url", "").strip()
        if fu in seen_exact:
            exact_dups += 1
        else:
            seen_exact.add(fu)
            after_exact.append(row)
    log.info("  After exact feed_url dedup: %d rows (%d exact duplicates dropped)",
             len(after_exact), exact_dups)

    # ── STEP 3: Layer 2 dedup — normalised final_url within CSV ─────────────
    # For each normalised final_url, keep the highest-quality row
    best: dict[str, dict] = {}
    no_url = 0
    for row in after_exact:
        final = (row.get("final_url") or row.get("feed_url") or "").strip()
        norm  = _normalise(final)
        if not norm:
            no_url += 1
            continue
        if norm not in best or _quality_key(row) > _quality_key(best[norm]):
            best[norm] = row

    deduped_rows = list(best.values())
    norm_dups = len(after_exact) - len(deduped_rows) - no_url
    log.info("  After normalised URL dedup: %d rows (%d duplicates dropped, %d had no URL)",
             len(deduped_rows), norm_dups, no_url)

    # ── STEP 4: Connect to Supabase ──────────────────────────────────────────
    log.info("Connecting to Supabase …")
    client      = _get_client()
    table_cols  = _discover_table_columns(client)
    existing    = _load_existing_normalised(client)
    log.info("  %d normalised URLs already in Supabase", len(existing))

    # ── STEP 5: Layer 3 dedup — filter against existing Supabase rows ────────
    new_rows: list[dict] = []
    already_exists = 0
    for norm_url, row in best.items():
        # Check both the normalised final_url AND normalised feed_url
        feed_norm = _normalise(row.get("feed_url", ""))
        if norm_url in existing or feed_norm in existing:
            already_exists += 1
        else:
            new_rows.append(row)

    log.info("  Already in Supabase: %d | Genuinely new: %d",
             already_exists, len(new_rows))

    # ── STEP 6: Summary before insert ────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  CSV rows total:              {len(raw_rows):>8,}")
    print(f"  Exact feed_url duplicates:   {exact_dups:>8,}")
    print(f"  Normalised URL duplicates:   {norm_dups:>8,}")
    print(f"  Already in Supabase:         {already_exists:>8,}")
    print(f"  ─────────────────────────────────────")
    print(f"  New feeds to insert:         {len(new_rows):>8,}")
    if args.dry_run:
        print(f"  DRY RUN — nothing inserted")
    print("═" * 60)
    print()

    if not new_rows:
        log.info("Nothing to insert. Exiting.")
        return

    if args.dry_run:
        log.info("Dry run complete. Re-run without --dry-run to insert.")
        return

    # ── STEP 7: Insert in batches ─────────────────────────────────────────────
    total_inserted = 0
    total_failed   = 0
    batch_count    = 0

    for i in range(0, len(new_rows), args.batch_size):
        batch_raw  = new_rows[i : i + args.batch_size]
        batch_feed = [_csv_row_to_feed(r, table_cols) for r in batch_raw]
        # Filter out any rows that ended up with no feed_url (safety check)
        batch_feed = [r for r in batch_feed if r.get("feed_url")]

        inserted, failed = _insert_batch(client, batch_feed, dry_run=False)
        total_inserted += inserted
        total_failed   += failed
        batch_count    += 1

        log.info(
            "Batch %d/%d — inserted %d, failed %d  (total so far: %d)",
            batch_count,
            -(-len(new_rows) // args.batch_size),   # ceiling division
            inserted, failed,
            total_inserted,
        )

    # ── STEP 8: Final report ─────────────────────────────────────────────────
    print()
    print("═" * 60)
    print(f"  Successfully inserted:  {total_inserted:>8,}")
    print(f"  Failed:                 {total_failed:>8,}")
    print("═" * 60)

    if total_failed > 0:
        log.warning("%d rows failed to insert — check logs above for details", total_failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
