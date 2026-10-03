#!/usr/bin/env python3
"""
archive.py
══════════
Article archive: keep a hot window of recent articles in Postgres, move older
ones to Parquet in a private Supabase Storage bucket — deleting a row only
after its Parquet copy has been re-downloaded and verified.

  run      archive → verify → prune every day older than --hot-days (default 30)
  status   per-day archive log (rows, size, status)
  fetch    download archived Parquet for a date range to a local folder

USAGE:
  python archive.py run --dry-run              # what would be archived
  python archive.py run --no-prune             # export + verify only, delete nothing
  python archive.py run                        # export + verify + prune
  python archive.py status
  python archive.py fetch --start 2026-04-09 --end 2026-04-30 --out ./archive

READING THE ARCHIVE (after `fetch`):
  import duckdb
  duckdb.sql("SELECT domain, count(*) FROM 'archive/articles/2026/04/*/*.parquet' GROUP BY 1")

Design and safety model: docs/archive_migration.sql and archiver/runner.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger("archive")
BUCKET = os.getenv("ARCHIVE_BUCKET", "article-archive")
HOT_DAYS = int(os.getenv("ARCHIVE_HOT_DAYS", "30"))


def _store():
    from archiver.store import SupabaseStore
    from enrichment.db import get_client
    return SupabaseStore(get_client(), BUCKET)


def cmd_run(args) -> int:
    from archiver import db
    from archiver.runner import run_archive
    report = run_archive(db, _store(), BUCKET, hot_days=args.hot_days, prune=not args.no_prune,
                         dry_run=args.dry_run, time_budget_minutes=args.time_budget,
                         max_days=args.max_days)
    print(f"days seen {report.days_seen} | exported {report.days_exported} | verified "
          f"{report.days_verified} | pruned {report.days_pruned} | articles archived "
          f"{report.articles_archived:,} | rows deleted {report.rows_pruned:,} | uploaded "
          f"{report.bytes_uploaded / 1e6:,.1f} MB | {report.stop_reason}")
    for f in report.failures:
        print("  FAILED", f)
    return 1 if report.failures else 0


def cmd_status(args) -> int:
    from archiver import db
    rows = db.all_logs()
    total = sum(r["total_bytes"] for r in rows)
    print(f"{len(rows)} days logged, {sum(r['article_rows'] for r in rows):,} articles, "
          f"{total / 1e9:.2f} GB in bucket '{BUCKET}'")
    for r in rows[-args.last:]:
        print(f"  {r['day']}  {r['status']:<9} {r['article_rows']:>7,} articles "
              f"{r['total_bytes'] / 1e6:>8.1f} MB  pruned {r['pruned_rows']:>7,}")
    return 0


def cmd_fetch(args) -> int:
    from archiver import db
    store, out = _store(), Path(args.out)
    logs = {r["day"]: r for r in db.all_logs()}
    day, end, n = date.fromisoformat(args.start), date.fromisoformat(args.end), 0
    while day <= end:
        row = logs.get(day.isoformat()) and db.get_log(day)
        for obj in (row or {}).get("objects", []):
            if args.kind != "all" and obj["kind"] != args.kind:
                continue
            target = out / obj["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(store.get(obj["path"]))
            n += 1
        day += timedelta(days=1)
    print(f"downloaded {n} files to {out}")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", "-v", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--hot-days", type=int, default=HOT_DAYS)
    r.add_argument("--no-prune", action="store_true", help="export + verify only")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--time-budget", type=float, default=0, help="minutes; 0 = no limit")
    r.add_argument("--max-days", type=int, default=None)
    s = sub.add_parser("status")
    s.add_argument("--last", type=int, default=40)
    f = sub.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--out", default="archive")
    f.add_argument("--kind", choices=["articles", "entities", "clusters", "all"], default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%S", stream=sys.stdout)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return {"run": cmd_run, "status": cmd_status, "fetch": cmd_fetch}[args.cmd](args)
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        log.exception("archive.py %s crashed: %s", args.cmd, e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
