#!/usr/bin/env python3
"""
tools/archive_local.py
══════════════════════
Run the article archive from a workstation over a DIRECT Postgres connection
(instead of the REST API the scheduled job uses) — for the one-off backlog.

Same code path and the same safety model as archive.py (archiver/runner.py):
export a day → upload → re-download from the BUCKET and verify → guarded
delete through wizer_prune_archived_day(). Differences:
  - direct SQL: no PostgREST JSON overhead and no 30 s API statement timeout,
    so one call can delete a whole day;
  - every Parquet part is ALSO written to a local folder (--local-dir), giving
    an offline copy of the archive alongside the bucket.

Environment:
  WIZER_ARCHIVE_DSN     libpq DSN with write access (e.g. the session pooler)
  SUPABASE_URL, SUPABASE_SERVICE_KEY   for Storage

USAGE:
  python tools/archive_local.py --local-dir D:/03_Data/wizer_archive [--no-prune] [--max-days N]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg                                   # noqa: E402

from archiver import runner                      # noqa: E402
from archiver.store import SupabaseStore         # noqa: E402
from tools.pg_backend import PgArchiveDB         # noqa: E402

log = logging.getLogger("archive_local")


class TeeStore:
    """Writes go to the bucket AND a local mirror; reads/lists come from the bucket,
    so verification always checks the copy the database deletion depends on."""

    def __init__(self, bucket_store: SupabaseStore, local_dir: Path):
        self.bucket = bucket_store
        self.local = local_dir

    def ensure_bucket(self):
        self.bucket.ensure_bucket()
        self.local.mkdir(parents=True, exist_ok=True)

    def put(self, path, data):
        self.bucket.put(path, data)
        target = self.local / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def get(self, path):
        return self.bucket.get(path)

    def list(self, prefix):
        return self.bucket.list(prefix)

    def remove(self, paths):
        self.bucket.remove(paths)
        for p in paths:
            (self.local / p).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--hot-days", type=int, default=30)
    ap.add_argument("--bucket", default=os.getenv("ARCHIVE_BUCKET", "article-archive"))
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--max-days", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from supabase import create_client
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    store = TeeStore(SupabaseStore(client, args.bucket), Path(args.local_dir))

    conn = psycopg.connect(os.environ["WIZER_ARCHIVE_DSN"], connect_timeout=30)
    conn.execute("SET statement_timeout = 0")
    conn.commit()

    runner.PRUNE_CHUNK = 100_000          # direct SQL: a whole day per delete call
    report = runner.run_archive(PgArchiveDB(conn), store, args.bucket, hot_days=args.hot_days,
                                prune=not args.no_prune, max_days=args.max_days)
    print(f"days {report.days_seen} | exported {report.days_exported} | verified {report.days_verified} | "
          f"pruned {report.days_pruned} | archived {report.articles_archived:,} | deleted {report.rows_pruned:,} | "
          f"uploaded {report.bytes_uploaded / 1e6:,.1f} MB | failures {len(report.failures)}")
    for f in report.failures:
        print("  FAILED", f)
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
