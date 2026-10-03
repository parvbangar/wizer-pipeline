"""
archiver/store.py
═════════════════
The archive's object store: a PRIVATE Supabase Storage bucket.

Layout (one UTC day of articles.crawled_at per folder):
  articles/YYYY/MM/DD/part-0001.parquet   … ≤ PART_ROWS articles per part
  entities/YYYY/MM/DD/part-0001.parquet   … named entities of those articles
  clusters/YYYY/MM/DD.parquet             … story clusters they belong to

Parts stay ~15 MB: Supabase Storage's default per-file upload limit is 50 MB,
and a full day (~50K articles) is ~70 MB.
"""

from __future__ import annotations

import logging
from datetime import date

log = logging.getLogger(__name__)

PARQUET_MIME = "application/vnd.apache.parquet"


def day_prefix(kind: str, day: date) -> str:
    return f"{kind}/{day:%Y/%m/%d}"


def part_path(kind: str, day: date, n: int) -> str:
    return f"{day_prefix(kind, day)}/part-{n:04d}.parquet"


def clusters_path(day: date) -> str:
    return f"clusters/{day:%Y/%m/%d}.parquet"


class SupabaseStore:
    """Thin wrapper over supabase-py Storage — the only module that talks to it."""

    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def ensure_bucket(self) -> None:
        """Create the bucket (private) if it does not exist yet."""
        try:
            self.client.storage.get_bucket(self.bucket)
        except Exception:
            log.info("Creating private storage bucket '%s'", self.bucket)
            self.client.storage.create_bucket(self.bucket, options={"public": False})

    def put(self, path: str, data: bytes) -> None:
        self.client.storage.from_(self.bucket).upload(
            path, data, {"content-type": PARQUET_MIME, "upsert": "true"})

    def get(self, path: str) -> bytes:
        return self.client.storage.from_(self.bucket).download(path)

    def remove(self, paths: list[str]) -> None:
        if paths:
            self.client.storage.from_(self.bucket).remove(paths)

    def list(self, prefix: str) -> list[str]:
        out, offset = [], 0
        while True:
            page = self.client.storage.from_(self.bucket).list(
                prefix, {"limit": 1000, "offset": offset, "sortBy": {"column": "name", "order": "asc"}})
            out.extend(f"{prefix}/{o['name']}" for o in page)
            if len(page) < 1000:
                return out
            offset += 1000
