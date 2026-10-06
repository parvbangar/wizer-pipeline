"""
enrichment/body_store.py
════════════════════════
The permanent home of article BODIES: gzipped JSON-lines files in the Supabase
Storage bucket `article-bodies`, one or more per processing runner.

WHY NOT POSTGRES:
  Bodies were ~60 % of the database (6 of 13 GB on 2026-10-05) and the Micro
  instance's disk-I/O budget could not afford storing or reading them.
  Storage is object storage — uploads do not touch the database's disk or CPU
  beyond one small storage.objects row per file. Nothing needs bodies back in
  real time (enrichment receives them in the hand-off), so they are written
  once and read only for research, re-enrichment or model training.

LAYOUT:  article-bodies/YYYY/MM/DD/<tag>-<seq>.jsonl.gz   (UTC date of writing)
RECORD:  {"id", "url", "domain", "language_code", "title", "published_at",
          "crawled_at", "crawl_strategy", "full_text"}
  Find an article's body by its id: the article row's crawled_at gives the day.

A failed upload is logged and does not fail the run: the same bodies are still
in the 7-day hand-off artifact and on the runner's disk for the job's lifetime.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

BUCKET = os.getenv("BODY_BUCKET", "article-bodies")
FLUSH_EVERY = int(os.getenv("BODY_FLUSH_EVERY", "500"))
_FIELDS = ("id", "url", "domain", "language_code", "title", "published_at",
           "crawled_at", "crawl_strategy", "full_text")


class BodyStore:
    def __init__(self, tag: str, client_factory=None, enabled: bool = True) -> None:
        self.tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in tag) or "run"
        self._client_factory = client_factory
        self.enabled = enabled
        self.pending: list[dict] = []
        self.seq = 0
        self.uploaded = 0
        self.failed = 0
        self._bucket_checked = False

    def add(self, record: dict) -> None:
        if not self.enabled or not (record.get("full_text") or "").strip():
            return
        self.pending.append({k: record.get(k) for k in _FIELDS})
        if len(self.pending) >= FLUSH_EVERY:
            self.flush()

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from enrichment.db import get_client
        return get_client()

    def _ensure_bucket(self, storage) -> None:
        if self._bucket_checked:
            return
        try:
            names = {b.name if hasattr(b, "name") else b.get("name") for b in storage.list_buckets()}
            if BUCKET not in names:
                storage.create_bucket(BUCKET, options={"public": False})
                log.info("Created Storage bucket %s", BUCKET)
        except Exception as e:                 # creation races / permissions: the upload will tell
            log.debug("bucket check for %s: %s", BUCKET, e)
        self._bucket_checked = True

    def flush(self) -> int:
        """Upload what is pending as one file; returns the number of bodies uploaded."""
        if not self.pending:
            return 0
        batch, self.pending = self.pending, []
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
            for rec in batch:
                gz.write(json.dumps(rec, ensure_ascii=False, default=str).encode("utf-8"))
                gz.write(b"\n")
        now = datetime.now(timezone.utc)
        self.seq += 1
        path = f"{now:%Y/%m/%d}/{self.tag}-{now:%H%M%S}-{self.seq:03d}.jsonl.gz"
        try:
            storage = self._client().storage
            self._ensure_bucket(storage)
            storage.from_(BUCKET).upload(path, buf.getvalue(),
                                         {"content-type": "application/gzip", "upsert": "true"})
            self.uploaded += len(batch)
            log.info("Body store: %d bodies → %s/%s (%.0f KB)", len(batch), BUCKET, path,
                     len(buf.getvalue()) / 1024)
            return len(batch)
        except Exception as e:
            self.failed += len(batch)
            log.error("Body store upload of %d bodies failed (still in the hand-off artifact): %s",
                      len(batch), e)
            return 0
