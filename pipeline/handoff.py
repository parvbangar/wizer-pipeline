"""
pipeline/handoff.py
═══════════════════
The ingest → clustering/enrichment hand-off: newly inserted articles WITH their
crawled body, written by the ingest job as one gzipped JSON-lines file and
uploaded as a GitHub Actions artifact (.github/workflows/ingest-*.yml).

WHY A FILE AND NOT THE DATABASE:
  full_text was ~60 % of the database (6 of 13 GB on 2026-10-05) and reading
  it back for enrichment cost the Micro instance disk-I/O credits it does not
  have. The body is only needed once — to enrich the article — so it travels
  next to the database instead of through it. The articles row (title, URL,
  description, metadata) is still written by ingestion as before; only
  full_text is left out (pipeline.config.STORE_FULL_TEXT).

  If an artifact is lost, nothing is lost for good: the article row exists
  unenriched, and the enrichment sweeper (enrich.py, queue v2) enriches it
  from title + description.

FORMAT: one JSON object per line, keys = HANDOFF_FIELDS. Readers shard by id
(`shard_of`), so N parallel enrichment runners split a file without talking
to each other.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

HANDOFF_FIELDS = (
    "id", "feed_id", "url", "title", "description", "full_text", "top_image_url",
    "published_at", "crawled_at", "domain", "language_code", "country_code",
    "iab_tier1", "iab_tier2", "is_crawled", "is_duplicate",
)


def _hash_key(value) -> int | None:
    """articles.url_hash is char(32) holding a space-padded integer in production."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


class Handoff:
    """Collects the articles a run inserted, matched to their crawled bodies."""

    def __init__(self, max_body_chars: int | None = None) -> None:
        self.records: list[dict] = []
        self.unmatched = 0
        self._max_body = max_body_chars

    def __len__(self) -> int:
        return len(self.records)

    def add_inserted(self, inserted_rows: Iterable[dict], articles: Iterable) -> None:
        """
        inserted_rows: rows the database returned for this insert (with ids)
        articles:      the CrawledArticle objects they were built from
        """
        by_hash = {a.url_hash: a for a in articles}
        for row in inserted_rows:
            art = by_hash.get(_hash_key(row.get("url_hash")))
            if art is None or row.get("id") is None:
                self.unmatched += 1
                continue
            rec = {k: row.get(k) for k in HANDOFF_FIELDS}
            body = art.full_text or ""
            rec["full_text"] = body[: self._max_body] if self._max_body else body
            self.records.append(rec)

    def write(self, path: str | Path) -> int:
        """Write all records (an empty file when there are none, so the artifact always exists)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str))
                f.write("\n")
        tmp.replace(path)
        if self.unmatched:
            log.warning("Hand-off: %d inserted rows had no matching crawl result", self.unmatched)
        log.info("Hand-off: wrote %d articles to %s", len(self.records), path)
        return len(self.records)


def read(paths: Iterable[str | Path]) -> list[dict]:
    """All records from one or more hand-off files (missing files are skipped)."""
    out: list[dict] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            log.warning("Hand-off file %s not found — skipped", p)
            continue
        with gzip.open(p, "rt", encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def find_files(root: str | Path) -> list[Path]:
    """Every hand-off file under `root` (download-artifact may nest directories)."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.jsonl.gz"))


def shard_of(records: list[dict], index: int, count: int) -> list[dict]:
    """The records runner `index` of `count` processes (stable by article id)."""
    if count <= 1:
        return list(records)
    return [r for r in records if int(r["id"]) % count == index]
