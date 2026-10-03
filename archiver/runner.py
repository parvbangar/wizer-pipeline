"""
archiver/runner.py
══════════════════
Archive old articles to Parquet in Supabase Storage, verify, then prune.

FOR EACH DAY older than the hot window (oldest first), resumably:

  1. EXPORT   page through the day's articles (1,000 rows per API call), write
              Parquet parts of ≤ PART_ROWS rows, upload each part; same for the
              day's entities and its story clusters. If the day's row count
              changed while exporting, stop — the next run redoes the day.
  2. LOG      article_archive_log row: status 'uploaded', objects with row
              counts, byte sizes and SHA-256 checksums.
  3. VERIFY   download every object again: checksum and Parquet row count must
              match, the article parts must hold exactly the day's ids, and the
              table must still hold exactly that many rows → status 'verified'.
  4. PRUNE    wizer_prune_archived_day() in chunks. The SQL function itself
              refuses unless the day is verified, outside the hot window, and
              unchanged since export — so even a bug here cannot delete an
              unarchived row.

Stale parts from an earlier interrupted attempt are removed after a day is
re-exported, so readers never see duplicate rows.

Every DB call goes through `db` (archiver/db.py in production) and every
object through `store` (archiver/store.SupabaseStore) — tests pass fakes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

from archiver.parquet import parquet_ids, parquet_row_count, sha256, to_parquet_bytes
from archiver.store import clusters_path, day_prefix, part_path

log = logging.getLogger(__name__)

PART_ROWS = 10_000            # ~15 MB per article part (Storage default limit: 50 MB/file)
ENTITY_PART_ROWS = 100_000    # entities are ~100 bytes each
PAGE = 1_000                  # PostgREST max rows per response
PRUNE_CHUNK = 5_000           # rows deleted per call — inside the 30 s API statement timeout


class ArchiveError(RuntimeError):
    """A day could not be archived safely; it is left for the next run."""


@dataclass
class ArchiveReport:
    days_seen: int = 0
    days_exported: int = 0
    days_verified: int = 0
    days_pruned: int = 0
    articles_archived: int = 0
    rows_pruned: int = 0
    bytes_uploaded: int = 0
    failures: list[str] = field(default_factory=list)
    stop_reason: str = "done"


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def _upload(store, objects: list, kind: str, path: str, rows: list[dict]) -> None:
    data = to_parquet_bytes(rows)
    store.put(path, data)
    objects.append({"kind": kind, "path": path, "rows": len(rows),
                    "bytes": len(data), "sha256": sha256(data)})


def export_day(day: date, db, store, bucket: str) -> dict:
    """Write and upload one day's Parquet; returns the article_archive_log row."""
    stats = db.day_stats(day)
    objects: list[dict] = []

    # Articles — keyset pages by id, cut into parts.
    part, n, after, exported = [], 0, 0, 0
    while True:
        page = db.article_page(day, after, PAGE)
        if not page:
            break
        part.extend(page)
        after = page[-1]["id"]
        exported += len(page)
        if len(part) >= PART_ROWS:
            n += 1
            _upload(store, objects, "articles", part_path("articles", day, n), part)
            part = []
    if part:
        n += 1
        _upload(store, objects, "articles", part_path("articles", day, n), part)

    if exported != stats["article_rows"]:
        raise ArchiveError(f"{day}: {exported} rows exported but the day holds "
                           f"{stats['article_rows']} — it changed during export; retry next run")

    # Entities of those articles.
    ents, n_e, key, entity_rows = [], 0, "", 0
    while True:
        page = db.entity_page(day, key, PAGE)
        if not page:
            break
        key = page[-1]["page_key"]
        ents.extend({k: v for k, v in r.items() if k != "page_key"} for r in page)
        entity_rows += len(page)
        if len(ents) >= ENTITY_PART_ROWS:
            n_e += 1
            _upload(store, objects, "entities", part_path("entities", day, n_e), ents)
            ents = []
    if ents:
        n_e += 1
        _upload(store, objects, "entities", part_path("entities", day, n_e), ents)

    # Story clusters the day's articles belong to.
    cl = db.clusters(day)
    if cl:
        _upload(store, objects, "clusters", clusters_path(day), cl)

    # Remove leftovers of an earlier, interrupted attempt at this day.
    keep = {o["path"] for o in objects}
    for prefix in (day_prefix("articles", day), day_prefix("entities", day)):
        stale = [p for p in store.list(prefix) if p not in keep]
        if stale:
            store.remove(stale)
            log.info("%s: removed %d stale parts under %s", day, len(stale), prefix)

    row = {
        "day": day.isoformat(), "status": "uploaded", "bucket": bucket,
        "article_rows": exported, "entity_rows": entity_rows, "cluster_rows": len(cl),
        "min_article_id": stats["min_id"], "max_article_id": stats["max_id"],
        "objects": objects, "total_bytes": sum(o["bytes"] for o in objects),
        "verified_at": None, "pruned_rows": 0, "pruned_at": None,
    }
    db.upsert_log(row)
    return row


def verify_day(day: date, row: dict, db, store) -> None:
    """Re-download everything and prove the export is complete; mark verified."""
    ids: list = []
    for obj in row["objects"]:
        data = store.get(obj["path"])
        if sha256(data) != obj["sha256"]:
            raise ArchiveError(f"{day}: checksum mismatch for {obj['path']}")
        if parquet_row_count(data) != obj["rows"]:
            raise ArchiveError(f"{day}: row count mismatch for {obj['path']}")
        if obj["kind"] == "articles":
            ids.extend(parquet_ids(data))
    if len(ids) != row["article_rows"] or len(set(ids)) != len(ids):
        raise ArchiveError(f"{day}: article parts hold {len(ids)} ids "
                           f"({len(set(ids))} distinct), expected {row['article_rows']}")
    if ids and (min(ids) != row["min_article_id"] or max(ids) != row["max_article_id"]):
        raise ArchiveError(f"{day}: archived id range differs from the table's")
    now = db.day_stats(day)
    if now["article_rows"] != row["article_rows"]:
        raise ArchiveError(f"{day}: table now holds {now['article_rows']} rows, "
                           f"archive holds {row['article_rows']} — re-archive")
    db.mark_verified(day)


def prune_day(day: date, db, hot_days: int) -> int:
    """Delete a verified day in chunks; returns rows deleted in this call."""
    total = 0
    while True:
        n = db.prune_day(day, hot_days, PRUNE_CHUNK)
        total += n
        if n < PRUNE_CHUNK:
            return total


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_archive(db, store, bucket: str, hot_days: int = 30, prune: bool = True,
                dry_run: bool = False, time_budget_minutes: float = 0,
                max_days: int | None = None) -> ArchiveReport:
    report = ArchiveReport()
    deadline = time.monotonic() + time_budget_minutes * 60 if time_budget_minutes else None
    if not dry_run:
        store.ensure_bucket()
    days = db.pending_days(hot_days)
    if max_days is not None:
        days = days[:max_days]
    log.info("%d days older than %d days to archive%s", len(days), hot_days,
             " [dry run]" if dry_run else "")

    for day, status in days:
        if deadline and time.monotonic() > deadline:
            report.stop_reason = "time_budget"
            break
        report.days_seen += 1
        if dry_run:
            stats = db.day_stats(day)
            log.info("DRY RUN %s: %d articles (status: %s)", day, stats["article_rows"], status or "new")
            report.articles_archived += stats["article_rows"]
            continue
        try:
            row = db.get_log(day)
            if row is None or row["status"] == "uploaded":
                t0 = time.monotonic()
                row = export_day(day, db, store, bucket)
                report.days_exported += 1
                report.bytes_uploaded += row["total_bytes"]
                verify_day(day, row, db, store)
                report.days_verified += 1
                report.articles_archived += row["article_rows"]
                log.info("%s: archived + verified %d articles, %d entities, %d clusters "
                         "(%.1f MB) in %.0fs", day, row["article_rows"], row["entity_rows"],
                         row["cluster_rows"], row["total_bytes"] / 1e6, time.monotonic() - t0)
            if prune:
                deleted = prune_day(day, db, hot_days)
                report.rows_pruned += deleted
                report.days_pruned += 1
                if deleted:
                    log.info("%s: pruned %d rows from Postgres", day, deleted)
        except Exception as e:
            # One bad day must not block the others; it is retried next run.
            report.failures.append(f"{day}: {e}")
            log.error("%s: %s", day, e)

    log.info("Archive run: %s", report)
    return report
