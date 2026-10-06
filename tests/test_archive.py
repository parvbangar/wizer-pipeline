"""
tests/test_archive.py
═════════════════════
Offline tests for the article archive (archiver/): Parquet typing, and the
export → verify → prune orchestration against a fake DB and an in-memory
bucket — including every way it must REFUSE to delete.

The SQL guards themselves are tested against real Postgres in
tests/test_clustering_sql.py::TestArchiveSQL.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone

import pyarrow.parquet as pq
import pytest

from archiver import parquet, runner
from archiver.runner import ArchiveError, run_archive

DAY = date(2026, 4, 10)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet
# ─────────────────────────────────────────────────────────────────────────────

class TestParquet:

    def test_types_and_roundtrip(self):
        rows = [
            {"id": 1, "title": "हिंदी शीर्षक", "published_at": "2026-04-10T05:00:00+00:00",
             "keywords": ["a", "b"], "og_tags": {"og:title": "x"}, "is_crawled": True,
             "sentiment_score": 0, "url_hash": "8507220197785563818             ", "story_id": None},
            {"id": 2, "title": "t2", "published_at": None, "keywords": None, "og_tags": None,
             "is_crawled": False, "sentiment_score": -0.25, "url_hash": "1", "story_id": None},
        ]
        data = parquet.to_parquet_bytes(rows)
        table = pq.read_table(io.BytesIO(data))
        schema = {f.name: str(f.type) for f in table.schema}
        assert schema["id"] == "int64"
        assert schema["published_at"] == "timestamp[us, tz=UTC]"
        assert schema["keywords"] == "string" and schema["og_tags"] == "string"   # jsonb → JSON text
        assert schema["is_crawled"] == "bool"
        assert schema["sentiment_score"] == "double"           # int 0 and float mixed
        assert schema["story_id"] == "string"                  # all-null → stable string type
        got = table.to_pylist()
        assert got[0]["title"] == "हिंदी शीर्षक"
        assert got[0]["keywords"] == '["a", "b"]'
        assert got[0]["published_at"] == datetime(2026, 4, 10, 5, tzinfo=timezone.utc)
        assert parquet.parquet_row_count(data) == 2 and parquet.parquet_ids(data) == [1, 2]

    def test_checksum_is_content_hash(self):
        assert parquet.sha256(b"abc") == \
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────

class FakeStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.ensured = False
        self.corrupt_on_get: set[str] = set()

    def ensure_bucket(self):
        self.ensured = True

    def put(self, path, data):
        self.objects[path] = data

    def get(self, path):
        data = self.objects[path]
        return data[:-1] + b"X" if path in self.corrupt_on_get else data

    def remove(self, paths):
        for p in paths:
            self.objects.pop(p, None)

    def list(self, prefix):
        return sorted(p for p in self.objects if p.startswith(prefix + "/"))


class FakeDB:
    """In-memory articles of one or more days + the archive log + prune guard."""

    def __init__(self, days: dict[date, int]):
        self.articles = {d: [{"id": d.toordinal() * 100_000 + i, "title": f"t{i}",
                              "crawled_at": f"{d}T01:00:00+00:00", "cluster_id": None}
                             for i in range(n)] for d, n in days.items()}
        self.logs: dict[str, dict] = {}
        self.added_during_export: date | None = None

    unenriched: dict = {}

    def day_unenriched(self, day):
        return self.unenriched.get(day, 0)

    def pending_days(self, hot_days):
        return [(d, (self.logs.get(d.isoformat()) or {}).get("status"))
                for d in sorted(self.articles)
                if (self.logs.get(d.isoformat()) or {}).get("status") != "pruned"]

    def day_stats(self, day):
        ids = [a["id"] for a in self.articles[day]]
        return {"article_rows": len(ids), "min_id": min(ids) if ids else None,
                "max_id": max(ids) if ids else None}

    def article_page(self, day, after_id, limit):
        if self.added_during_export == day:
            self.articles[day].append({"id": 10**12, "title": "late"})
            self.added_during_export = None
        rows = [a for a in sorted(self.articles[day], key=lambda a: a["id"]) if a["id"] > after_id]
        return rows[:limit]

    def entity_page(self, day, after_key, limit):
        ents = [{"page_key": f"{a['id']:020d}:e", "article_id": a["id"], "entity_text": "X",
                 "entity_type": "ORG", "salience": 0.5} for a in self.articles[day]]
        return [e for e in ents if e["page_key"] > after_key][:limit]

    def clusters(self, day):
        return [{"id": "c1", "headline": "h", "article_count": 2}] if self.articles[day] else []

    def get_log(self, day):
        return self.logs.get(day.isoformat())

    def upsert_log(self, row):
        self.logs[row["day"]] = dict(row)

    def mark_verified(self, day):
        if self.logs[day.isoformat()]["status"] == "uploaded":
            self.logs[day.isoformat()]["status"] = "verified"

    def prune_day(self, day, min_age_days, max_delete):
        row = self.logs[day.isoformat()]
        assert row["status"] in ("verified", "pruned"), "prune called on unverified day"
        left = self.articles[day]
        assert len(left) + row["pruned_rows"] == row["article_rows"]
        n = min(max_delete, len(left))
        del left[:n]
        row["pruned_rows"] += n
        if not left:
            row["status"] = "pruned"
        return n


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

class TestRunArchive:

    def test_happy_path_multipart_verify_prune(self, monkeypatch):
        monkeypatch.setattr(runner, "PART_ROWS", 1_000)
        monkeypatch.setattr(runner, "PRUNE_CHUNK", 700)
        db, store = FakeDB({DAY: 2_500}), FakeStore()
        rep = run_archive(db, store, "bucket")
        log = db.logs[DAY.isoformat()]
        assert store.ensured
        assert [o["path"] for o in log["objects"] if o["kind"] == "articles"] == [
            "articles/2026/04/10/part-0001.parquet", "articles/2026/04/10/part-0002.parquet",
            "articles/2026/04/10/part-0003.parquet"]
        assert "entities/2026/04/10/part-0001.parquet" in store.objects
        assert "clusters/2026/04/10.parquet" in store.objects
        assert log["article_rows"] == 2_500 and log["entity_rows"] == 2_500 and log["cluster_rows"] == 1
        assert log["status"] == "pruned" and db.articles[DAY] == []
        assert rep.articles_archived == 2_500 and rep.rows_pruned == 2_500 and not rep.failures

    def test_no_prune_mode_keeps_rows(self):
        db, store = FakeDB({DAY: 10}), FakeStore()
        run_archive(db, store, "bucket", prune=False)
        assert db.logs[DAY.isoformat()]["status"] == "verified" and len(db.articles[DAY]) == 10

    def test_corrupted_upload_is_never_verified_or_pruned(self):
        db, store = FakeDB({DAY: 10}), FakeStore()
        store.corrupt_on_get.add("articles/2026/04/10/part-0001.parquet")
        rep = run_archive(db, store, "bucket")
        assert db.logs[DAY.isoformat()]["status"] == "uploaded"
        assert len(db.articles[DAY]) == 10
        assert "checksum mismatch" in rep.failures[0]

    def test_rows_added_during_export_abort_the_day(self):
        db, store = FakeDB({DAY: 10}), FakeStore()
        db.added_during_export = DAY
        rep = run_archive(db, store, "bucket")
        assert DAY.isoformat() not in db.logs and len(db.articles[DAY]) == 11
        assert "changed during export" in rep.failures[0]

    def test_table_changed_after_export_blocks_verification(self):
        db, store = FakeDB({DAY: 10}), FakeStore()
        row = runner.export_day(DAY, db, store, "bucket")
        db.articles[DAY].append({"id": 10**12})
        with pytest.raises(ArchiveError, match="re-archive"):
            runner.verify_day(DAY, row, db, store)
        assert db.logs[DAY.isoformat()]["status"] == "uploaded"

    def test_reexport_removes_stale_parts(self, monkeypatch):
        monkeypatch.setattr(runner, "PART_ROWS", 5)
        monkeypatch.setattr(runner, "PAGE", 5)          # parts are cut on page boundaries
        db, store = FakeDB({DAY: 12}), FakeStore()
        store.objects["articles/2026/04/10/part-0009.parquet"] = b"left over from a crashed run"
        runner.export_day(DAY, db, store, "bucket")
        assert "articles/2026/04/10/part-0009.parquet" not in store.objects
        assert len([p for p in store.objects if p.startswith("articles/")]) == 3

    def test_resume_verified_day_goes_straight_to_prune(self):
        db, store = FakeDB({DAY: 10}), FakeStore()
        run_archive(db, store, "bucket", prune=False)
        uploads_before = dict(store.objects)
        run_archive(db, store, "bucket")
        assert store.objects == uploads_before            # nothing re-exported
        assert db.logs[DAY.isoformat()]["status"] == "pruned"

    def test_one_failing_day_does_not_block_others(self):
        other = date(2026, 4, 11)
        db, store = FakeDB({DAY: 5, other: 5}), FakeStore()
        store.corrupt_on_get.add("articles/2026/04/10/part-0001.parquet")
        rep = run_archive(db, store, "bucket")
        assert db.logs[other.isoformat()]["status"] == "pruned"
        assert len(rep.failures) == 1

    def test_day_with_unenriched_articles_waits(self):
        """Every article is enriched before it leaves Postgres (archive_guard_migration.sql)."""
        other = date(2026, 4, 11)
        db, store = FakeDB({DAY: 5, other: 5}), FakeStore()
        db.unenriched = {DAY: 2}
        rep = run_archive(db, store, "bucket")
        assert DAY.isoformat() not in db.logs and len(db.articles[DAY]) == 5     # untouched
        assert db.logs[other.isoformat()]["status"] == "pruned"                    # others proceed
        assert rep.days_waiting == 1 and not rep.failures
        db.unenriched = {}
        run_archive(db, store, "bucket")                                           # sweeper caught up
        assert db.logs[DAY.isoformat()]["status"] == "pruned"

    def test_empty_day_is_logged_and_closed(self):
        db, store = FakeDB({DAY: 0}), FakeStore()
        rep = run_archive(db, store, "bucket")
        assert db.logs[DAY.isoformat()]["status"] == "pruned" and store.objects == {}
        assert not rep.failures

    def test_dry_run_writes_nothing(self):
        db, store = FakeDB({DAY: 10}), FakeStore()
        rep = run_archive(db, store, "bucket", dry_run=True)
        assert not store.ensured and store.objects == {} and db.logs == {}
        assert rep.articles_archived == 10

    def test_time_budget_stops_between_days(self, monkeypatch):
        clock = iter([0.0, 999_999.0])
        monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock, 999_999.0))
        db, store = FakeDB({DAY: 3, date(2026, 4, 11): 3}), FakeStore()
        rep = run_archive(db, store, "bucket", time_budget_minutes=1)
        assert rep.stop_reason == "time_budget" and rep.days_seen == 0
