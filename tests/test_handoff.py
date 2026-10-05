"""tests/test_handoff.py — pipeline/handoff.py (ingest → processing file)."""

from __future__ import annotations

from pipeline.crawler import CrawledArticle
from pipeline.handoff import Handoff, find_files, read, shard_of


def _art(h, body):
    return CrawledArticle(feed_id="f", url=f"https://x/{h}", url_hash=h, title=f"t{h}", full_text=body)


def test_round_trip_shards_and_truncation(tmp_path):
    h = Handoff(max_body_chars=5)
    rows = [{"id": i, "url_hash": f"{1000 + i:<32}", "title": f"t{i}"} for i in range(6)]
    h.add_inserted(rows, [_art(1000 + i, "abcdefgh") for i in range(6)])
    path = tmp_path / "nested" / "handoff.jsonl.gz"
    assert h.write(path) == 6
    back = read(find_files(tmp_path))
    assert [r["id"] for r in back] == list(range(6)) and back[0]["full_text"] == "abcde"
    parts = [shard_of(back, i, 3) for i in range(3)]
    assert sorted(r["id"] for p in parts for r in p) == list(range(6))
    assert all(r["id"] % 3 == i for i, p in enumerate(parts) for r in p)


def test_unmatched_rows_are_counted_not_written(tmp_path):
    h = Handoff()
    h.add_inserted([{"id": 1, "url_hash": "999"}, {"id": None, "url_hash": "5"}], [_art(5, "b")])
    assert len(h) == 0 and h.unmatched == 2


def test_empty_file_still_written_and_missing_files_skipped(tmp_path):
    p = tmp_path / "e.jsonl.gz"
    assert Handoff().write(p) == 0 and p.exists()
    assert read([p, tmp_path / "missing.jsonl.gz"]) == []
