"""
tests/test_cluster_flush.py — cluster_job._apply survives a saturated database.

2026-10-07: with eight enrichment writers on the Micro instance, a 150-cluster
write timed out twice and crashed the clustering job. Re-applying a chunk is
harmless (absolute values), so _apply backs off, retries and splits instead.
"""

from __future__ import annotations

import pytest

from enrichment import cluster_job


def _chunk(n):
    clusters = [{"id": f"c{i}"} for i in range(n)]
    stamps = [{"cluster_id": f"c{i}", "id": 100 + i} for i in range(n)]
    return clusters, stamps


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(cluster_job.time, "sleep", slept.append)
    return slept


def _ok(chunk, stamps, t="2026-10-07T08:00:00+00:00"):
    return {"clusters_written": len(chunk), "articles_written": len(stamps), "synced_at": t}


def test_big_chunk_that_times_out_is_split_until_it_fits(monkeypatch):
    calls = []

    def apply(model, chunk, stamps):
        calls.append(len(chunk))
        if len(chunk) > 40:
            raise Exception("The read operation timed out")
        assert {s["cluster_id"] for s in stamps} == {c["id"] for c in chunk}   # stamps travel with their cluster
        return _ok(chunk, stamps)
    monkeypatch.setattr(cluster_job.db, "apply_cluster_changes", apply)
    clusters, stamps = _chunk(150)
    row = cluster_job._apply("m", clusters, stamps)
    assert row["clusters_written"] == 150 and row["articles_written"] == 150
    assert calls[0] == 150 and max(c for c in calls[1:] if c <= 40) <= 40


def test_single_cluster_is_retried_with_backoff(monkeypatch, no_sleep):
    attempts = []

    def apply(model, chunk, stamps):
        attempts.append(1)
        if len(attempts) < 3:
            raise Exception({"code": "57014", "message": "canceling statement due to statement timeout"})
        return _ok(chunk, stamps)
    monkeypatch.setattr(cluster_job.db, "apply_cluster_changes", apply)
    clusters, stamps = _chunk(1)
    assert cluster_job._apply("m", clusters, stamps)["clusters_written"] == 1
    assert no_sleep == list(cluster_job.APPLY_BACKOFF_S)


def test_persistent_timeout_on_one_cluster_raises(monkeypatch):
    def apply(model, chunk, stamps):
        raise Exception("The read operation timed out")
    monkeypatch.setattr(cluster_job.db, "apply_cluster_changes", apply)
    clusters, stamps = _chunk(1)
    with pytest.raises(Exception, match="timed out"):
        cluster_job._apply("m", clusters, stamps)


def test_other_errors_are_not_retried(monkeypatch):
    calls = []

    def apply(model, chunk, stamps):
        calls.append(1)
        raise ValueError("invalid input syntax for type uuid")
    monkeypatch.setattr(cluster_job.db, "apply_cluster_changes", apply)
    clusters, stamps = _chunk(10)
    with pytest.raises(ValueError):
        cluster_job._apply("m", clusters, stamps)
    assert calls == [1]
