"""
archiver/db.py
══════════════
Database calls for the archive (docs/archive_migration.sql), over the same
Supabase client and reconnect-and-retry logic as enrichment/db.py.
"""

from __future__ import annotations

from datetime import date

from enrichment.db import _run_with_retry, get_client

LOG_TABLE = "article_archive_log"


def _rpc(name: str, params: dict) -> list:
    return _run_with_retry(lambda: get_client().rpc(name, params).execute()).data or []


def pending_days(hot_days: int) -> list[tuple[date, str | None]]:
    """Days older than the hot window that are not pruned yet, oldest first."""
    return [(date.fromisoformat(r["day"]), r.get("status")) for r in
            _rpc("wizer_archive_days", {"p_hot_days": hot_days})]


def day_stats(day: date) -> dict:
    rows = _rpc("wizer_archive_day_stats", {"p_day": day.isoformat()})
    r = rows[0] if rows else {}
    return {"article_rows": int(r.get("article_rows") or 0),
            "min_id": r.get("min_id"), "max_id": r.get("max_id")}


def article_page(day: date, after_id: int, limit: int = 1000) -> list[dict]:
    return _rpc("wizer_archive_page", {"p_day": day.isoformat(), "p_after_id": after_id, "p_limit": limit})


def entity_page(day: date, after_key: str, limit: int = 1000) -> list[dict]:
    return _rpc("wizer_archive_entities_page",
                {"p_day": day.isoformat(), "p_after_key": after_key, "p_limit": limit})


def clusters(day: date) -> list[dict]:
    return _rpc("wizer_archive_clusters", {"p_day": day.isoformat()})


def day_unenriched(day: date) -> int:
    """Articles of `day` still waiting for enrichment (dead letters that gave up excluded)."""
    data = _rpc("wizer_archive_day_unenriched", {"p_day": day.isoformat()})
    # PostgREST returns a scalar function's value bare (0 becomes [] via _rpc).
    if isinstance(data, (int, float)):
        return int(data)
    return int(data[0]) if data else 0


def get_log(day: date) -> dict | None:
    resp = _run_with_retry(lambda: get_client().table(LOG_TABLE).select("*")
                           .eq("day", day.isoformat()).limit(1).execute())
    rows = resp.data or []
    return rows[0] if rows else None


def upsert_log(row: dict) -> None:
    _run_with_retry(lambda: get_client().table(LOG_TABLE).upsert(row, on_conflict="day").execute())


def mark_verified(day: date) -> None:
    from datetime import datetime, timezone
    _run_with_retry(lambda: get_client().table(LOG_TABLE).update({
        "status": "verified", "verified_at": datetime.now(timezone.utc).isoformat(),
    }).eq("day", day.isoformat()).eq("status", "uploaded").execute())


def prune_day(day: date, min_age_days: int, max_delete: int) -> int:
    rows = _run_with_retry(lambda: get_client().rpc("wizer_prune_archived_day", {
        "p_day": day.isoformat(), "p_min_age_days": min_age_days, "p_max_delete": max_delete,
    }).execute())
    return int(rows.data or 0)


def all_logs() -> list[dict]:
    resp = _run_with_retry(lambda: get_client().table(LOG_TABLE).select(
        "day, status, article_rows, entity_rows, cluster_rows, total_bytes, pruned_rows, "
        "archived_at, verified_at, pruned_at").order("day").limit(1000).execute())
    return resp.data or []
