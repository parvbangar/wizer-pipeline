"""
enrichment/cluster_state_sync.py
════════════════════════════════
Bring the in-memory cluster state (enrichment/memory_clustering.py) up to date
with the database, reading as little as possible.

  load_full    cold start / cache miss: every active cluster last seen inside
               the window (keyset-paged wizer_cluster_state).
  apply_delta  normal run: only clusters the database changed after the
               state's bookmark — in practice maintenance merges, because the
               clusterer's own writes are stamped with the bookmark itself
               (wizer_apply_cluster_changes returns its transaction time).

`fetch(**params)` calls wizer_cluster_state with named p_* arguments and returns
its rows (enrichment/db.py in production, a direct SQL call in tests), so this
module has no I/O of its own.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from enrichment.memory_clustering import ClusterState

log = logging.getLogger(__name__)

_PAGE = 200
_NIL_UUID = "00000000-0000-0000-0000-000000000000"


def _iso(ts) -> str | None:
    if ts is None:
        return None
    return ts.isoformat() if isinstance(ts, datetime) else str(ts)


def _as_dt(ts) -> datetime:
    return ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _pages(fetch: Callable, **params):
    after = _NIL_UUID
    while True:
        rows = fetch(p_after_id=after, p_limit=_PAGE, **params) or []
        yield rows
        if len(rows) < _PAGE:
            return
        after = str(rows[-1]["id"])


def load_full(fetch: Callable, model: str, window_since, db_now: Callable) -> ClusterState:
    """
    Every active cluster of `model` last seen since `window_since`.
    The bookmark is the DB clock read BEFORE paging, so anything that changes
    while the load runs is picked up again by the next delta.
    """
    state = ClusterState(model=model, synced_at=_iso(db_now()))
    n = 0
    for rows in _pages(fetch, p_model=model, p_changed_since=None, p_window_since=_iso(window_since)):
        for row in rows:
            state.upsert_row(row)
        n += len(rows)
    log.info("Cluster state: full load of %d active clusters (since %s)", n, _iso(window_since))
    return state


def apply_delta(fetch: Callable, state: ClusterState, model: str) -> int:
    """Apply clusters changed after state.synced_at; returns how many rows were applied."""
    if not state.synced_at:
        raise ValueError("state has no sync bookmark — use load_full")
    since = state.synced_at
    newest = _as_dt(since)
    n = 0
    for rows in _pages(fetch, p_model=model, p_changed_since=since, p_window_since=None):
        for row in rows:
            state.upsert_row(row)
            if row.get("updated_at") is not None:
                newest = max(newest, _as_dt(row["updated_at"]))
        n += len(rows)
    if n:
        state.synced_at = newest.isoformat()
        log.info("Cluster state: applied %d changed clusters since %s", n, since)
    return n
