"""
enrichment/cluster_job.py
═════════════════════════
Story clustering in memory, results written in bulk
(`python cluster.py run --memory --state FILE [--bodies DIR]`,
.github/workflows/process.yml).

  1. cluster state: the cached file (GitHub Actions cache) brought up to date
     with a delta sync — or, on a cold start, a full load of the live window;
  2. work list: EVERY article still without a cluster, oldest ingested first,
     from the database (wizer_fetch_unclustered_ingested — titles and
     descriptions only). Taking it from the database, not from one run's
     hand-off, means a skipped or superseded job loses nothing: the next one
     picks the articles up. Bodies from the hand-off files at hand (--bodies)
     are overlaid for the embedding text where the description is too short;
  3. embed a page, assign each article with enrichment/memory_clustering.py;
  4. write clusters + article stamps with wizer_apply_cluster_changes, in
     chunks small enough for the API statement timeout on Micro;
  5. drop clusters no fresh article can join any more, save the state.

Only ONE instance may run at a time (concurrency group "story-clustering"):
the state file is the source of truth for the clusters it holds, and the
database is updated from it.

FAILURE: if a chunk write fails the job raises WITHOUT saving the state. The
next run starts from the previous cache and its delta sync re-reads whatever
the failed run did write (those rows carry newer updated_at), so state and
database never disagree; the articles still without a cluster are simply on
the next run's work list.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from enrichment import db
from enrichment.cluster_state_sync import apply_delta, load_full
from enrichment.config import CLUSTER_EMBEDDING_MODEL, CLUSTER_STATE_RETAIN_HOURS
from enrichment.memory_clustering import ClusterState, check_supported_config
from enrichment.steps.embedding import build_embedding_text, embed_texts

log = logging.getLogger(__name__)

CHUNK_CLUSTERS = 150


def _article_time(rec: dict):
    """The time the clusterer uses: published_at, unless missing or in the future."""
    now = datetime.now(timezone.utc)
    crawled = rec.get("crawled_at") or rec.get("ingested_at")
    pub = rec.get("published_at")
    if pub:
        t = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        if t <= now + timedelta(hours=1):
            return t
    return crawled or now


def load_state(state_path: str | Path | None, model: str = CLUSTER_EMBEDDING_MODEL) -> ClusterState:
    state = ClusterState.load(state_path, model) if state_path else None
    if state is None or not state.synced_at:
        window = datetime.now(timezone.utc) - timedelta(hours=CLUSTER_STATE_RETAIN_HOURS)
        return load_full(db.fetch_cluster_state, model, window_since=window, db_now=db.db_now)
    apply_delta(db.fetch_cluster_state, state, model)
    return state


def flush(state: ClusterState, model: str = CLUSTER_EMBEDDING_MODEL) -> dict:
    """Write pending changes in chunks; marks the state flushed only if every chunk succeeded."""
    clusters, articles = state.pending_changes()
    by_cluster: dict[str, list[dict]] = {}
    for a in articles:
        by_cluster.setdefault(a["cluster_id"], []).append(a)
    written = {"clusters": 0, "articles": 0}
    synced = None
    for i in range(0, len(clusters), CHUNK_CLUSTERS):
        chunk = clusters[i:i + CHUNK_CLUSTERS]
        stamps = [a for c in chunk for a in by_cluster.pop(c["id"], [])]
        row = db.apply_cluster_changes(model, chunk, stamps)
        written["clusters"] += int(row["clusters_written"])
        written["articles"] += int(row["articles_written"])
        synced = row["synced_at"]
    leftover = [a for rest in by_cluster.values() for a in rest]   # stamps for clusters not dirty (none expected)
    if leftover:
        row = db.apply_cluster_changes(model, [], leftover)
        written["articles"] += int(row["articles_written"])
        synced = row["synced_at"]
    state.mark_flushed(synced if isinstance(synced, str) or synced is None else synced.isoformat())
    return written


def run_memory_clustering(state_path: str | Path | None, bodies: dict | None = None,
                          since_hours: float = 72, page: int = 500, limit: int = 200_000,
                          time_budget_minutes: float = 0,
                          model: str = CLUSTER_EMBEDDING_MODEL) -> dict:
    """
    Cluster every unclustered article ingested in the last `since_hours`.
    bodies: {article_id: full_text} from hand-off files (optional).
    """
    check_supported_config()
    t0 = time.perf_counter()
    deadline = t0 + time_budget_minutes * 60 if time_budget_minutes > 0 else None
    bodies = bodies or {}
    state = load_state(state_path, model)
    log.info("Cluster state: %d live clusters (bookmark %s)", len(state), state.synced_at)

    since = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    counts = {"seen": 0, "seed": 0, "join": 0, "skipped": 0, "with_body": 0}
    after_ts = after_id = None
    stop = "drained"
    while counts["seen"] < limit:
        if deadline is not None and time.perf_counter() >= deadline:
            stop = "time_budget"
            break
        rows = db.fetch_unclustered_ingested(since, min(page, limit - counts["seen"]), after_ts, after_id)
        if not rows:
            break
        counts["seen"] += len(rows)
        after_ts, after_id = rows[-1]["ingested_at"], rows[-1]["id"]
        texts = []
        for r in rows:
            body = bodies.get(int(r["id"]))
            counts["with_body"] += body is not None
            texts.append(build_embedding_text(r.get("title"), r.get("description"), body))
        vectors = embed_texts(texts)
        batch = sorted(zip(rows, vectors), key=lambda rv: (str(_article_time(rv[0])), int(rv[0]["id"])))
        for rec, vec in batch:
            if vec is None:
                counts["skipped"] += 1
                continue
            try:
                a = state.assign(int(rec["id"]), vec, _article_time(rec), rec.get("domain"),
                                 rec.get("title"), rec.get("language_code"))
            except ValueError as e:
                log.warning("[%s] not clustered: %s", rec.get("id"), e)
                counts["skipped"] += 1
                continue
            counts[a.action] += 1

    written = flush(state, model)
    keep_after = (datetime.now(timezone.utc) - timedelta(hours=CLUSTER_STATE_RETAIN_HOURS)).timestamp()
    pruned = state.prune(keep_after)
    if state_path:
        state.save(state_path)
    summary = {**counts, **{f"written_{k}": v for k, v in written.items()},
               "live_clusters": len(state), "pruned": pruned, "stop_reason": stop,
               "duration_s": round(time.perf_counter() - t0, 1)}
    log.info("In-memory clustering done: %s", summary)
    return summary
