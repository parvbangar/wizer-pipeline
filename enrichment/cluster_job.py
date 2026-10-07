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


# Backoff between attempts of one chunk while the database is saturated.
APPLY_BACKOFF_S = (20, 60)


def _saturated(e: Exception) -> bool:
    msg = str(e).lower()
    return any(t in msg for t in ("timed out", "timeout", "57014", "canceling statement",
                                  "connection", "server disconnected", "too many"))


def _apply(model: str, chunk: list[dict], stamps: list[dict]) -> dict:
    """
    wizer_apply_cluster_changes for one chunk, surviving a saturated database.

    It writes absolute values (sums, counts, stamps from memory), so re-applying a
    chunk whose earlier attempt did commit changes nothing. On a timeout the chunk
    is retried after a backoff, then split in half (2026-10-07: with 8 enrichment
    writers on Micro a 150-cluster chunk timed out twice and crashed the job).
    """
    for i, wait in enumerate((0, *APPLY_BACKOFF_S)):
        if wait:
            log.warning("Cluster write of %d clusters timed out — retrying in %d s", len(chunk), wait)
            time.sleep(wait)
        try:
            return db.apply_cluster_changes(model, chunk, stamps)
        except Exception as e:
            if not _saturated(e):
                raise
            last = e
            if len(chunk) > 1 and i == 0:
                break                                    # split straight away; smaller writes get through
    if len(chunk) <= 1:
        raise last
    half = len(chunk) // 2
    ids_a = {c["id"] for c in chunk[:half]}
    a = _apply(model, chunk[:half], [s for s in stamps if s["cluster_id"] in ids_a])
    b = _apply(model, chunk[half:], [s for s in stamps if s["cluster_id"] not in ids_a])
    return {"clusters_written": int(a["clusters_written"]) + int(b["clusters_written"]),
            "articles_written": int(a["articles_written"]) + int(b["articles_written"]),
            "synced_at": max(str(a["synced_at"]), str(b["synced_at"]))}


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
        row = _apply(model, chunk, stamps)
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


MERGE_ROUNDS = 5


def merge_twins(state: ClusterState, touched: set[str], model: str = CLUSTER_EMBEDDING_MODEL) -> int:
    """
    Fold twin clusters together — the maintenance merge, decided in memory.

    Online assignment can seed two clusters for one story when its first
    articles arrive worded differently. The candidates come from the state
    (merge_candidates: the same rule as the SQL sweep, without an ANN index
    in Postgres); the merge itself stays in SQL (wizer_merge_clusters moves
    the members and adds the sums under the advisory lock), and a delta sync
    brings the merged rows back into the state. Probes are the clusters this
    run touched, then each round's winners.
    """
    from enrichment.cluster_maintenance import plan_merges
    from enrichment.config import CLUSTER_MERGE_THRESHOLD
    merged = 0
    probes = set(touched)
    for _ in range(MERGE_ROUNDS):
        plan = plan_merges(state.merge_candidates(probes, CLUSTER_MERGE_THRESHOLD), CLUSTER_MERGE_THRESHOLD)
        if not plan:
            break
        winners = set()
        for winner, loser, sim in plan:
            try:
                if db.merge_clusters(winner, loser):
                    merged += 1
                    winners.add(winner)
                    log.debug("merged %s → %s (avg-link %.3f)", loser, winner, sim)
            except Exception as e:
                log.warning("merge %s → %s failed: %s", loser, winner, e)
        if not winners:
            break
        apply_delta(db.fetch_cluster_state, state, model)     # merged rows back into memory
        probes = winners
    if merged:
        log.info("Merged %d twin clusters", merged)
    return merged


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
    written: dict[str, int] = {}
    touched: set[str] = set()
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
            touched.add(a.cluster_id)
        # Write each page as soon as it is assigned: a job killed by its
        # timeout keeps everything up to here (the next run's delta sync
        # re-reads these rows; the rest is still on its work list).
        page_written = flush(state, model)
        for k, v in page_written.items():
            written[k] = written.get(k, 0) + v

    final = flush(state, model)
    for k, v in final.items():
        written[k] = written.get(k, 0) + v
    try:
        counts["merges"] = merge_twins(state, touched, model)
    except Exception as e:
        # Everything above is already written. A merge done in SQL but not yet
        # mirrored here is picked up by the next run's delta (the bookmark has
        # not moved past it), so the state is still worth saving (2026-10-07:
        # a delta-read timeout here crashed the job and lost its cache).
        log.error("Twin merging stopped: %s — the next run's delta sync catches up", e)
        counts["merges"] = 0
    keep_after = (datetime.now(timezone.utc) - timedelta(hours=CLUSTER_STATE_RETAIN_HOURS)).timestamp()
    pruned = state.prune(keep_after)
    if state_path:
        state.save(state_path)
    summary = {**counts, **{f"written_{k}": v for k, v in written.items()},
               "live_clusters": len(state), "pruned": pruned, "stop_reason": stop,
               "duration_s": round(time.perf_counter() - t0, 1)}
    log.info("In-memory clustering done: %s", summary)
    return summary
