"""
enrichment/db.py
════════════════
All database operations for the Layer 2 enrichment pipeline.

WHY CENTRALISE DB CALLS HERE?
  Same reason as pipeline/db.py — one file for all DB logic means:
  - Column name changes = fix in one place
  - Easy to spot and prevent N+1 query patterns
  - The runner stays clean (no raw Supabase calls scattered around)

CONNECTION:
  Reuses the same Supabase service-key client as Layer 1.
  The client is created once (singleton) and reused for the entire run.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from supabase import create_client, Client, ClientOptions

from enrichment.config import (
    SUPABASE_URL, SUPABASE_KEY,
    TABLE_ARTICLES, TABLE_ENTITIES, TABLE_RUNS,
    ENRICH_MAX_AGE_HOURS, ENRICH_CLAIM_LEASE_MINUTES, ENRICH_MAX_ATTEMPTS,
)

log = logging.getLogger(__name__)

_client: Client | None = None

# Default postgrest timeout is 120 s — long enough that an idle TCP connection
# (dropped by OS/Supabase after ~2 min of NLP work) causes every write to hang
# for the full 2 min before raising ReadTimeout.
# (Previously 20 s, to fail fast on dead connections — see below.)
# 45 s: above the server's 30 s statement_timeout, so a slow-but-valid query on
# the small production instance is not abandoned by the client first (the
# first production clustering run timed out at 20 s on a 17 s query). Dead
# idle connections — the original reason for a short timeout — are avoided by
# the idle reconnect below instead.
_DB_TIMEOUT_SECONDS = 45

# A pooled HTTP connection that sat idle while models ran is often silently
# dropped by the network (NAT / load-balancer idle timeouts). The next request
# then hangs for the full _DB_TIMEOUT_SECONDS before the retry reconnects —
# production enrichment averaged ~65 s per article against ~16 s of actual NLP
# work. Re-creating the client after an idle gap avoids the dead connection
# instead of waiting it out.
_IDLE_RESET_SECONDS = 45
_last_used = 0.0


def get_client() -> Client:
    """
    Return the singleton Supabase client, creating it on first call — or
    re-creating it if it has been idle longer than _IDLE_RESET_SECONDS.
    """
    global _client, _last_used
    now = time.monotonic()
    if _client is not None and _last_used and now - _last_used > _IDLE_RESET_SECONDS:
        log.debug("DB client idle %.0fs — reconnecting proactively", now - _last_used)
        _client = None
    _last_used = now
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file"
            )
        _client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY,
            options=ClientOptions(postgrest_client_timeout=_DB_TIMEOUT_SECONDS),
        )
    return _client


def _reset_client() -> None:
    """Drop the cached client so the next call to get_client() reconnects."""
    global _client
    _client = None


def _is_retriable(e: Exception) -> bool:
    """True for network timeout/connection errors that warrant a reconnect."""
    import httpx
    if isinstance(e, (
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.PoolTimeout,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    )):
        return True
    # Fallback: check error text for cases wrapped by supabase/postgrest
    msg = str(e).lower()
    return "timed out" in msg or "connection refused" in msg or "broken pipe" in msg


def _run_with_retry(operation):
    """
    Execute a DB operation lambda, retrying once with a fresh connection on timeout.

    WHY:
      NLP steps (LaBSE, IndicNER, DeBERTa) take 1–5 min per article.
      During that time the Supabase TCP connection sits idle and is dropped
      by the network. The first write attempt then times out (20 s with our
      new limit). We reset the singleton, sleep briefly for the server to
      drain, and retry — the second attempt opens a fresh TCP connection and
      almost always succeeds.
    """
    try:
        return operation()
    except Exception as e:
        if _is_retriable(e):
            log.warning("DB connection error — reconnecting and retrying once: %s", e)
            _reset_client()
            time.sleep(1)
            return operation()   # let the caller catch if this also fails
        raise


# ─────────────────────────────────────────────────────────────────────────────
# READING: FETCH ARTICLES TO ENRICH
# ─────────────────────────────────────────────────────────────────────────────

_PAGE_SIZE = 1000   # Supabase PostgREST hard limit per request
_ENTITY_CHUNK = 33  # articles per entity lookup: 33 × ≤30 entities < 1000 rows


def fetch_unenriched_batch(limit: int, offset: int = 0) -> list[dict]:
    """
    Fetch a batch of articles that haven't been enriched yet.

    Paginates internally in chunks of 1,000 to work around Supabase's
    PostgREST default max-rows limit without requiring dashboard changes.

    QUERY LOGIC:
      - enriched_at IS NULL       → not yet processed by Layer 2
      - is_crawled = true         → full_text has been fetched; uncrawled articles
                                     have no body text, making NER/keywords/sentiment
                                     useless. Skip them entirely.
      - has some content          → at least a title (skip empty shells)
      - published_at > NOW()-48h  → only enrich fresh articles; stale articles
                                     will never surface in a freshness-ranked feed
                                     and waste compute. Gate disabled if
                                     ENRICH_MAX_AGE_HOURS=0.
      - ORDER BY published_at DESC → process newest articles first so the app
                                     layer gets enriched data for fresh articles
                                     before stale ones.

    Returns a list of article dicts with all columns needed by the enrichment steps.
    Returns an empty list if something goes wrong (pipeline continues).
    """
    age_cutoff: str | None = None
    if ENRICH_MAX_AGE_HOURS > 0:
        age_cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=ENRICH_MAX_AGE_HOURS)
        ).isoformat()

    results: list[dict] = []
    fetched = 0
    while fetched < limit:
        page_size = min(_PAGE_SIZE, limit - fetched)
        page_offset = offset + fetched
        try:
            query = (
                get_client()
                .table(TABLE_ARTICLES)
                .select(
                    "id, title, description, full_text, top_image_url, "
                    "url, domain, language_code, country_code, "
                    "published_at, iab_tier1, iab_tier2"
                )
                .is_("enriched_at", "null")
                .eq("is_crawled", True)
                .not_.is_("title", "null")
            )
            if age_cutoff:
                query = query.gt("published_at", age_cutoff)
            resp = (
                query
                .order("published_at", desc=True)
                .range(page_offset, page_offset + page_size - 1)
                .execute()
            )
        except Exception as e:
            log.error("fetch_unenriched_batch failed at offset %d: %s", page_offset, e)
            break
        page = resp.data or []
        results.extend(page)
        fetched += len(page)
        if len(page) < page_size:
            break   # no more rows
    return results


def fetch_unenriched_batch_forced(limit: int, offset: int = 0) -> list[dict]:
    """
    Same as fetch_unenriched_batch but fetches ALL articles (including already
    enriched ones). Used when --force flag is passed to re-enrich everything.
    Paginates internally in chunks of 1,000.
    """
    results: list[dict] = []
    fetched = 0
    while fetched < limit:
        page_size = min(_PAGE_SIZE, limit - fetched)
        page_offset = offset + fetched
        try:
            resp = (
                get_client()
                .table(TABLE_ARTICLES)
                .select(
                    "id, title, description, full_text, top_image_url, "
                    "url, domain, language_code, country_code, "
                    "published_at, iab_tier1, iab_tier2"
                )
                .not_.is_("title", "null")
                .order("published_at", desc=True)
                .range(page_offset, page_offset + page_size - 1)
                .execute()
            )
        except Exception as e:
            log.error("fetch_unenriched_batch_forced failed at offset %d: %s", page_offset, e)
            break
        page = resp.data or []
        results.extend(page)
        fetched += len(page)
        if len(page) < page_size:
            break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# WRITING: SAVE ENRICHMENT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_article_enrichment(article_id: str, update: dict) -> bool:
    """
    Update a single article row with enrichment outputs.

    Always sets enriched_at = now() so the article won't be picked up again
    on the next enrichment run.

    Args:
      article_id: UUID of the article to update
      update:     Dict of column → value pairs to write.
                  e.g. {"word_count": 412, "category": "politics", ...}

    Returns True on success, False on failure.
    """
    update["enriched_at"] = datetime.now(timezone.utc).isoformat()
    try:
        _run_with_retry(
            lambda: get_client().table(TABLE_ARTICLES).update(update).eq("id", article_id).execute()
        )
        return True
    except Exception as e:
        log.error("save_article_enrichment failed for %s: %s", article_id, e)
        return False


def save_entities(article_id: str, entities: list[dict]) -> bool:
    """
    Insert named entities for one article into article_entities.

    On re-enrichment (--force), deletes existing entities first to prevent
    duplicates. The DELETE + INSERT is not atomic but is safe for our use case —
    worst case is missing entities on a crashed re-run (just re-run again).

    Args:
      article_id: UUID of the article
      entities:   List of dicts with keys: entity_text, entity_type, salience

    Returns True on success, False on failure.
    """
    if not entities:
        return True

    # Delete existing entities for this article (handles re-enrichment)
    try:
        _run_with_retry(
            lambda: get_client().table(TABLE_ENTITIES).delete().eq("article_id", article_id).execute()
        )
    except Exception:
        pass  # If delete fails, insert will just create duplicates — tolerable

    rows = [
        {
            "article_id":  article_id,
            "entity_text": e["entity_text"],
            "entity_type": e["entity_type"],
            "salience":    e["salience"],
        }
        for e in entities
    ]

    try:
        _run_with_retry(
            lambda: get_client().table(TABLE_ENTITIES).insert(rows).execute()
        )
        return True
    except Exception as e:
        log.error("save_entities failed for %s: %s", article_id, e)
        return False




# ─────────────────────────────────────────────────────────────────────────────
# WORK QUEUE — claim / release (docs/enrichment_queue_migration.sql)
# ─────────────────────────────────────────────────────────────────────────────

# PostgREST caps every response at max_rows (1000 on Supabase by default), RPC
# result sets included. Claiming in pages well under that keeps a claimed row
# from ever being "claimed but not returned" (which would strand it until its
# lease expired).
_CLAIM_PAGE = 500


def claim_batch(
    limit: int,
    max_age_hours: int = ENRICH_MAX_AGE_HOURS,
    lease_minutes: int = ENRICH_CLAIM_LEASE_MINUTES,
    max_attempts: int = ENRICH_MAX_ATTEMPTS,
) -> list[dict]:
    """
    Atomically lease up to `limit` unenriched articles for this runner.

    Concurrent runners (parallel shards, overlapping workflow_run triggers)
    each get disjoint rows — FOR UPDATE SKIP LOCKED in
    wizer_claim_enrichment_batch(). Returned newest-first.

    Raises if the RPC is missing (migration not applied) or the DB is
    unreachable: a run that cannot claim work must fail loudly, not report a
    successful empty run.
    """
    claimed: list[dict] = []
    while len(claimed) < limit:
        page = min(_CLAIM_PAGE, limit - len(claimed))
        resp = _run_with_retry(lambda: get_client().rpc("wizer_claim_enrichment_batch", {
            "p_limit":          page,
            "p_max_age_hours":  max_age_hours,
            "p_lease_minutes":  lease_minutes,
            "p_max_attempts":   max_attempts,
        }).execute())
        rows = resp.data or []
        claimed.extend(rows)
        if len(rows) < page:
            break
    claimed.sort(key=lambda r: r.get("published_at") or "", reverse=True)
    return claimed


def release_claims(article_ids: list) -> int:
    """
    Return unprocessed claimed articles to the queue immediately (instead of
    making them wait out the lease). Best-effort: on failure the lease simply
    expires later. Returns the number of rows released.
    """
    if not article_ids:
        return 0
    released = 0
    for i in range(0, len(article_ids), 1000):
        chunk = article_ids[i:i + 1000]
        try:
            resp = _run_with_retry(lambda: get_client().rpc(
                "wizer_release_enrichment_claims", {"p_ids": chunk}
            ).execute())
            released += int(resp.data or 0)
        except Exception as e:
            log.warning("release_claims failed for %d ids (lease will expire instead): %s",
                        len(chunk), e)
    return released


def queue_depth(
    max_age_hours: int = ENRICH_MAX_AGE_HOURS,
    max_attempts: int = ENRICH_MAX_ATTEMPTS,
) -> int | None:
    """Eligible unenriched articles right now (None if the count failed)."""
    try:
        resp = _run_with_retry(lambda: get_client().rpc("wizer_enrichment_queue_depth", {
            "p_max_age_hours": max_age_hours,
            "p_max_attempts":  max_attempts,
        }).execute())
        return int(resp.data) if resp.data is not None else None
    except Exception as e:
        log.warning("queue_depth failed: %s", e)
        return None


def mark_enrichment_failed(article_id, error: str) -> bool:
    """
    Park an article that crashed enrich_one() on its final attempt:
    enriched_at is set (it leaves the queue) and enrich_error records why.
    """
    return save_article_enrichment(article_id, {"enrich_error": (error or "")[:1000]})


# ─────────────────────────────────────────────────────────────────────────────
# RUN LOG — enrichment_runs
# ─────────────────────────────────────────────────────────────────────────────

def log_run_start(row: dict) -> str | None:
    """Insert an enrichment_runs row; returns its id (None if logging failed)."""
    try:
        resp = _run_with_retry(
            lambda: get_client().table(TABLE_RUNS).insert(row).execute()
        )
        rows = resp.data or []
        return rows[0]["id"] if rows else None
    except Exception as e:
        log.warning("enrichment_runs insert failed (run continues unlogged): %s", e)
        return None


def log_run_finish(run_id: str | None, row: dict) -> None:
    """Complete an enrichment_runs row with final counters."""
    if not run_id:
        return
    row = {**row, "finished_at": datetime.now(timezone.utc).isoformat()}
    try:
        _run_with_retry(
            lambda: get_client().table(TABLE_RUNS).update(row).eq("id", run_id).execute()
        )
    except Exception as e:
        log.warning("enrichment_runs update failed: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# STORY CLUSTERING (docs/clustering_v2_migration.sql)
# ─────────────────────────────────────────────────────────────────────────────

def assign_cluster(params: dict) -> dict | None:
    """
    Call wizer_assign_cluster — find/create the article's story cluster and
    stamp articles.cluster_id, atomically.

    Safe to retry: the SQL function is idempotent (an already-clustered
    article returns action='existing' without changing anything), so the
    reconnect-and-retry in _run_with_retry cannot double-count an article.

    Returns {cluster_id, action, similarity, article_count, outlet_count} or
    None on failure (logged; the article is enriched without a cluster and
    `python cluster.py backfill` can pick it up later).
    """
    try:
        resp = _run_with_retry(
            lambda: get_client().rpc("wizer_assign_cluster", params).execute()
        )
    except Exception as e:
        log.warning("[%s] wizer_assign_cluster failed: %s", params.get("p_article_id"), e)
        return None
    rows = resp.data or []
    return rows[0] if rows else None


def fetch_unclustered(
    since_iso: str,
    limit: int = 500,
    after_published: str | None = None,
    after_id: int | None = None,
) -> list[dict]:
    """
    One page of articles without a cluster (wizer_fetch_unclustered), oldest
    first, keyset-paged on (published_at, id). Includes articles that have not
    been NLP-enriched yet: clustering covers every ingested article.
    """
    params = {"p_since": since_iso, "p_limit": min(limit, _PAGE_SIZE)}
    if after_published is not None:
        params.update({"p_after_published": after_published, "p_after_id": after_id})
    resp = _run_with_retry(lambda: get_client().rpc("wizer_fetch_unclustered", params).execute())
    return resp.data or []


def assign_cluster_batch(items: list[dict], shared: dict) -> list[dict] | None:
    """
    wizer_assign_cluster_batch — many articles per round trip. Per-item
    failures come back in each row's `error`; None means the whole call failed.
    Safe to retry: assignment is idempotent per article.
    """
    try:
        resp = _run_with_retry(lambda: get_client().rpc(
            "wizer_assign_cluster_batch", {"p_items": items, **shared}).execute())
        return resp.data or []
    except Exception as e:
        log.warning("wizer_assign_cluster_batch failed for %d items: %s", len(items), e)
        return None


def fetch_entities_for_articles(article_ids: list) -> dict:
    """
    {article_id: [entity dicts]} from article_entities.

    Chunks of 33 articles, ONE request each, no ORDER BY / paging: NER stores at
    most 30 entities per article, so a chunk returns ≤ 990 rows (< PostgREST's
    1000 cap). The earlier `IN (…200 ids…) ORDER BY article_id LIMIT 1000`
    form let the planner walk the whole 2.9M-row article_id index looking for
    matches — and brand-new articles have none — so it hit the 20 s timeout in
    the first production clustering run.
    """
    out: dict = {aid: [] for aid in article_ids}
    for i in range(0, len(article_ids), _ENTITY_CHUNK):
        chunk = article_ids[i:i + _ENTITY_CHUNK]
        resp = _run_with_retry(lambda: (
            get_client().table(TABLE_ENTITIES)
            .select("article_id, entity_text, entity_type, salience")
            .in_("article_id", chunk)
            .execute()
        ))
        for r in resp.data or []:
            out.setdefault(r["article_id"], []).append(r)
    return out


def find_merge_candidates(params: dict) -> list[dict]:
    """One page of wizer_find_cluster_merge_candidates (maintenance)."""
    resp = _run_with_retry(
        lambda: get_client().rpc("wizer_find_cluster_merge_candidates", params).execute()
    )
    return resp.data or []


def merge_clusters(winner_id: str, loser_id: str) -> bool:
    """Fold loser into winner; False if either is no longer mergeable."""
    resp = _run_with_retry(lambda: get_client().rpc("wizer_merge_clusters", {
        "p_winner": winner_id, "p_loser": loser_id,
    }).execute())
    return bool(resp.data)


def reconcile_cluster_counts(since_iso: str) -> int:
    """Recount cluster stats from member articles; returns clusters repaired."""
    resp = _run_with_retry(lambda: get_client().rpc(
        "wizer_reconcile_cluster_counts", {"p_since": since_iso}
    ).execute())
    return int(resp.data or 0)


def prune_orphan_clusters(older_than_hours: int) -> int:
    """Delete idle clusters that no article points at; returns count deleted."""
    resp = _run_with_retry(lambda: get_client().rpc(
        "wizer_prune_orphan_clusters", {"p_older_than_hours": older_than_hours}
    ).execute())
    return int(resp.data or 0)


def fetch_view(view: str, limit: int = 50) -> list[dict]:
    """Read a monitoring view (cluster_health, top_stories_24h, …)."""
    resp = _run_with_retry(
        lambda: get_client().table(view).select("*").limit(limit).execute()
    )
    return resp.data or []
