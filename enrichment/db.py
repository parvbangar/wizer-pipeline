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
from datetime import datetime, timezone

from supabase import create_client, Client, ClientOptions

from enrichment.config import (
    SUPABASE_URL, SUPABASE_KEY,
    TABLE_ARTICLES, TABLE_ENTITIES, TABLE_CLUSTERS,
    CLUSTER_WINDOW_HOURS, CLUSTER_EMBEDDING_THRESHOLD,
)

log = logging.getLogger(__name__)

_client: Client | None = None

# Default postgrest timeout is 120 s — long enough that an idle TCP connection
# (dropped by OS/Supabase after ~2 min of NLP work) causes every write to hang
# for the full 2 min before raising ReadTimeout.
# 20 s is generous for any real write while failing fast on dead connections.
_DB_TIMEOUT_SECONDS = 20


def get_client() -> Client:
    """Return the singleton Supabase client, creating it on first call."""
    global _client
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
      - ORDER BY published_at DESC → process newest articles first so the app
                                     layer gets enriched data for fresh articles
                                     before stale ones.

    Returns a list of article dicts with all columns needed by the enrichment steps.
    Returns an empty list if something goes wrong (pipeline continues).
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
                .is_("enriched_at", "null")
                .eq("is_crawled", True)
                .not_.is_("title", "null")
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
# READING: NEAREST CLUSTER SEARCH (pgvector HNSW)
# ─────────────────────────────────────────────────────────────────────────────

def find_nearest_cluster(
    embedding: list[float],
    threshold: float | None = None,
    window_hours: int | None = None,
) -> dict | None:
    """
    Find the closest cluster to an article embedding using pgvector HNSW.

    Calls the find_nearest_cluster() SQL function (docs/pgvector_migration.sql).
    One DB round-trip per article, sub-millisecond query regardless of cluster count.

    Args:
      embedding:    768-dim LaBSE embedding for the article (normalised floats)
      threshold:    Min cosine similarity to qualify (default: CLUSTER_EMBEDDING_THRESHOLD)
      window_hours: Only search clusters seen in last N hours (default: CLUSTER_WINDOW_HOURS)
                    Pass 0 to search all clusters with no time limit.

    Returns the best-matching cluster dict (with 'similarity' key), or None.
    """
    if threshold is None:
        threshold = CLUSTER_EMBEDDING_THRESHOLD
    if window_hours is None:
        window_hours = CLUSTER_WINDOW_HOURS

    # Format as pgvector literal: '[x,y,z,...]'
    vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"

    try:
        resp = _run_with_retry(lambda: get_client().rpc("find_nearest_cluster", {
            "query_embedding":      vec_str,
            "similarity_threshold": threshold,
            "window_hours":         window_hours,
        }).execute())
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception as e:
        log.warning("find_nearest_cluster RPC failed: %s — article will start new cluster", e)
        return None


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
# WRITING: CLUSTER OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_cluster(cluster_data: dict) -> str | None:
    """
    Insert a new cluster row and return its uuid.

    Writes both canonical_embedding (jsonb, for backward compat) and
    embedding_vec (vector, for pgvector HNSW search). If cluster_data
    contains a 'canonical_embedding' list, embedding_vec is derived from it.

    Returns the new cluster uuid, or None on failure.
    """
    # Populate embedding_vec from canonical_embedding if present
    raw_embedding = cluster_data.get("canonical_embedding")
    if raw_embedding and isinstance(raw_embedding, list):
        cluster_data["embedding_vec"] = (
            "[" + ",".join(f"{v:.6f}" for v in raw_embedding) + "]"
        )

    try:
        resp = _run_with_retry(
            lambda: get_client().table(TABLE_CLUSTERS).insert(cluster_data).execute()
        )
        rows = resp.data or []
        if rows:
            return rows[0].get("id")
        return None
    except Exception as e:
        log.error("create_cluster failed: %s", e)
        return None


def update_cluster(cluster_id: str, update: dict) -> bool:
    """
    Update an existing cluster with new stats (outlet_count, article_count,
    entity_set, outlet_set, top_entities, canonical_embedding, etc.).

    When canonical_embedding is included (running mean update), also formats
    embedding_vec as a pgvector literal so the HNSW index stays current.

    Called when an article joins an existing cluster.
    Returns True on success, False on failure.
    """
    update["updated_at"] = datetime.now(timezone.utc).isoformat()
    # Keep embedding_vec in sync with canonical_embedding
    raw_embedding = update.get("canonical_embedding")
    if raw_embedding and isinstance(raw_embedding, list):
        update["embedding_vec"] = (
            "[" + ",".join(f"{v:.6f}" for v in raw_embedding) + "]"
        )
    try:
        _run_with_retry(
            lambda: get_client().table(TABLE_CLUSTERS).update(update).eq("id", cluster_id).execute()
        )
        return True
    except Exception as e:
        log.error("update_cluster failed for %s: %s", cluster_id, e)
        return False
