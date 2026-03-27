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
from datetime import datetime, timezone, timedelta
from typing import Any

from supabase import create_client, Client

from enrichment.config import (
    SUPABASE_URL, SUPABASE_KEY,
    TABLE_ARTICLES, TABLE_ENTITIES, TABLE_CLUSTERS,
    CLUSTER_WINDOW_HOURS, CLUSTER_MAX_CANDIDATES,
)

log = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    """Return the singleton Supabase client, creating it on first call."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# READING: FETCH ARTICLES TO ENRICH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_unenriched_batch(limit: int, offset: int = 0) -> list[dict]:
    """
    Fetch a batch of articles that haven't been enriched yet.

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
            .eq("is_crawled", True)            # only articles with full text fetched
            .not_.is_("title", "null")         # skip articles with no title
            .order("published_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        log.error("fetch_unenriched_batch failed: %s", e)
        return []


def fetch_unenriched_batch_forced(limit: int, offset: int = 0) -> list[dict]:
    """
    Same as fetch_unenriched_batch but fetches ALL articles (including already
    enriched ones). Used when --force flag is passed to re-enrich everything.
    """
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
            .range(offset, offset + limit - 1)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        log.error("fetch_unenriched_batch_forced failed: %s", e)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# READING: FETCH RECENT CLUSTERS (for clustering step)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_recent_clusters() -> list[dict]:
    """
    Load all clusters updated in the last CLUSTER_WINDOW_HOURS hours.

    WHY LOAD INTO MEMORY?
      The clustering step compares each article against all recent clusters.
      Loading them once per batch (not once per article) keeps DB calls minimal.
      At 48h window with ~10K articles/day → ~10K clusters max in memory.
      Each cluster dict is ~500 bytes → ~5 MB total. Totally fine.

    Returns list of cluster dicts (empty list on error).
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=CLUSTER_WINDOW_HOURS)
    ).isoformat()

    try:
        resp = (
            get_client()
            .table(TABLE_CLUSTERS)
            .select("id, canonical_article_id, headline, outlet_count, "
                    "article_count, entity_set, top_entities, canonical_simhash, "
                    "canonical_embedding, first_seen_at, last_seen_at")
            .gte("last_seen_at", cutoff)
            .order("last_seen_at", desc=True)
            .limit(CLUSTER_MAX_CANDIDATES)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        log.warning("fetch_recent_clusters failed: %s — clustering will create new clusters", e)
        return []


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
        get_client().table(TABLE_ARTICLES).update(update).eq("id", article_id).execute()
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
        get_client().table(TABLE_ENTITIES).delete().eq("article_id", article_id).execute()
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
        get_client().table(TABLE_ENTITIES).insert(rows).execute()
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

    Called when no matching cluster is found for a new article.
    The article becomes the canonical (first) article in the cluster.

    Returns the new cluster uuid, or None on failure.
    """
    try:
        resp = get_client().table(TABLE_CLUSTERS).insert(cluster_data).execute()
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
    entity_set, top_entities, last_seen_at, etc.).

    Called when an article joins an existing cluster.
    Returns True on success, False on failure.
    """
    update["updated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        get_client().table(TABLE_CLUSTERS).update(update).eq("id", cluster_id).execute()
        return True
    except Exception as e:
        log.error("update_cluster failed for %s: %s", cluster_id, e)
        return False
