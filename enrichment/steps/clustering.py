"""
enrichment/steps/clustering.py
════════════════════════════════
Groups articles about the same real-world event into story clusters.

WHY CLUSTERING IS CRITICAL FOR PROPENSITY SCORING:
  PTI (Press Trust of India) is India's main wire service.
  It distributes one story to 100+ outlets. Without clustering:
    - Your model sees 100 identical signals → thinks this story is 100x important
    - Your app shows the user the same story 100 times
  With clustering:
    - "1 story, covered by 87 outlets in the last 4 hours"
    - outlet_count = 87 → strong virality signal
    - User sees one best version of the story

HOW THE ALGORITHM WORKS:
  Two articles are in the same cluster if they cover the same event.
  We detect this with two signals:

  Signal 1 — Entity overlap (primary signal, weight 0.6):
    Extract entity sets from both articles.
    Compute Jaccard similarity: |intersection| / |union|
    Two articles about "Modi" + "Budget" + "Parliament" are very likely
    the same story.

  Signal 2 — Title SimHash (secondary signal, weight 0.4):
    Compare title SimHash bit distance (already computed in Layer 1).
    Distance ≤ 5 bits = nearly identical titles = same story.

  Combined score = 0.6 × entity_jaccard + 0.4 × title_similarity
  If score ≥ CLUSTER_SCORE_THRESHOLD (0.35) → same cluster.

CLUSTERING WINDOW:
  Only compare against clusters from the last CLUSTER_WINDOW_HOURS (48h).
  A "Modi Budget" story from 3 months ago should NOT cluster with today's
  "Modi Budget" story. 48h covers the full lifecycle of breaking news.

CANONICAL ARTICLE:
  The first article in the cluster becomes the canonical article.
  Its headline is stored as the cluster headline.
  When the app shows "1 story from 87 outlets", it shows the canonical
  article's title and the outlet_count.

ENTITY SET FORMAT (stored in article_clusters.entity_set):
  A jsonb array of lowercase entity text strings:
  ["narendra modi", "budget 2024", "parliament", "bjp"]
  Used for fast in-Python set intersection during matching.

TOP ENTITIES FORMAT (stored in article_clusters.top_entities):
  A jsonb array of dicts with counts, for display:
  [{"text": "Narendra Modi", "type": "PERSON", "count": 12}, ...]
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from enrichment.config import (
    CLUSTER_SCORE_THRESHOLD,
    CLUSTER_SIMHASH_THRESHOLD,
    CLUSTER_MIN_SHARED_ENTITIES,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SIMHASH HELPERS
# (same logic as pipeline/dedup.py — duplicated to keep enrichment independent)
# ─────────────────────────────────────────────────────────────────────────────

def _simhash_distance(hash_a: int | None, hash_b: int | None) -> int:
    """Count differing bits between two 64-bit hashes (Hamming distance)."""
    if hash_a is None or hash_b is None:
        return 64   # treat unknown as maximally distant
    return bin(hash_a ^ hash_b).count("1")


# ─────────────────────────────────────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _entity_jaccard(set_a: set[str], set_b: set[str]) -> float:
    """
    Jaccard similarity between two entity sets.
    Returns 0.0 if either set is empty.
    """
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    if intersection < CLUSTER_MIN_SHARED_ENTITIES:
        return 0.0   # fast path: not enough shared entities
    union = len(set_a | set_b)
    return intersection / union


def _title_similarity(dist: int) -> float:
    """Convert SimHash bit distance to a 0–1 similarity score."""
    if dist <= CLUSTER_SIMHASH_THRESHOLD:
        return 1.0 - (dist / 64)
    return max(0.0, 1.0 - (dist / 32))


def _cluster_score(
    article_entities: set[str],
    cluster_entity_set: set[str],
    article_simhash: int | None,
    cluster_simhash: int | None,
) -> float:
    """
    Combined cluster membership score (0.0 to 1.0).
    Higher = more likely same story.
    """
    entity_score = _entity_jaccard(article_entities, cluster_entity_set)
    title_score  = _title_similarity(
        _simhash_distance(article_simhash, cluster_simhash)
    )
    return round(0.6 * entity_score + 0.4 * title_score, 3)


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER ENTITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_top_entities(entities: list[dict]) -> list[dict]:
    """
    Convert flat entity list to top_entities format for cluster storage.
    e.g. [{"text": "Narendra Modi", "type": "PERSON", "count": 1}, ...]
    """
    return [
        {
            "text":  e["entity_text"],
            "type":  e["entity_type"],
            "count": 1,
        }
        for e in entities[:10]   # store top 10 entities per cluster
    ]


def _merge_top_entities(
    existing: list[dict],
    new_entities: list[dict],
) -> list[dict]:
    """
    Merge new article entities into a cluster's top_entities list,
    incrementing counts for entities that already appear.
    """
    existing_map = {e["text"].lower(): e for e in existing}

    for new_ent in new_entities:
        key = new_ent["entity_text"].lower()
        if key in existing_map:
            existing_map[key]["count"] += 1
        else:
            existing_map[key] = {
                "text":  new_ent["entity_text"],
                "type":  new_ent["entity_type"],
                "count": 1,
            }

    merged = list(existing_map.values())
    merged.sort(key=lambda x: x["count"], reverse=True)
    return merged[:15]   # keep top 15 entities per cluster


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def find_or_create_cluster(
    article: dict,
    entities: list[dict],
    title_simhash: int | None,
    recent_clusters: list[dict],
) -> tuple[str | None, dict | None, str]:
    """
    Find the best matching cluster for an article, or signal that a new one
    should be created.

    This function is PURE — it does not touch the database.
    The runner calls db.create_cluster or db.update_cluster based on the result.

    Args:
      article:         Article dict (needs: id, title, domain, published_at)
      entities:        Entity list from NER step
      title_simhash:   SimHash of article title (may be None)
      recent_clusters: All clusters from last 48h (pre-loaded by runner)

    Returns:
      (cluster_id, cluster_update, action) where action is:
        'join'   — article joins an existing cluster
                   cluster_id is the existing cluster's uuid
                   cluster_update is the update payload for db.update_cluster
        'create' — no match found, create a new cluster
                   cluster_id is None (will be assigned after DB insert)
                   cluster_update is the new cluster payload for db.create_cluster
        'skip'   — article has no entities and weak title → don't cluster
                   cluster_id is None, cluster_update is None
    """
    article_entity_set = {e["entity_text"].lower() for e in entities}
    published_at = article.get("published_at") or datetime.now(timezone.utc).isoformat()
    domain       = article.get("domain", "")
    title        = article.get("title", "")

    # ── Find best matching cluster ────────────────────────────────────────────
    best_cluster   = None
    best_score     = 0.0

    for cluster in recent_clusters:
        cluster_entity_set = set(cluster.get("entity_set") or [])
        cluster_simhash    = cluster.get("canonical_simhash")

        score = _cluster_score(
            article_entity_set,
            cluster_entity_set,
            title_simhash,
            cluster_simhash,
        )

        if score > best_score:
            best_score   = score
            best_cluster = cluster

    # ── JOIN existing cluster ─────────────────────────────────────────────────
    if best_score >= CLUSTER_SCORE_THRESHOLD and best_cluster:
        cluster_id = best_cluster["id"]

        # Build update payload for the cluster
        merged_entities = _merge_top_entities(
            best_cluster.get("top_entities") or [],
            entities,
        )
        merged_entity_set = list(
            set(best_cluster.get("entity_set") or []) | article_entity_set
        )

        # Count distinct outlets (domain uniqueness is tracked by outlet_count)
        # We increment conservatively — the true unique outlet count
        # would need a separate outlets table. This approximation is fine.
        new_outlet_count  = best_cluster.get("outlet_count", 1)
        if domain and domain not in (best_cluster.get("entity_set") or []):
            new_outlet_count += 1

        cluster_update = {
            "article_count": best_cluster.get("article_count", 1) + 1,
            "outlet_count":  new_outlet_count,
            "entity_set":    merged_entity_set,
            "top_entities":  merged_entities,
            "last_seen_at":  published_at,
        }

        log.debug(
            "Article '%s...' joined cluster %s (score=%.2f)",
            title[:40], cluster_id, best_score,
        )
        return cluster_id, cluster_update, "join"

    # ── CREATE new cluster ────────────────────────────────────────────────────
    # Only create a cluster if the article has meaningful entity signals
    if not article_entity_set and title_simhash is None:
        log.debug("Article '%s...' skipped clustering — no entities or simhash", title[:40])
        return None, None, "skip"

    new_cluster = {
        "canonical_article_id": article.get("id"),
        "headline":             title[:500] if title else None,
        "outlet_count":         1,
        "article_count":        1,
        "entity_set":           list(article_entity_set),
        "top_entities":         _build_top_entities(entities),
        "canonical_simhash":    title_simhash,
        "first_seen_at":        published_at,
        "last_seen_at":         published_at,
    }

    log.debug("Article '%s...' starts new cluster", title[:40])
    return None, new_cluster, "create"
