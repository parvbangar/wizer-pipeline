"""
enrichment/clustering.py
════════════════════════
Story clustering — the Python half.

WHAT A CLUSTER IS:
  One real-world story (an event and its immediate follow-ups), told by any
  number of outlets in any number of languages. A PTI wire story reprinted by
  50 sites is ONE cluster with outlet_count = 50 — that number is the
  pipeline's core virality signal, and the unit the app shows users instead of
  the same story 50 times.

WHO DOES WHAT:
  Python (this file)                       Postgres (wizer_assign_cluster)
  ───────────────────────────────────────  ──────────────────────────────────
  embed the article (steps/embedding.py)   lock, find nearest compatible
  pick the entities that count as             clusters (exact cosine, time
    clustering evidence                       window), apply the decision rule,
  drop meaningless image hashes               create or update the cluster and
  pass thresholds from config                 stamp articles.cluster_id —
                                              all in ONE transaction

  The decision lives in SQL because only a single transaction can make
  "find → decide → write" atomic across concurrent runners. Everything that
  can be pure Python (and unit-tested without a database) is here.

The algorithm, its failure modes and the calibration evidence are written up
in docs/CLUSTERING.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from enrichment import db
from enrichment.config import (
    CLUSTER_ANCHOR_THRESHOLD,
    CLUSTER_CANDIDATES,
    CLUSTER_EMBEDDING_MODEL,
    CLUSTER_ENTITY_MIN_SALIENCE,
    CLUSTER_ENTITY_STOPLIST,
    CLUSTER_ENTITY_TYPES,
    CLUSTER_GRAY_THRESHOLD,
    CLUSTER_IMAGE_HASH_DENYLIST,
    CLUSTER_IMAGE_MAX_DISTANCE,
    CLUSTER_JOIN_THRESHOLD,
    CLUSTER_MAX_GAP_HOURS,
    CLUSTER_MAX_SPAN_HOURS,
    CLUSTER_MIN_SHARED_ENTITIES,
)
from enrichment.steps.embedding import to_pgvector

log = logging.getLogger(__name__)

# How many of the article's entities travel to the cluster's top_entities.
_TOP_ENTITIES_PER_ARTICLE = 10
# Cap on evidence keys: the most salient ones are the ones that identify a story.
_MAX_ENTITY_KEYS = 15


@dataclass(frozen=True)
class ClusterResult:
    cluster_id: str
    action: str                 # seed | join | gray_join | existing
    similarity: float | None
    article_count: int | None
    outlet_count: int | None


# ─────────────────────────────────────────────────────────────────────────────
# PURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def entity_key(text: str) -> str:
    """Case- and whitespace-insensitive identity of an entity mention."""
    return " ".join((text or "").lower().split())


def entity_keys_for_clustering(entities: list[dict]) -> list[str]:
    """
    The article's entities that count as evidence "this is the same story".

    Kept:    salient (≥ CLUSTER_ENTITY_MIN_SALIENCE) PERSON / ORG / GPE /
             EVENT / LAW / PRODUCT mentions, most salient first.
    Dropped: entities on CLUSTER_ENTITY_STOPLIST — "India", "PTI", "Centre"
             appear in a huge share of Indian stories, so sharing them proves
             nothing — and single-character noise.
    """
    keys: list[str] = []
    seen: set[str] = set()
    ranked = sorted(entities or [], key=lambda e: e.get("salience", 0.0), reverse=True)
    for ent in ranked:
        if ent.get("salience", 0.0) < CLUSTER_ENTITY_MIN_SALIENCE:
            continue
        if ent.get("entity_type") not in CLUSTER_ENTITY_TYPES:
            continue
        key = entity_key(ent.get("entity_text", ""))
        if len(key) < 2 or key in CLUSTER_ENTITY_STOPLIST or key in seen:
            continue
        seen.add(key)
        keys.append(key)
        if len(keys) >= _MAX_ENTITY_KEYS:
            break
    return keys


def top_entities_payload(entities: list[dict]) -> list[dict]:
    """The article's contribution to the cluster's top_entities tally."""
    ranked = sorted(entities or [], key=lambda e: e.get("salience", 0.0), reverse=True)
    return [
        {"text": e["entity_text"], "type": e["entity_type"]}
        for e in ranked[:_TOP_ENTITIES_PER_ARTICLE]
        if e.get("entity_text")
    ]


def usable_image_phash(phash: int | None) -> int | None:
    """Drop hashes of blank / solid-colour images — they match everything."""
    if phash is None or phash in CLUSTER_IMAGE_HASH_DENYLIST:
        return None
    return phash


def build_assign_params(
    article: dict,
    embedding: list[float],
    entities: list[dict],
    image_phash: int | None,
    language: str | None,
    model_id: str = CLUSTER_EMBEDDING_MODEL,
) -> dict:
    """
    Arguments for the wizer_assign_cluster RPC (names match the SQL signature).
    Pure — no I/O — so the full parameter contract is unit-testable.
    """
    return {
        "p_article_id":          article["id"],
        "p_embedding":           to_pgvector(embedding),
        "p_model":               model_id,
        "p_published_at":        article.get("published_at"),
        "p_domain":              article.get("domain"),
        "p_title":               article.get("title"),
        "p_language":            (language or article.get("language_code") or None),
        "p_entities":            top_entities_payload(entities),
        "p_entity_keys":         entity_keys_for_clustering(entities),
        "p_image_phash":         usable_image_phash(image_phash),
        "p_join_threshold":      CLUSTER_JOIN_THRESHOLD,
        "p_gray_threshold":      CLUSTER_GRAY_THRESHOLD,
        "p_anchor_threshold":    CLUSTER_ANCHOR_THRESHOLD,
        "p_min_shared_entities": CLUSTER_MIN_SHARED_ENTITIES,
        "p_image_max_distance":  CLUSTER_IMAGE_MAX_DISTANCE,
        "p_max_gap_hours":       CLUSTER_MAX_GAP_HOURS,
        "p_max_span_hours":      CLUSTER_MAX_SPAN_HOURS,
        "p_candidates":          CLUSTER_CANDIDATES,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_cluster(
    article: dict,
    embedding: list[float] | None,
    entities: list[dict],
    image_phash: int | None,
    language: str | None,
) -> ClusterResult | None:
    """
    Put one article into its story cluster (creating the cluster if needed).

    Returns None when the article cannot be clustered (no embedding) or the
    RPC failed — the article is then enriched without a cluster_id, and the
    backfill command (`python cluster.py backfill`) can cluster it later.
    Never raises: clustering problems must not cost an article its enrichment.
    """
    if embedding is None:
        return None
    params = build_assign_params(article, embedding, entities, image_phash, language)
    row = db.assign_cluster(params)
    if not row:
        return None
    return ClusterResult(
        cluster_id    = row["cluster_id"],
        action        = row["action"],
        similarity    = row.get("similarity"),
        article_count = row.get("article_count"),
        outlet_count  = row.get("outlet_count"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BATCHED ASSIGNMENT (cluster.py run)
# ─────────────────────────────────────────────────────────────────────────────

# Per-item keys of wizer_assign_cluster_batch; everything else is shared.
_ITEM_KEYS = ("p_article_id", "p_embedding", "p_published_at", "p_domain", "p_title",
              "p_language", "p_entities", "p_entity_keys", "p_image_phash")
_SHARED_KEYS = ("p_model", "p_join_threshold", "p_gray_threshold", "p_anchor_threshold",
                "p_min_shared_entities", "p_image_max_distance", "p_max_gap_hours",
                "p_max_span_hours", "p_candidates")
BATCH_SIZE = 50     # ~50 × 5 ms of SQL per call: far inside the 30 s API statement timeout


def assign_clusters_batch(items: list[tuple]) -> dict:
    """
    Assign many articles in as few round trips as possible.

    items: [(article, embedding, entities, image_phash, language), …] — the same
           arguments as assign_cluster(). Items without an embedding are skipped.

    Returns {article_id: ClusterResult | None}; None = not clustered (no
    embedding, or the item failed inside SQL — the reason is logged). Order of
    processing = order of `items`, so pass them oldest first.
    """
    out: dict = {}
    payload: list[dict] = []
    shared: dict | None = None
    for article, embedding, entities, phash, language in items:
        if embedding is None:
            out[article["id"]] = None
            continue
        params = build_assign_params(article, embedding, entities, phash, language)
        shared = shared or {k: params[k] for k in _SHARED_KEYS}
        payload.append({k: params[k] for k in _ITEM_KEYS})

    for i in range(0, len(payload), BATCH_SIZE):
        chunk = payload[i:i + BATCH_SIZE]
        rows = db.assign_cluster_batch(chunk, shared)
        if rows is None:                       # whole call failed — logged in db
            for it in chunk:
                out[it["p_article_id"]] = None
            continue
        for row in rows:
            if row.get("error"):
                log.warning("[%s] clustering failed: %s", row["article_id"], row["error"])
                out[row["article_id"]] = None
            else:
                out[row["article_id"]] = ClusterResult(
                    row["cluster_id"], row["action"], row.get("similarity"),
                    row.get("article_count"), row.get("outlet_count"))
    return out
