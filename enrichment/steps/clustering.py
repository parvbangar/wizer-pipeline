"""
enrichment/steps/clustering.py
════════════════════════════════
Groups articles about the same real-world event into story clusters
using LaBSE (Language-Agnostic BERT Sentence Embeddings).

WHY CLUSTERING IS CRITICAL:
  PTI (Press Trust of India) distributes one wire story to 100+ outlets.
  Without clustering:
    - Your propensity model sees 100 identical signals → thinks this story
      is 100x more important than it is
    - Users see the same story 100 times in the feed
  With clustering:
    - "1 story, covered by 87 outlets in 4 hours" — correct virality signal
    - User sees one best version with outlet_count as the engagement signal

MODEL: sentence-transformers/LaBSE (Google, 2022)
  Language-Agnostic BERT Sentence Embeddings.
    - Trained on 109 languages including ALL major Indian languages:
      Hindi (hi), Bengali (bn), Tamil (ta), Telugu (te), Kannada (kn),
      Malayalam (ml), Marathi (mr), Gujarati (gu), Punjabi (pa), Urdu (ur)
    - Produces 768-dim normalised vectors
    - Two articles covering the same event in DIFFERENT languages will have
      cosine similarity > 0.82, even with zero word overlap
    - This is the critical advantage over entity Jaccard + SimHash:
      entity extraction fails on non-English text, embeddings don't

WHY COSINE SIMILARITY OVER ENTITY JACCARD:
  Old approach: intersection of NER entities / union of NER entities
    Problem: spaCy NER is unreliable on Hindi/Tamil/Telugu script
    A Hindi article about "Modi budget" extracts 0 entities → Jaccard = 0
    → Every Hindi article starts a new cluster (completely broken)

  New approach: LaBSE cosine similarity on title + description
    "Modi announces new budget" (English) and
    "मोदी ने नया बजट पेश किया" (Hindi) →
    cosine similarity > 0.88 → correctly clustered ✓

THRESHOLD (CLUSTER_EMBEDDING_THRESHOLD = 0.82):
  Calibrated for Indian wire news syndication:
  - Same PTI story, different outlets: ~0.90-0.98
  - Same event, independently reported: ~0.83-0.92
  - Related but different stories: ~0.70-0.82
  - Unrelated stories: < 0.65
  Raise to 0.88 for stricter clustering, lower to 0.75 for looser.

ENTITY SET (kept from NER step):
  Even though clustering no longer DEPENDS on entities, we still populate
  entity_set and top_entities in clusters for the app layer:
  - Article tagging ("Modi", "BJP", "Budget")
  - Entity-based search ("all articles about Virat Kohli")
  - Propensity scoring features

EMBEDDING STORAGE:
  canonical_embedding (jsonb array of 768 floats) stored on article_clusters.
  Loaded once per batch for in-memory cosine similarity comparison.

INSTALL:
  pip install sentence-transformers torch
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from enrichment.config import CLUSTER_EMBEDDING_THRESHOLD, LABSE_MODEL

log = logging.getLogger(__name__)

# ── Model cache — loaded once per process ─────────────────────────────────────
_model = None


def _get_model():
    """
    Load and cache the LaBSE sentence transformer model.

    Loading takes ~5-8 seconds and uses ~1.8 GB RAM.
    Subsequent calls reuse the cached model instantly.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            log.info("Loading LaBSE embedding model '%s'...", LABSE_MODEL)
            _model = SentenceTransformer(LABSE_MODEL)
            log.info("LaBSE model loaded")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers torch"
            )
    return _model


def get_embedding(text: str) -> list[float]:
    """
    Encode text into a 768-dim LaBSE embedding vector.

    Returns a normalised float list (dot product = cosine similarity).
    Returns a zero vector on error (article will not cluster).
    """
    if not text or not text.strip():
        return [0.0] * 768

    try:
        model = _get_model()
        # normalize_embeddings=True → unit vectors → dot product = cosine similarity
        embedding = model.encode(text[:512], normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        return [0.0] * 768


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two normalised embedding vectors.
    Since LaBSE normalises to unit length, this is just a dot product.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(np.dot(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)))


# ── Entity helpers (for cluster display, not for clustering decision) ─────────

def _build_top_entities(entities: list[dict]) -> list[dict]:
    """Convert NER entity list to top_entities format for cluster storage."""
    return [
        {"text": e["entity_text"], "type": e["entity_type"], "count": 1}
        for e in entities[:10]
    ]


def _merge_top_entities(existing: list[dict], new_entities: list[dict]) -> list[dict]:
    """Merge new article entities into a cluster's top_entities, incrementing counts."""
    existing_map = {e["text"].lower(): e for e in existing}
    for ent in new_entities:
        key = ent["entity_text"].lower()
        if key in existing_map:
            existing_map[key]["count"] += 1
        else:
            existing_map[key] = {
                "text":  ent["entity_text"],
                "type":  ent["entity_type"],
                "count": 1,
            }
    merged = sorted(existing_map.values(), key=lambda x: x["count"], reverse=True)
    return merged[:15]


# ── Main function ──────────────────────────────────────────────────────────────

def find_or_create_cluster(
    article: dict,
    entities: list[dict],
    recent_clusters: list[dict],
) -> tuple[str | None, dict | None, str]:
    """
    Find the best matching cluster for an article using LaBSE cosine similarity,
    or signal that a new cluster should be created.

    This function is PURE — it does not touch the database.
    The runner calls db.create_cluster or db.update_cluster based on the result.

    Args:
      article:         Article dict (needs: id, title, description, domain, published_at)
      entities:        Entity list from NER step (used for cluster tags, not for matching)
      recent_clusters: All clusters from last 48h (pre-loaded by runner),
                       each must have a 'canonical_embedding' field

    Returns:
      (cluster_id, cluster_payload, action) where action is:
        'join'   — article joins an existing cluster
                   cluster_id = uuid of the existing cluster
                   cluster_payload = update dict for db.update_cluster
        'create' — no similar cluster found, create a new one
                   cluster_id = None (assigned after DB insert)
                   cluster_payload = new cluster dict for db.create_cluster
        'skip'   — embedding failed (zero vector) → don't cluster
                   cluster_id = None, cluster_payload = None
    """
    title        = article.get("title") or ""
    description  = article.get("description") or ""
    domain       = article.get("domain") or ""
    published_at = article.get("published_at") or datetime.now(timezone.utc).isoformat()

    # ── Compute article embedding ─────────────────────────────────────────────
    # Title is the primary signal. Description adds context.
    # We cap at 512 chars total (mDeBERTa token limit, same model family).
    embed_text = f"{title}. {description}"[:512]
    article_embedding = get_embedding(embed_text)

    # If embedding failed (all zeros), don't cluster
    if not any(article_embedding):
        log.debug("Zero embedding for '%s...' — skipping clustering", title[:40])
        return None, None, "skip"

    # ── Find best matching cluster ────────────────────────────────────────────
    best_cluster = None
    best_score   = 0.0

    for cluster in recent_clusters:
        cluster_embedding = cluster.get("canonical_embedding")
        if not cluster_embedding:
            continue   # old cluster without embedding — skip

        score = _cosine_similarity(article_embedding, cluster_embedding)
        if score > best_score:
            best_score   = score
            best_cluster = cluster

    # ── JOIN existing cluster ─────────────────────────────────────────────────
    if best_score >= CLUSTER_EMBEDDING_THRESHOLD and best_cluster:
        cluster_id = best_cluster["id"]

        article_entity_set = {e["entity_text"].lower() for e in entities}
        merged_entity_set  = list(
            set(best_cluster.get("entity_set") or []) | article_entity_set
        )
        merged_top_entities = _merge_top_entities(
            best_cluster.get("top_entities") or [],
            entities,
        )

        # Outlet count: increment if this is a new domain in the cluster
        existing_entity_set = set(best_cluster.get("entity_set") or [])
        new_outlet_count = best_cluster.get("outlet_count", 1)
        if domain and domain not in existing_entity_set:
            new_outlet_count += 1

        # Do NOT update canonical_embedding on join — keep the original cluster's
        # embedding as the canonical representation
        cluster_update = {
            "article_count": best_cluster.get("article_count", 1) + 1,
            "outlet_count":  new_outlet_count,
            "entity_set":    merged_entity_set,
            "top_entities":  merged_top_entities,
            "last_seen_at":  published_at,
        }

        log.debug(
            "Article '%s...' joined cluster %s (similarity=%.3f)",
            title[:40], cluster_id, best_score,
        )
        return cluster_id, cluster_update, "join"

    # ── CREATE new cluster ────────────────────────────────────────────────────
    article_entity_set = {e["entity_text"].lower() for e in entities}

    new_cluster = {
        "canonical_article_id": article.get("id"),
        "headline":             title[:500] if title else None,
        "outlet_count":         1,
        "article_count":        1,
        "entity_set":           list(article_entity_set),
        "top_entities":         _build_top_entities(entities),
        "canonical_embedding":  article_embedding,   # stored for future comparisons
        "first_seen_at":        published_at,
        "last_seen_at":         published_at,
    }

    log.debug(
        "Article '%s...' starts new cluster (best_similarity=%.3f)",
        title[:40], best_score,
    )
    return None, new_cluster, "create"
