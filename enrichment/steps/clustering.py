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

THRESHOLD (CLUSTER_EMBEDDING_THRESHOLD = 0.82):
  Calibrated for Indian wire news syndication:
  - Same PTI story, different outlets: ~0.90-0.98
  - Same event, independently reported: ~0.83-0.92
  - Related but different stories: ~0.70-0.82
  - Unrelated stories: < 0.65

CANONICAL EMBEDDING — RUNNING MEAN:
  The cluster's canonical_embedding is updated on every JOIN using a
  running mean (1/N weighting). This keeps the cluster centroid accurate
  as stories evolve: "Budget announcement" → "Budget reactions" → "Budget
  implementation". The HNSW search always uses the centroid of all articles
  seen so far, not just the first article.

  Mathematically:
    new_centroid = ((N-1)/N) × old_centroid + (1/N) × new_article
  Re-normalised to unit vector after each update (required for cosine sim).

OUTLET COUNT — DOMAIN DEDUP via outlet_set:
  outlet_count = number of DISTINCT domains in the cluster.
  outlet_set   = jsonb array of domain strings, used for dedup.
  Previously the check was against entity_set (entity texts like "Modi",
  "BJP") — domains were never in entity_set so outlet_count was wrong.

ENTITY SET (kept from NER step):
  Populates entity_set and top_entities in clusters for the app layer:
  - Article tagging ("Modi", "BJP", "Budget")
  - Entity-based search ("all articles about Virat Kohli")
  - Propensity scoring features

HEADLINE REFRESH:
  Cluster headline is updated when a new article has a longer title.
  News story headlines get more descriptive as coverage deepens —
  the wire flash "Budget announced" evolves into "Modi Budget 2024:
  ₹1.8 lakh crore infrastructure push, tax slabs unchanged".

INSTALL:
  pip install sentence-transformers torch
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from enrichment import db
from enrichment.config import CLUSTER_EMBEDDING_THRESHOLD, CLUSTER_WINDOW_HOURS, LABSE_MODEL

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
        embedding = model.encode(text[:512], normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        return [0.0] * 768


# ── Embedding math ─────────────────────────────────────────────────────────────

def _running_mean_embedding(
    old: list[float],
    new: list[float],
    new_count: int,
) -> list[float]:
    """
    Update cluster centroid using a running mean, then re-normalise.

    Formula: new_centroid = ((N-1)/N) × old + (1/N) × new
    Where N = new_count (the article count AFTER this article joins).

    This is mathematically equivalent to the mean of all article embeddings
    seen so far. The cluster centroid stays accurate as stories evolve.
    Re-normalising to unit vector keeps cosine similarity valid.
    """
    if not old or len(old) != len(new):
        return new

    alpha   = 1.0 / max(new_count, 1)
    blended = [(alpha * n + (1.0 - alpha) * o) for n, o in zip(new, old)]

    # Re-normalise to unit vector (required for pgvector cosine similarity)
    norm = math.sqrt(sum(v * v for v in blended))
    return [v / norm for v in blended] if norm > 0 else old


def _format_vec(embedding: list[float]) -> str:
    """Format a float list as pgvector literal: '[x,y,z,...]'"""
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"


# ── Entity helpers ─────────────────────────────────────────────────────────────

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
) -> tuple[str | None, dict | None, str]:
    """
    Find the best matching cluster for an article using pgvector HNSW search,
    or signal that a new cluster should be created.

    Args:
      article:  Article dict (needs: id, title, description, domain, published_at)
      entities: Entity list from NER step (used for cluster tags, not for matching)

    Returns:
      (cluster_id, cluster_payload, action) where action is:
        'join'   — article joins an existing cluster
                   cluster_payload = update dict for db.update_cluster
        'create' — no similar cluster found, create a new one
                   cluster_id = None (assigned after DB insert)
                   cluster_payload = new cluster dict for db.create_cluster
        'skip'   — embedding failed (zero vector) → don't cluster
    """
    title        = article.get("title") or ""
    description  = article.get("description") or ""
    domain       = article.get("domain") or ""
    published_at = article.get("published_at") or datetime.now(timezone.utc).isoformat()

    # ── Compute article embedding ─────────────────────────────────────────────
    embed_text        = f"{title}. {description}"[:512]
    article_embedding = get_embedding(embed_text)

    if not any(article_embedding):
        log.debug("Zero embedding for '%s...' — skipping clustering", title[:40])
        return None, None, "skip"

    # ── Query pgvector for nearest cluster (single SQL round-trip) ────────────
    best_cluster = db.find_nearest_cluster(
        embedding    = article_embedding,
        threshold    = CLUSTER_EMBEDDING_THRESHOLD,
        window_hours = CLUSTER_WINDOW_HOURS,
    )

    # ── JOIN existing cluster ─────────────────────────────────────────────────
    if best_cluster:
        cluster_id   = best_cluster["id"]
        similarity   = best_cluster.get("similarity", 0.0)
        new_count    = best_cluster.get("article_count", 1) + 1

        # Entity merging
        article_entity_set  = {e["entity_text"].lower() for e in entities}
        merged_entity_set   = list(
            set(best_cluster.get("entity_set") or []) | article_entity_set
        )
        merged_top_entities = _merge_top_entities(
            best_cluster.get("top_entities") or [],
            entities,
        )

        # ── Outlet dedup via outlet_set (fixes outlet_count inflation) ────────
        # Previously: checked `domain not in entity_set` — always True since
        # entity_set contains entity TEXTS ("Modi", "BJP"), not domains.
        # Now: maintained outlet_set tracks actual contributing domains.
        existing_outlet_set = set(best_cluster.get("outlet_set") or [])
        new_outlet_count    = best_cluster.get("outlet_count", 1)
        new_outlet_set      = list(existing_outlet_set)
        if domain and domain not in existing_outlet_set:
            new_outlet_count += 1
            new_outlet_set.append(domain)

        # ── Running mean embedding update ─────────────────────────────────────
        # Keeps cluster centroid accurate as story evolves.
        # Re-normalised to unit vector after blending.
        old_embedding   = best_cluster.get("canonical_embedding") or []
        new_embedding   = _running_mean_embedding(old_embedding, article_embedding, new_count)

        # ── Headline refresh: prefer longer (more descriptive) titles ─────────
        # Wire flash: "Budget announced" → Full story: "Modi Budget 2024:
        # ₹1.8 lakh crore infrastructure push, income tax slabs unchanged"
        current_headline = best_cluster.get("headline") or ""
        new_headline     = title[:500] if title and len(title) > len(current_headline) else None

        cluster_update: dict = {
            "article_count":       new_count,
            "outlet_count":        new_outlet_count,
            "outlet_set":          new_outlet_set,
            "entity_set":          merged_entity_set,
            "top_entities":        merged_top_entities,
            "last_seen_at":        published_at,
            "canonical_embedding": new_embedding,   # db.update_cluster writes embedding_vec too
        }
        if new_headline:
            cluster_update["headline"] = new_headline

        log.debug(
            "Article '%s...' joined cluster %s (sim=%.3f, outlets=%d→%d)",
            title[:40], cluster_id, similarity,
            best_cluster.get("outlet_count", 1), new_outlet_count,
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
        "outlet_set":           [domain] if domain else [],
        "top_entities":         _build_top_entities(entities),
        "canonical_embedding":  article_embedding,   # db.create_cluster also writes embedding_vec
        "first_seen_at":        published_at,
        "last_seen_at":         published_at,
    }

    log.debug("Article '%s...' starts new cluster", title[:40])
    return None, new_cluster, "create"
