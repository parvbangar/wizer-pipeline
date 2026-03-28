"""
enrichment/steps/propensity.py
════════════════════════════════
Estimates the virality / engagement propensity of a news article.

WHY PROPENSITY SCORING:
  Not all news is equal. "Modi announces budget" reported by 87 outlets in
  4 hours has 100x the engagement signal of a local municipal board story.
  Propensity score (0.0–1.0) gives the app layer a single number to:
    - Rank the feed (high propensity = top of feed)
    - Gate breaking-news notifications (score > 0.70 → push notification)
    - Filter sponsored content placement (avoid next to high-drama stories)
    - Feed the recommendation model as a feature

SELF-SUPERVISED TRAINING SIGNAL:
  outlet_count on article_clusters is the ground truth for importance.
  If 50 independent outlets pick up the same story, it's genuinely important.
  PTI/ANI wire stories about major events regularly hit outlet_count > 30.
  LightGBM is trained weekly on this label — no human annotation needed.

TWO-PHASE ARCHITECTURE:

  Phase 1 — weighted formula (runs immediately, no model needed):
    score = 0.40 × outlet_virality    (log-normalised outlet_count)
          + 0.25 × category_weight    (cricket=0.90, general=0.30)
          + 0.20 × entity_signal      (avg salience of top-5 entities)
          + 0.15 × recency_factor     (exp decay over 24h)

  Phase 2 — LightGBM regression (loads from models/propensity_model.lgb):
    Same features fed to a gradient-boosted tree trained on log(1 + outlet_count).
    Output squashed to [0, 1] via log-normalisation.
    Falls back to Phase 1 if model file is missing.

FEATURE VECTOR (same for formula + ML, ensuring consistency):
  [word_count, category_encoded, is_english, sentiment_num, sentiment_score]

  word_count:       int   — article length (longer = more substance)
  category_encoded: int   — cricket=0, politics=1, ... general=12
  is_english:       0/1   — English articles have richer NER features
  sentiment_num:    -1/0/1 — negative/neutral/positive
  sentiment_score:  float  — compound score from sentiment step
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from enrichment.config import (
    PROPENSITY_MODEL_PATH,
    PROPENSITY_MAX_OUTLET_COUNT,
    PROPENSITY_CATEGORY_WEIGHTS,
)

log = logging.getLogger(__name__)

# ── LightGBM model cache ──────────────────────────────────────────────────────
_lgb_model       = None
_lgb_model_tried = False   # avoid retrying a missing model every article

# Feature encoding maps (must match train_propensity.py)
_CATEGORY_ENCODE = {
    "cricket":       0,
    "politics":      1,
    "business":      2,
    "entertainment": 3,
    "technology":    4,
    "sports":        5,
    "health":        6,
    "world":         7,
    "crime":         8,
    "education":     9,
    "environment":   10,
    "crypto":        11,
    "general":       12,
}
_SENTIMENT_ENCODE = {"positive": 1, "neutral": 0, "negative": -1}

# log(1 + MAX) — denominator for normalising LightGBM raw output
_LOG_MAX = math.log1p(PROPENSITY_MAX_OUTLET_COUNT)


def _get_lgb_model():
    """Load LightGBM model once; return None if file missing or load fails."""
    global _lgb_model, _lgb_model_tried
    if _lgb_model_tried:
        return _lgb_model
    _lgb_model_tried = True

    model_path = Path(PROPENSITY_MODEL_PATH)
    if not model_path.exists():
        log.debug("No propensity model at %s — using weighted formula", model_path)
        return None

    try:
        import lightgbm as lgb
        _lgb_model = lgb.Booster(model_file=str(model_path))
        log.info("Propensity LightGBM model loaded from %s", model_path)
    except Exception as e:
        log.warning("Failed to load propensity model: %s — using formula", e)
    return _lgb_model


def _build_features(
    update: dict,
    entities: list[dict],
) -> list[float]:
    """
    Build the feature vector used by both the formula and LightGBM.

    Args:
      update:   Current enrichment update dict (has category, word_count, etc.)
      entities: NER entity list from the NER step

    Returns a 5-element list matching the training feature order.
    """
    category     = update.get("category") or "general"
    language     = (update.get("language_detected") or "").lower()
    sentiment    = update.get("sentiment") or "neutral"
    sent_score   = float(update.get("sentiment_score") or 0.0)

    top5_salience = sorted(
        (e.get("salience", 0.0) for e in entities),
        reverse=True,
    )[:5]
    avg_salience = sum(top5_salience) / len(top5_salience) if top5_salience else 0.0

    return [
        float(update.get("word_count") or 0),
        float(_CATEGORY_ENCODE.get(category, 12)),
        1.0 if language.startswith("en") else 0.0,
        float(_SENTIMENT_ENCODE.get(sentiment, 0)),
        sent_score,
        avg_salience,
    ]


def _formula_score(
    features: list[float],
    outlet_count: int,
    published_at: str | None,
) -> float:
    """
    Phase 1 weighted formula.

    Weights calibrated on Indian news wire syndication patterns:
      outlet_virality — strongest signal; accounts for 40% of score
      category_weight — cricket/politics dominate Indian news engagement
      entity_signal   — articles with salient named entities rank higher
      recency_factor  — news decays fast; fresh articles score higher
    """
    word_count, category_enc, is_english, sentiment_num, sent_score, avg_sal = features

    # Outlet virality: log-normalised (0 to 1)
    outlet_virality = math.log1p(min(outlet_count, PROPENSITY_MAX_OUTLET_COUNT)) / _LOG_MAX

    # Category weight
    category_name = next(
        (k for k, v in _CATEGORY_ENCODE.items() if v == int(category_enc)),
        "general",
    )
    cat_weight = PROPENSITY_CATEGORY_WEIGHTS.get(category_name, 0.30)

    # Entity signal: avg salience of top 5 entities
    entity_signal = min(avg_sal, 1.0)

    # Recency: exponential decay over 24 hours (1.0 for just-published)
    recency = 1.0
    if published_at:
        try:
            pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            hours_old = max(0.0, (datetime.now(timezone.utc) - pub).total_seconds() / 3600)
            recency = math.exp(-hours_old / 24.0)
        except Exception:
            pass

    score = (
        0.40 * outlet_virality +
        0.25 * cat_weight +
        0.20 * entity_signal +
        0.15 * recency
    )
    return min(round(score, 4), 1.0)


def compute_propensity(
    article: dict,
    update: dict,
    entities: list[dict],
    cluster_payload: dict | None,
    cluster_action: str,
) -> float:
    """
    Compute the propensity score (0.0–1.0) for an article.

    Uses Phase 2 (LightGBM) if the model file exists, otherwise Phase 1
    (weighted formula). Both use the same feature vector for consistency.

    Args:
      article:         Article dict from DB
      update:          Current enrichment update dict
      entities:        NER entities list
      cluster_payload: Cluster update/create dict (has outlet_count), or None
      cluster_action:  'join' | 'create' | 'skip'

    Returns:
      Float in [0.0, 1.0]. Higher = more likely to go viral.
    """
    # Determine outlet_count from clustering results
    if cluster_action == "join" and cluster_payload:
        outlet_count = int(cluster_payload.get("outlet_count") or 1)
    elif cluster_action == "create":
        outlet_count = 1
    else:
        outlet_count = 1

    features    = _build_features(update, entities)
    published_at = article.get("published_at")

    # ── Phase 2: LightGBM ─────────────────────────────────────────────────────
    model = _get_lgb_model()
    if model is not None:
        try:
            raw_pred = model.predict([features])[0]
            # raw_pred ≈ log(1 + outlet_count) → normalise to [0, 1]
            score = min(float(raw_pred) / _LOG_MAX, 1.0)
            # Blend with recency so stale articles don't score artificially high
            recency = 1.0
            if published_at:
                try:
                    pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    hours_old = max(0.0, (datetime.now(timezone.utc) - pub).total_seconds() / 3600)
                    recency = math.exp(-hours_old / 24.0)
                except Exception:
                    pass
            score = round(score * 0.85 + recency * 0.15, 4)
            return min(score, 1.0)
        except Exception as e:
            log.debug("LightGBM propensity prediction failed: %s — using formula", e)

    # ── Phase 1: Weighted formula ─────────────────────────────────────────────
    return _formula_score(features, outlet_count, published_at)
