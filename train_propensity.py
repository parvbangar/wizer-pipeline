"""
train_propensity.py
════════════════════
Weekly training script for the LightGBM propensity scoring model.

SELF-SUPERVISED APPROACH:
  No manual labels needed. outlet_count on article_clusters is the training
  signal: if 50 independent outlets reported the same story, it was genuinely
  important news. LightGBM learns to predict this from article-level features
  available at enrichment time.

TRAINING DATA:
  All articles with:
    - enriched_at IS NOT NULL   (fully processed)
    - cluster_id  IS NOT NULL   (has a cluster with outlet_count)
    - word_count  IS NOT NULL   (has enrichment signal)

  Fetched in batches to handle large tables without memory issues.

FEATURES (must match enrichment/steps/propensity.py _build_features()):
  word_count:       article length (proxy for substance)
  category_encoded: category as int (cricket=0 ... general=12)
  is_english:       1 if language starts with "en", else 0
  sentiment_num:    -1 / 0 / 1 (negative / neutral / positive)
  sentiment_score:  compound score -1.0 to +1.0
  avg_top5_sal:     average salience of top-5 entities (from article_entities)

TARGET:
  log(1 + outlet_count) — log-transform reduces skew from PTI stories with
  outlet_count=200 dominating the loss function.

OUTPUT:
  models/propensity_model.lgb — LightGBM native text format (~200–500 KB).
  Committed to the repo and loaded by the enrichment pipeline.

USAGE:
  python train_propensity.py [--min-rows 500] [--output models/propensity_model.lgb]

  GitHub Actions runs this weekly via .github/workflows/train_propensity.yml.
  It commits the updated model file and pushes to main.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("train_propensity")

# ── Feature encoding (must match propensity.py) ───────────────────────────────
CATEGORY_ENCODE = {
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
SENTIMENT_ENCODE = {"positive": 1, "neutral": 0, "negative": -1}

_PAGE_SIZE = 1000


# ── Data fetching ─────────────────────────────────────────────────────────────

def _get_client():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key)


def fetch_training_rows(min_rows: int) -> list[dict]:
    """
    Fetch articles with cluster outlet_count for training.

    Joins articles → article_entities to get avg entity salience.
    Returns list of dicts with feature columns + outlet_count label.
    """
    client = _get_client()
    rows: list[dict] = []
    offset = 0

    log.info("Fetching training data (this may take a minute)...")

    while True:
        try:
            resp = (
                client
                .table("articles")
                .select(
                    "id, word_count, category, language_detected, "
                    "sentiment, sentiment_score, cluster_id, "
                    "article_clusters!inner(outlet_count)"
                )
                .not_.is_("enriched_at", "null")
                .not_.is_("cluster_id", "null")
                .not_.is_("word_count", "null")
                .range(offset, offset + _PAGE_SIZE - 1)
                .execute()
            )
        except Exception as e:
            log.error("Failed to fetch training batch at offset %d: %s", offset, e)
            break

        page = resp.data or []
        rows.extend(page)
        log.info("  Fetched %d rows (total: %d)", len(page), len(rows))

        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    log.info("Total training rows: %d", len(rows))
    return rows


def fetch_entity_salience(article_ids: list[str]) -> dict[str, float]:
    """
    Fetch average top-5 entity salience per article from article_entities.
    Returns {article_id: avg_salience}.
    """
    client = _get_client()
    salience_map: dict[str, list[float]] = {}

    # Fetch in batches of 500 article IDs
    for i in range(0, len(article_ids), 500):
        batch_ids = article_ids[i:i + 500]
        try:
            resp = (
                client
                .table("article_entities")
                .select("article_id, salience")
                .in_("article_id", batch_ids)
                .execute()
            )
            for row in (resp.data or []):
                aid = row["article_id"]
                salience_map.setdefault(aid, []).append(float(row.get("salience") or 0.0))
        except Exception as e:
            log.warning("Entity salience fetch failed for batch: %s", e)

    # Average of top-5 per article
    return {
        aid: sum(sorted(sals, reverse=True)[:5]) / min(5, len(sals))
        for aid, sals in salience_map.items()
    }


# ── Feature/label extraction ──────────────────────────────────────────────────

def build_features_label(
    row: dict,
    salience_map: dict[str, float],
) -> tuple[list[float], float] | None:
    """
    Build (features, label) for one training row.
    Returns None if the row has unusable data.
    """
    cluster_data = row.get("article_clusters") or {}
    outlet_count = cluster_data.get("outlet_count") if isinstance(cluster_data, dict) else None
    if outlet_count is None:
        return None

    category   = row.get("category") or "general"
    language   = (row.get("language_detected") or "").lower()
    sentiment  = row.get("sentiment") or "neutral"
    sent_score = float(row.get("sentiment_score") or 0.0)
    word_count = float(row.get("word_count") or 0)
    avg_sal    = salience_map.get(row.get("id", ""), 0.0)

    features = [
        word_count,
        float(CATEGORY_ENCODE.get(category, 12)),
        1.0 if language.startswith("en") else 0.0,
        float(SENTIMENT_ENCODE.get(sentiment, 0)),
        sent_score,
        avg_sal,
    ]
    label = math.log1p(max(0, int(outlet_count)))
    return features, label


# ── Training ──────────────────────────────────────────────────────────────────

def train(min_rows: int, output_path: str) -> bool:
    """
    Fetch data, train LightGBM model, save to output_path.
    Returns True on success.
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        log.error("lightgbm or numpy not installed. Run: pip install lightgbm numpy")
        return False

    rows = fetch_training_rows(min_rows)
    if len(rows) < min_rows:
        log.warning(
            "Only %d training rows available (need %d) — skipping training",
            len(rows), min_rows,
        )
        return False

    article_ids  = [r.get("id", "") for r in rows]
    salience_map = fetch_entity_salience(article_ids)

    X, y = [], []
    for row in rows:
        result = build_features_label(row, salience_map)
        if result is not None:
            features, label = result
            X.append(features)
            y.append(label)

    if len(X) < min_rows:
        log.warning("Only %d usable rows after feature extraction — skipping", len(X))
        return False

    log.info("Training LightGBM on %d rows...", len(X))

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_arr, y_arr)

    # Save model
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(output))
    log.info("Model saved to %s", output)

    # Feature importance summary
    feature_names = [
        "word_count", "category_enc", "is_english",
        "sentiment_num", "sentiment_score", "avg_top5_salience",
    ]
    importances = model.booster_.feature_importance(importance_type="gain")
    log.info("Feature importances (gain):")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        log.info("  %-22s %.1f", name, imp)

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train propensity scoring model")
    parser.add_argument(
        "--min-rows", type=int, default=500,
        help="Minimum training rows required (default: 500)",
    )
    parser.add_argument(
        "--output", default="models/propensity_model.lgb",
        help="Output path for trained model (default: models/propensity_model.lgb)",
    )
    args = parser.parse_args()

    success = train(args.min_rows, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
