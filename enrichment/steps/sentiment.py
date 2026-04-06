"""
enrichment/steps/sentiment.py
══════════════════════════════
Multilingual sentiment using distilbert-base-multilingual-cased-sentiments-student.

WHY THIS UPGRADE FROM VADER:
  VADER was English-only — 40–50% of Indian news articles (Hindi, Tamil,
  Telugu, Bengali) got sentiment = NULL. The multilingual distilbert model
  covers all major Indian languages while staying small (268 MB) and fast
  on CPU (~50 ms/article on a 2-vCPU runner).

MODEL: lxyuan/distilbert-base-multilingual-cased-sentiments-student
  - Distilled from mDeBERTa-v3; strong multilingual sentiment accuracy
  - Native: EN, AR, DE, JA, ES, FR, ZH
  - Generalises: Hindi, Tamil, Telugu, Bengali via multilingual pretraining
  - ~268 MB download, cached in ~/.cache/huggingface/hub after first run
  - Returns: 'positive', 'neutral', 'negative' label probabilities

SENTIMENT_SCORE:
  positive_probability − negative_probability → −1.0 to +1.0
  Matches VADER compound score semantics for backward compatibility.
"""

from __future__ import annotations

import logging

from enrichment.config import SENTIMENT_MULTILINGUAL_MODEL

log = logging.getLogger(__name__)

# Pipeline loaded once per process, reused across the batch
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            log.info("Loading sentiment model '%s'...", SENTIMENT_MULTILINGUAL_MODEL)
            _pipeline = hf_pipeline(
                "text-classification",
                model=SENTIMENT_MULTILINGUAL_MODEL,
                top_k=None,   # return scores for all labels
            )
            log.info("Sentiment model loaded")
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
    return _pipeline


def analyse_sentiment(
    title: str,
    description: str,
    language_detected: str | None,
) -> dict:
    """
    Compute sentiment for an article using a multilingual transformer.

    Works on all languages — English, Hindi, Tamil, Telugu, Bengali, etc.
    Falls back to NULL on error or empty text.

    Analysis text: title + description (capped at 512 tokens) — short text
    is where sentiment signal is strongest in news articles.

    Args:
      title:             Article headline
      description:       RSS description / summary
      language_detected: ISO language code (unused — model is multilingual,
                         kept for interface compatibility with runner.py)

    Returns dict with:
      sentiment        (str | None)   — 'positive', 'negative', 'neutral', or None
      sentiment_score  (float | None) — positive_prob − negative_prob, −1.0 to +1.0
    """
    null_result = {"sentiment": None, "sentiment_score": None}

    # Title + description is where sentiment signal is strongest in news.
    # We pass up to ~1000 chars (≈200 tokens) — well within the 512-token limit.
    # truncation=True ensures the pipeline handles long inputs safely.
    text = f"{title or ''}. {description or ''}".strip()
    if not text:
        return null_result

    try:
        pipe    = _get_pipeline()
        results = pipe(text[:1000], truncation=True, max_length=512)
        # top_k=None → [[{'label': 'positive', 'score': 0.9}, ...]]
        scores    = results[0] if results else []
        score_map = {r["label"]: r["score"] for r in scores}

        if not score_map:
            return null_result

        top_label = max(score_map, key=score_map.get)
        pos       = score_map.get("positive", 0.0)
        neg       = score_map.get("negative", 0.0)
        compound  = round(pos - neg, 4)

        return {
            "sentiment":       top_label,
            "sentiment_score": compound,
        }
    except ImportError as e:
        log.warning("Sentiment skipped: %s", e)
        return null_result
    except Exception as e:
        log.debug("Sentiment analysis failed: %s", e)
        return null_result
