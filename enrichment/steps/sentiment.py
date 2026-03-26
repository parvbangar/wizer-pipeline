"""
enrichment/steps/sentiment.py
══════════════════════════════
Classifies article sentiment as positive, negative, or neutral using VADER.

WHY SENTIMENT MATTERS FOR AN INDIAN NEWS APP:
  - Users engage differently with positive vs negative news
  - Negative news (crime, disaster) has higher short-term virality
    but lower long-term user retention
  - Sentiment is a feature in propensity scoring: pure doom-scroll
    feeds burn users out; balanced feeds retain them
  - Useful for brand-safety filtering if you add sponsored content

WHY VADER (not a transformer)?
  VADER (Valence Aware Dictionary and sEntiment Reasoner) is:
    - Trained specifically on news headlines and social media
    - Fast: dictionary lookup, no GPU needed
    - Good for short texts (headlines + leads)
    - Pre-built for English — no training needed

  LIMITATION: English only.
  Hindi, Tamil, Telugu articles get sentiment = NULL, sentiment_score = NULL.
  This is honest. A future upgrade can add MuRIL (Multilingual RoBERTa for
  Indian Languages) for regional language sentiment.

SENTIMENT VALUES:
  'positive'  — compound score ≥  0.05
  'negative'  — compound score ≤ -0.05
  'neutral'   — compound score between -0.05 and +0.05

INSTALL: pip install vaderSentiment
"""

from __future__ import annotations

import logging

from enrichment.config import (
    SENTIMENT_POSITIVE_THRESHOLD,
    SENTIMENT_NEGATIVE_THRESHOLD,
    SENTIMENT_ENGLISH_ONLY_LANGS,
)

log = logging.getLogger(__name__)

# VADER analyser is loaded once per process
_analyser = None


def _get_analyser():
    global _analyser
    if _analyser is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _analyser = SentimentIntensityAnalyzer()
        except ImportError:
            raise ImportError(
                "vaderSentiment not installed. Run: pip install vaderSentiment"
            )
    return _analyser


def analyse_sentiment(
    title: str,
    description: str,
    language_detected: str | None,
) -> dict:
    """
    Compute sentiment for an article.

    Only runs on English articles — returns NULL values for other languages.

    Analysis text: title + description (not full_text — VADER is optimised
    for short, punchy text like headlines, not long-form prose).

    Args:
      title:             Article headline
      description:       RSS description / summary
      language_detected: ISO 639-1 code detected by language.py step
                         (e.g. 'en', 'hi', 'ta'). None if detection failed.

    Returns dict with:
      sentiment        (str | None)   — 'positive', 'negative', 'neutral', or None
      sentiment_score  (float | None) — compound score -1.0 to +1.0, or None
    """
    null_result = {"sentiment": None, "sentiment_score": None}

    # Only analyse English text — VADER is not reliable on other languages
    lang = (language_detected or "").lower()
    if lang and lang not in SENTIMENT_ENGLISH_ONLY_LANGS:
        return null_result

    text = f"{title or ''} {description or ''}".strip()
    if not text:
        return null_result

    try:
        analyser = _get_analyser()
        scores   = analyser.polarity_scores(text)
        compound = scores["compound"]

        if compound >= SENTIMENT_POSITIVE_THRESHOLD:
            label = "positive"
        elif compound <= SENTIMENT_NEGATIVE_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {
            "sentiment":       label,
            "sentiment_score": round(compound, 4),
        }
    except ImportError as e:
        log.warning("Sentiment skipped: %s", e)
        return null_result
    except Exception as e:
        log.debug("Sentiment analysis failed: %s", e)
        return null_result
