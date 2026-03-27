"""
enrichment/steps/classifier.py
════════════════════════════════
Classifies each article into an Indian-news-specific category using
zero-shot NLI classification with mDeBERTa.

MODEL: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
  State-of-the-art multilingual NLI model. Works by framing classification
  as a Natural Language Inference problem:
    "Does this article ENTAIL the hypothesis: this article is about [category]?"
  The category with the highest entailment score wins.

WHY THIS OVER RULE-BASED KEYWORDS?
  - Understands CONTEXT, not just keywords.
    "Modi launches new app" → politics (not technology), because the model
    understands that a Prime Minister launching something is political news.
  - Works in ALL Indian languages natively.
    A Hindi article saying "बजट में बड़ा बदलाव" is correctly classified
    as business without any Hindi keywords in your config.
  - No false positives from substring collisions.
    "odi" in "commodity" was a bug. The model reads meaning, not substrings.
  - Zero training data required — just define what the categories mean.

HOW ZERO-SHOT CLASSIFICATION WORKS:
  1. The model receives: article_text + each candidate label
  2. For each label it computes: P(article entails "this is about [label]")
  3. Returns labels sorted by that probability
  4. We return the highest-scoring label above CLASSIFY_CONFIDENCE_THRESHOLD

CANDIDATE LABELS:
  Labels are descriptive English phrases. mDeBERTa is cross-lingual —
  it maps a Tamil article and an English label into the same semantic space,
  so English labels work correctly regardless of article language.

PERFORMANCE:
  ~150-200ms per article on CPU (GitHub Actions 2 vCPU).
  Model is loaded once and cached for the entire batch (~560 MB).
  On a 500-article batch: ~1.5 min of classification time.

INSTALL:
  pip install transformers torch
"""

from __future__ import annotations

import logging

from enrichment.config import DEBERTA_MODEL, CLASSIFY_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

# ── Model cache — loaded once per process ─────────────────────────────────────
_pipeline = None


def _get_pipeline():
    """
    Load and cache the zero-shot classification pipeline.

    Uses device=-1 (CPU) which is what GitHub Actions runners provide.
    Loading takes ~3-5 seconds and uses ~1.2 GB RAM. Subsequent calls
    reuse the cached pipeline instantly.
    """
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            log.info("Loading mDeBERTa classification model '%s'...", DEBERTA_MODEL)
            _pipeline = pipeline(
                "zero-shot-classification",
                model=DEBERTA_MODEL,
                device=-1,       # -1 = CPU
                multi_label=False,
            )
            log.info("mDeBERTa model loaded")
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )
    return _pipeline


# ── Category labels ────────────────────────────────────────────────────────────
#
# Labels are English phrases that best describe each category.
# Descriptive phrases work better than single words — "politics and elections"
# is a clearer hypothesis than just "politics".
#
# The map converts the winning label back to the short category key stored in DB.

_CANDIDATE_LABELS = [
    "cricket match IPL BCCI batting bowling",
    "politics elections government parliament minister",
    "business stock market economy finance investment",
    "entertainment movies bollywood music celebrity",
    "technology artificial intelligence software internet",
    "sports football hockey badminton tennis olympics",
    "health medicine disease hospital doctor vaccine",
    "education school college exam university",
    "crime arrest police court murder fraud",
    "environment climate pollution flood earthquake",
    "international news foreign affairs diplomacy",
    "cryptocurrency bitcoin blockchain digital currency",
]

_LABEL_TO_CATEGORY: dict[str, str] = {
    "cricket match IPL BCCI batting bowling":              "cricket",
    "politics elections government parliament minister":    "politics",
    "business stock market economy finance investment":     "business",
    "entertainment movies bollywood music celebrity":       "entertainment",
    "technology artificial intelligence software internet": "technology",
    "sports football hockey badminton tennis olympics":     "sports",
    "health medicine disease hospital doctor vaccine":      "health",
    "education school college exam university":             "education",
    "crime arrest police court murder fraud":               "crime",
    "environment climate pollution flood earthquake":       "environment",
    "international news foreign affairs diplomacy":         "world",
    "cryptocurrency bitcoin blockchain digital currency":   "crypto",
}


def classify_article(
    title: str,
    description: str,
    iab_tier1: str,
    full_text: str = "",
) -> str:
    """
    Return the best-matching category for an article using zero-shot NLI.

    Args:
      title:       Article headline (most informative signal)
      description: RSS description / summary
      iab_tier1:   Feed-level IAB category (used as fallback tiebreaker)
      full_text:   Article body — first 200 chars used for extra context

    Returns:
      Category string: cricket | politics | business | entertainment |
                       technology | sports | health | education | crime |
                       environment | world | crypto | general
    """
    # Build the text to classify.
    # Title is the strongest signal. Description and start of full_text
    # provide additional context without exceeding mDeBERTa's token limit.
    text = f"{title}. {description} {(full_text or '')[:200]}".strip()

    if not text:
        return "general"

    try:
        clf = _get_pipeline()
        result = clf(
            text[:512],           # mDeBERTa tokenizer limit
            candidate_labels=_CANDIDATE_LABELS,
            multi_label=False,
        )

        best_label = result["labels"][0]
        best_score = result["scores"][0]

        log.debug(
            "classify: '%s...' → %s (score=%.2f)",
            title[:50], _LABEL_TO_CATEGORY.get(best_label, "general"), best_score,
        )

        if best_score < CLASSIFY_CONFIDENCE_THRESHOLD:
            log.debug("Low confidence (%.2f) → defaulting to 'general'", best_score)
            return "general"

        return _LABEL_TO_CATEGORY.get(best_label, "general")

    except Exception as e:
        log.warning("Classification failed (%s) — returning 'general'", e)
        return "general"
