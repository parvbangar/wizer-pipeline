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
# Labels are written as NLI hypothesis completions.
# The transformers zero-shot pipeline internally forms:
#   "This example is [label]."
# So labels must read naturally in that template.
#
# Keyword-soup labels ("cricket IPL T20 batting...") do NOT work well because
# the model is trained on natural-language entailment, not keyword matching.
# "about cricket or the IPL" reads as a proper hypothesis; a keyword list does not.
#
# Labels are kept distinct so the softmax spread is clean:
#   - "cricket" is separate from "sports" to avoid splitting cricket traffic
#   - "international news" is scoped narrowly to avoid absorbing everything that
#     mentions a foreign country (which caused the 37% "world" overcounting bug)

_CANDIDATE_LABELS = [
    "about cricket, the IPL, or a cricket match or tournament",
    "about Indian politics, elections, parliament, or government policy",
    "about business, the economy, stock markets, or corporate finance",
    "about Bollywood, movies, music, entertainment, or Indian celebrities",
    "about technology, software, artificial intelligence, or tech startups",
    "about sports other than cricket, such as football, hockey, kabaddi, or athletics",
    "about health, medicine, hospitals, disease, or public healthcare",
    "about education, schools, colleges, university exams, or student affairs",
    "about crime, police, courts, arrests, or legal investigations",
    "about the environment, climate change, floods, droughts, or pollution",
    "about international relations, foreign affairs, or events outside India",
    "about cryptocurrency, bitcoin, blockchain, or digital currency",
]

_LABEL_TO_CATEGORY: dict[str, str] = {
    "about cricket, the IPL, or a cricket match or tournament":                             "cricket",
    "about Indian politics, elections, parliament, or government policy":                   "politics",
    "about business, the economy, stock markets, or corporate finance":                     "business",
    "about Bollywood, movies, music, entertainment, or Indian celebrities":                 "entertainment",
    "about technology, software, artificial intelligence, or tech startups":                "technology",
    "about sports other than cricket, such as football, hockey, kabaddi, or athletics":     "sports",
    "about health, medicine, hospitals, disease, or public healthcare":                     "health",
    "about education, schools, colleges, university exams, or student affairs":             "education",
    "about crime, police, courts, arrests, or legal investigations":                        "crime",
    "about the environment, climate change, floods, droughts, or pollution":                "environment",
    "about international relations, foreign affairs, or events outside India":              "world",
    "about cryptocurrency, bitcoin, blockchain, or digital currency":                       "crypto",
}


def classify_article(
    title: str,
    description: str,
    full_text: str = "",
) -> str:
    """
    Return the best-matching category for an article using zero-shot NLI.

    Args:
      title:       Article headline (most informative signal)
      description: RSS description / summary
      full_text:   Article body — first 500 chars used for extra context

    Returns:
      Category string: cricket | politics | business | entertainment |
                       technology | sports | health | education | crime |
                       environment | world | crypto | general
    """
    # Build the classification text.
    # Title carries the strongest signal; description and start of full_text
    # add context.  We use up to ~1500 chars which stays well within mDeBERTa's
    # 512-token limit for English/Hindi (~4 chars/token on average).
    body_prefix = (full_text or "").strip()[:500]
    text = f"{title}. {description} {body_prefix}".strip()

    if not text:
        return "general"

    try:
        clf = _get_pipeline()
        result = clf(
            text[:1500],
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
