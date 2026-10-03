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

from enrichment.config import DEBERTA_MODEL, CLASSIFY_CONFIDENCE_THRESHOLD, CLASSIFY_TAG_THRESHOLD

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
            )
            log.info("mDeBERTa model loaded")
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers torch"
            )
    return _pipeline


# ── Category labels ────────────────────────────────────────────────────────────
#
# MEASURED, NOT GUESSED — tools/classifier_eval/ scores designs against 160
# hand-labelled live Indian headlines (English + Hindi):
#
#   design                                              accuracy  "general"
#   long "about X, Y, or Z" phrases, softmax, t=0.20      33.8 %     55 %
#   short single-concept labels, independent, t=0.30      64.4 %      7 %
#   + cricket⊂sports rule, headline+description input     68.1 %     11 %   ← this
#
# Why the old design failed: NLI models lean on word overlap between premise
# and hypothesis. The old "world" label ended "…or events outside India", so
# nearly EVERY Indian headline overlapped it — "world" won almost everything,
# "crypto" came second, and the 0.20 cut-off then sent most articles to
# "general" (84 % of articles in an end-to-end run).
#
# The fix:
#   - short labels naming ONE concept ("crime", "law and courts"), several of
#     which may map to the same category — easier entailment targets than a
#     comma-separated list of four ideas;
#   - an explicit news template ("This news article is about {}.");
#   - multi_label=True: every label gets its own entailment-vs-contradiction
#     probability, instead of a softmax that forces 12 labels to share 1.0
#     (with softmax, ANY single label rarely clears a fixed threshold).
HYPOTHESIS_TEMPLATE = "This news article is about {}."
CLASSIFY_MULTI_LABEL = True

_LABEL_TO_CATEGORY: dict[str, str] = {
    "cricket":                "cricket",
    "politics":               "politics",
    "business":               "business",
    "economy":                "business",
    "stock market":           "business",
    "entertainment":          "entertainment",
    "movies":                 "entertainment",
    "technology":             "technology",
    "sports":                 "sports",
    "health":                 "health",
    "education":              "education",
    "crime":                  "crime",
    "law and courts":         "crime",
    "environment":            "environment",
    "weather":                "environment",
    "international news":     "world",
    "war":                    "world",
    "cryptocurrency":         "crypto",
}
_CANDIDATE_LABELS = list(_LABEL_TO_CATEGORY)


# Cricket is a sub-type of sports. Scored independently, "sports" almost always
# edges out "cricket" on a cricket story (e.g. sports 0.96 / cricket 0.54 for an
# Irani Cup report), so the more specific label wins whenever it is itself
# confidently entailed.
CRICKET_SUBTYPE_MIN = 0.5


def pick_category(scores: dict[str, float]) -> str:
    """
    Decide the category from per-label entailment scores (pure, unit-tested).

      best label below CLASSIFY_CONFIDENCE_THRESHOLD → "general"
      best category "sports" and cricket ≥ CRICKET_SUBTYPE_MIN → "cricket"
    """
    if not scores:
        return "general"
    best_label = max(scores, key=scores.get)
    if scores[best_label] < CLASSIFY_CONFIDENCE_THRESHOLD:
        return "general"
    category = _LABEL_TO_CATEGORY.get(best_label, "general")
    if category == "sports" and scores.get("cricket", 0.0) >= CRICKET_SUBTYPE_MIN:
        return "cricket"
    return category


# Text the classifier sees: headline + description, or headline + the body lead
# when the description is missing / repeats the headline — the same builder the
# clustering embeddings use. Measured on production articles (2 CPU threads,
# category + tags): headline+description+500 body chars 15.4 s/article →
# headline+description 8.4 s, and 5.5 s with batched passes. The 66.2 % accuracy
# was measured on headline+description (68.1 % with this exact builder), so
# the faster input is also the validated one; NLI cost grows with every token of premise × every label.
CLASSIFY_MAX_CHARS = 600
# Hypothesis pairs scored per forward pass. 32 vs 1: 5.5 s vs 8.4 s per article.
CLASSIFY_BATCH_SIZE = 32


def _classification_text(title: str, description: str, full_text: str) -> str:
    """Headline + description (or body lead), ≤ CLASSIFY_MAX_CHARS; "" if nothing."""
    from enrichment.steps.embedding import build_embedding_text
    return build_embedding_text(title, description, full_text, max_chars=CLASSIFY_MAX_CHARS)


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
    text = _classification_text(title, description, full_text)

    if not text:
        return "general"

    try:
        clf = _get_pipeline()
        result = clf(
            text,
            candidate_labels=_CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=CLASSIFY_MULTI_LABEL,
            batch_size=CLASSIFY_BATCH_SIZE,
        )

        category = pick_category(dict(zip(result["labels"], result["scores"])))
        log.debug("classify: '%s...' → %s", (title or "")[:50], category)
        return category

    except Exception as e:
        log.warning("Classification failed (%s) — returning 'general'", e)
        return "general"


# ── AI topic tags — finer-grained multi-label tags ─────────────────────────────
#
# These complement the single primary category with up to 5 specific topic tags.
# Examples:
#   Category "politics" → ai_tag ["government", "monetary policy", "economic policy"]
#   Category "technology" → ai_tag ["artificial intelligence", "startup"]
#
# Labels are written as NLI hypothesis completions (same template as above).
# multi_label=True → independent sigmoid per label (scores are NOT softmax-normalized).
# Tags with score ≥ CLASSIFY_TAG_THRESHOLD are kept (default 0.25).

_TAG_CANDIDATE_LABELS = [
    "about government policy or legislation",
    "about elections or political campaigns",
    "about financial markets or stock exchange",
    "about corporate earnings or company results",
    "about monetary policy or central bank interest rates",
    "about economic policy or fiscal measures",
    "about a cricket match or cricket tournament",
    "about a sports competition or athletic event",
    "about Bollywood films or the entertainment industry",
    "about a public health crisis or disease outbreak",
    "about medical treatment or healthcare system",
    "about education reform or academic examinations",
    "about a crime investigation or criminal trial",
    "about technology product launch or digital innovation",
    "about artificial intelligence or machine learning",
    "about an environmental disaster or climate event",
    "about international diplomacy or foreign policy",
    "about military conflict or armed forces",
    "about startup funding or entrepreneurship",
    "about space research or scientific discovery",
]

_TAG_LABEL_TO_TAG: dict[str, str] = {
    "about government policy or legislation":               "government",
    "about elections or political campaigns":               "elections",
    "about financial markets or stock exchange":            "financial markets",
    "about corporate earnings or company results":          "corporate",
    "about monetary policy or central bank interest rates": "monetary policy",
    "about economic policy or fiscal measures":             "economic policy",
    "about a cricket match or cricket tournament":          "cricket",
    "about a sports competition or athletic event":         "sports",
    "about Bollywood films or the entertainment industry":  "entertainment",
    "about a public health crisis or disease outbreak":     "public health",
    "about medical treatment or healthcare system":         "healthcare",
    "about education reform or academic examinations":      "education",
    "about a crime investigation or criminal trial":        "crime",
    "about technology product launch or digital innovation":"technology",
    "about artificial intelligence or machine learning":    "artificial intelligence",
    "about an environmental disaster or climate event":     "environment",
    "about international diplomacy or foreign policy":      "foreign policy",
    "about military conflict or armed forces":              "conflict",
    "about startup funding or entrepreneurship":            "startup",
    "about space research or scientific discovery":         "science",
}


def classify_tags(
    title: str,
    description: str,
    full_text: str = "",
) -> list[str]:
    """
    Return up to 5 fine-grained topic tags for an article.

    Uses the same mDeBERTa pipeline as classify_article() (already in memory).
    Runs with multi_label=True so each tag is scored independently — an article
    can match multiple tags simultaneously.

    Args:
      title:       Article headline
      description: RSS/OG description
      full_text:   Article body — first 500 chars for context

    Returns:
      List of tag strings, e.g. ["government", "monetary policy", "economic policy"]
      Returns empty list on error or low-confidence articles.
    """
    text = _classification_text(title, description, full_text)

    if not text:
        return []

    try:
        clf = _get_pipeline()
        result = clf(
            text,
            candidate_labels=_TAG_CANDIDATE_LABELS,
            multi_label=True,   # independent sigmoid score per tag
            batch_size=CLASSIFY_BATCH_SIZE,
        )

        tags: list[str] = []
        for label, score in zip(result["labels"], result["scores"]):
            if score >= CLASSIFY_TAG_THRESHOLD:
                tag = _TAG_LABEL_TO_TAG.get(label)
                if tag:
                    tags.append(tag)
            if len(tags) >= 5:
                break

        return tags

    except Exception as e:
        log.warning("Tag classification failed (%s) — returning []", e)
        return []
