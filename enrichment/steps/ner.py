"""
enrichment/steps/ner.py
════════════════════════
Named Entity Recognition — extracts people, organisations, and locations
from article text using spaCy.

WHY NER IS THE BACKBONE OF LAYER 2:
  Almost every downstream step depends on entities:
  - Story clustering:    two articles about "Modi" + "Budget" = same story
  - Propensity scoring:  "Virat Kohli" → sports → high Indian engagement
  - App layer:           article tags, entity-based search, related articles
  - GNews enrichment:    query API with entity names, not raw titles

ENTITY TYPES STORED (mapped to our schema from spaCy labels):
  PERSON   — Narendra Modi, Virat Kohli, Mukesh Ambani
  ORG      — BJP, BCCI, Reliance, RBI, Supreme Court
  GPE      — India, Maharashtra, Delhi, Mumbai (geo-political entities)
  EVENT    — Budget 2024, IPL 2024, G20 Summit
  PRODUCT  — iPhone, WhatsApp, UPI, Aadhaar
  LAW      — CAA, RTI, Article 370, PMLA

SALIENCE SCORING:
  Salience (0.0–1.0) measures how central an entity is to the article.

  Algorithm:
    - Entity in title:              +0.5 base
    - Entity in first 200 chars:    +0.3 base
    - Each mention in body:         +0.1 (capped at 0.3 bonus)
    - Final score clamped to [0, 1]

  Entities below NER_MIN_SALIENCE (0.1) are discarded.

MODEL PERFORMANCE NOTES:
  en_core_web_sm is trained on English web/news text.
  It handles most Indian English news well.
  Known limitations:
    - May miss lesser-known regional politicians
    - May tag common Hindi words as entities if mixed into English text
    - Indian city names (Bengaluru, Thiruvananthapuram) sometimes missed

  Upgrade path: en_core_web_md gives ~5% better recall at 3x the model size.

INSTALL:
  pip install spacy
  python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging

from enrichment.config import SPACY_MODEL, SPACY_MULTILINGUAL_MODEL, NER_MIN_SALIENCE, ENTITY_TYPE_MAP

log = logging.getLogger(__name__)

# Models are loaded once per process (~500ms each) and reused across the batch.
# Two models:
#   _nlp_en  — en_core_web_sm  — English articles (higher accuracy on English)
#   _nlp_xx  — xx_ent_wiki_sm  — All Indian regional languages (hi, ta, te, bn,
#                                 mr, gu, kn, ml, pa, ur, etc.)
#
# WHY TWO MODELS?
#   en_core_web_sm is trained on English news — best accuracy for English.
#   xx_ent_wiki_sm is spaCy's multilingual model trained on Wikipedia across
#   50+ languages, giving reasonable NER for Hindi, Bengali, and other Indian
#   languages that appear in Wikipedia.
#   Running en_core_web_sm on Devanagari/Tamil script still catches English
#   entities embedded in the text (Modi, BJP, IPL), but xx_ent_wiki_sm
#   additionally recognises entities written in the native script.

_nlp_en = None   # English model
_nlp_xx = None   # Multilingual model


# Languages that use the English model
_ENGLISH_LANGS = frozenset({"en", "en-us", "en-gb", "en-in", "en-au"})


def _get_nlp(language: str | None):
    """
    Return the appropriate spaCy model for the given language code.

    - English (en, en-*) → en_core_web_sm
    - Everything else    → xx_ent_wiki_sm (multilingual)

    If the multilingual model is not installed, falls back to the English
    model so NER still runs (it will still catch English-named entities
    embedded in regional language text).
    """
    global _nlp_en, _nlp_xx
    import spacy

    lang = (language or "").lower()
    use_english = lang in _ENGLISH_LANGS or lang == ""

    if use_english:
        if _nlp_en is None:
            try:
                _nlp_en = spacy.load(SPACY_MODEL)
                log.info("spaCy English model '%s' loaded", SPACY_MODEL)
            except OSError:
                raise OSError(
                    f"spaCy model '{SPACY_MODEL}' not found. "
                    f"Run: python -m spacy download {SPACY_MODEL}"
                )
        return _nlp_en
    else:
        if _nlp_xx is None:
            try:
                _nlp_xx = spacy.load(SPACY_MULTILINGUAL_MODEL)
                log.info("spaCy multilingual model '%s' loaded", SPACY_MULTILINGUAL_MODEL)
            except OSError:
                # Multilingual model not installed — fall back to English model
                # (still catches English entity names embedded in regional text)
                log.warning(
                    "spaCy multilingual model '%s' not found — falling back to '%s'. "
                    "Run: python -m spacy download %s",
                    SPACY_MULTILINGUAL_MODEL, SPACY_MODEL, SPACY_MULTILINGUAL_MODEL,
                )
                if _nlp_en is None:
                    _nlp_en = spacy.load(SPACY_MODEL)
                _nlp_xx = _nlp_en   # share the same model instance
        return _nlp_xx


def _compute_salience(
    entity_text: str,
    title: str,
    text_body: str,
    all_mentions: int,
) -> float:
    """
    Compute a salience score for one entity (0.0 to 1.0).

    Higher salience = entity is more central to the article.
    Used to filter low-salience incidental mentions.
    """
    score = 0.0
    entity_lower = entity_text.lower()

    # In title: strongest signal
    if entity_lower in title.lower():
        score += 0.5

    # In the first 200 chars of body (lead paragraph)
    if entity_lower in text_body[:200].lower():
        score += 0.3

    # Frequency bonus: each mention adds 0.1, capped at 0.3
    score += min(all_mentions * 0.1, 0.3)

    return min(round(score, 2), 1.0)


def extract_entities(
    title: str,
    full_text: str,
    description: str,
    language_detected: str | None = None,
) -> list[dict]:
    """
    Extract named entities from article content.

    Processes: title + first 5000 chars of body (or description as fallback).
    Capping at 5000 chars keeps processing fast while covering the
    most entity-rich part of a news article (lede and first few paragraphs).

    Args:
      title:             Article headline
      full_text:         Article body text
      description:       RSS description (fallback if full_text is empty)
      language_detected: ISO language code from the language detection step.
                         Used to select the right spaCy model:
                           English (en/en-*) → en_core_web_sm
                           Indian regional   → xx_ent_wiki_sm (multilingual)

    Returns:
      List of entity dicts:
        [{"entity_text": "Narendra Modi", "entity_type": "PERSON", "salience": 0.8}, ...]

      Returns empty list on error or if no entities found above threshold.
    """
    try:
        nlp = _get_nlp(language_detected)
    except (ImportError, OSError) as e:
        log.warning("NER skipped: %s", e)
        return []

    # Build the text to process: title first, then body
    body = (full_text or "").strip() or (description or "").strip()
    combined = f"{title}\n\n{body[:5000]}" if title else body[:5000]

    if not combined.strip():
        return []

    try:
        doc = nlp(combined)
    except Exception as e:
        log.debug("spaCy processing failed: %s", e)
        return []

    # Count entity mentions across the full text for salience calculation
    mention_counts: dict[str, int] = {}
    raw_entities: dict[str, str] = {}   # text → type

    for ent in doc.ents:
        mapped_type = ENTITY_TYPE_MAP.get(ent.label_)
        if mapped_type is None:
            continue   # skip DATE, TIME, MONEY, CARDINAL etc.

        # Normalise entity text: strip whitespace, title-case for persons
        entity_text = ent.text.strip()
        if len(entity_text) < 2:
            continue   # skip single characters

        key = entity_text.lower()
        mention_counts[key] = mention_counts.get(key, 0) + 1
        # Keep first-seen capitalisation (title/lead paragraph version)
        if key not in raw_entities:
            raw_entities[key] = (entity_text, mapped_type)

    # Build final entity list with salience scores
    results: list[dict] = []
    title_lower = title.lower() if title else ""

    for key, (entity_text, entity_type) in raw_entities.items():
        salience = _compute_salience(
            entity_text,
            title_lower,
            body,
            mention_counts[key],
        )
        if salience < NER_MIN_SALIENCE:
            continue
        results.append({
            "entity_text": entity_text,
            "entity_type": entity_type,
            "salience":    salience,
        })

    # Sort by salience descending — most central entities first
    results.sort(key=lambda x: x["salience"], reverse=True)

    # Cap at 30 entities per article to keep article_entities table lean
    return results[:30]
