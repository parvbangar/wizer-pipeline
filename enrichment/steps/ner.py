"""
enrichment/steps/ner.py
════════════════════════
Named Entity Recognition — extracts people, organisations, and locations
from article text.

WHY NER IS THE BACKBONE OF LAYER 2:
  Almost every downstream step depends on entities:
  - Story clustering:    two articles about "Modi" + "Budget" = same story
  - Propensity scoring:  "Virat Kohli" → sports → high Indian engagement
  - App layer:           article tags, entity-based search, related articles
  - GNews enrichment:    query API with entity names, not raw titles

ENTITY TYPES STORED (mapped to our schema):
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

MODEL DISPATCH (three models based on language):
  English (en, en-*):
    spaCy en_core_web_sm — best accuracy on English news headlines

  Indian regional (hi, bn, gu, kn, ml, mr, or, pa, ta, te, as, ur):
    ai4bharat/IndicNER — trained on Naamapadam, a 11-language Indian NER
    dataset. Correctly recognises entities written in Devanagari, Tamil,
    Telugu, Bengali scripts where spaCy's multilingual model struggles.

  Other languages (or IndicNER unavailable):
    spaCy xx_ent_wiki_sm — multilingual fallback trained on Wikipedia

WHY IndicNER OVER xx_ent_wiki_sm FOR INDIAN LANGUAGES:
  xx_ent_wiki_sm extracts entities from Wikipedia-style prose.
  Indian news uses very different sentence structure and named entities.
  IndicNER is specifically trained on Indian news wire data → much higher
  recall on politician names, party names, city names in native script.

INSTALL:
  pip install spacy transformers
  python -m spacy download en_core_web_sm
  python -m spacy download xx_ent_wiki_sm
"""

from __future__ import annotations

import logging

from enrichment.config import (
    SPACY_MODEL, SPACY_MULTILINGUAL_MODEL, INDIC_NER_MODEL,
    NER_MIN_SALIENCE, ENTITY_TYPE_MAP,
)

log = logging.getLogger(__name__)

# ── Model caches — loaded once per process ────────────────────────────────────
_nlp_en    = None   # spaCy English model
_nlp_xx    = None   # spaCy multilingual fallback
_nlp_indic = None   # HuggingFace IndicNER pipeline (False = tried and failed)

# Languages that use the English spaCy model
_ENGLISH_LANGS = frozenset({"en", "en-us", "en-gb", "en-in", "en-au"})

# Indian languages that should use IndicNER (supported by Naamapadam training set)
_INDIC_LANGS = frozenset({
    "hi",   # Hindi
    "bn",   # Bengali
    "gu",   # Gujarati
    "kn",   # Kannada
    "ml",   # Malayalam
    "mr",   # Marathi
    "or",   # Odia
    "pa",   # Punjabi
    "ta",   # Tamil
    "te",   # Telugu
    "as",   # Assamese
    "ur",   # Urdu
})

# IndicNER entity_group → our schema type
_INDIC_TYPE_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "GPE",
    # MISC omitted — too noisy for news entity tagging
}


# ── Model loaders ─────────────────────────────────────────────────────────────

def _get_nlp_en():
    global _nlp_en
    if _nlp_en is None:
        import spacy
        try:
            _nlp_en = spacy.load(SPACY_MODEL)
            log.info("spaCy English model '%s' loaded", SPACY_MODEL)
        except OSError:
            raise OSError(
                f"spaCy model '{SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {SPACY_MODEL}"
            )
    return _nlp_en


def _get_nlp_xx():
    global _nlp_xx
    if _nlp_xx is None:
        import spacy
        try:
            _nlp_xx = spacy.load(SPACY_MULTILINGUAL_MODEL)
            log.info("spaCy multilingual model '%s' loaded", SPACY_MULTILINGUAL_MODEL)
        except OSError:
            log.warning(
                "spaCy multilingual model '%s' not found — falling back to '%s'",
                SPACY_MULTILINGUAL_MODEL, SPACY_MODEL,
            )
            _nlp_xx = _get_nlp_en()
    return _nlp_xx


def _get_indic_pipeline():
    """
    Load (and cache) the ai4bharat/IndicNER HuggingFace pipeline.
    Returns None if the model is unavailable (falls back to xx_ent_wiki_sm).
    """
    global _nlp_indic
    if _nlp_indic is None:
        try:
            from transformers import pipeline as hf_pipeline
            log.info("Loading IndicNER model '%s'...", INDIC_NER_MODEL)
            _nlp_indic = hf_pipeline(
                "ner",
                model=INDIC_NER_MODEL,
                aggregation_strategy="simple",
            )
            log.info("IndicNER model loaded")
        except Exception as e:
            log.warning(
                "IndicNER unavailable: %s — Indian language NER falls back to xx_ent_wiki_sm",
                e,
            )
            _nlp_indic = False   # sentinel: tried and failed, don't retry
    return _nlp_indic if _nlp_indic is not False else None


# ── Salience computation ──────────────────────────────────────────────────────

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

    if entity_lower in title.lower():
        score += 0.5

    if entity_lower in text_body[:200].lower():
        score += 0.3

    score += min(all_mentions * 0.1, 0.3)

    return min(round(score, 2), 1.0)


# ── IndicNER extraction path ──────────────────────────────────────────────────

def _extract_entities_indic(
    title: str,
    combined: str,
    body: str,
) -> list[dict]:
    """
    Extract entities from Indian-language text using ai4bharat/IndicNER.

    Falls back to an empty list on any error — the spaCy path handles recovery.
    """
    pipe = _get_indic_pipeline()
    if pipe is None:
        return []

    try:
        raw = pipe(combined[:512])
    except Exception as e:
        log.debug("IndicNER pipeline failed: %s", e)
        return []

    # Count mentions for salience (simple substring count in body)
    title_lower = title.lower()
    results: list[dict] = []
    seen: set[str] = set()

    for ent in raw:
        entity_type = _INDIC_TYPE_MAP.get(ent.get("entity_group", ""), "")
        if not entity_type:
            continue

        entity_text = (ent.get("word") or "").strip()
        if len(entity_text) < 2:
            continue

        key = entity_text.lower()
        if key in seen:
            continue
        seen.add(key)

        # Count occurrences in combined text for salience
        mention_count = combined.lower().count(key)
        salience = _compute_salience(entity_text, title_lower, body, mention_count)
        if salience < NER_MIN_SALIENCE:
            continue

        results.append({
            "entity_text": entity_text,
            "entity_type": entity_type,
            "salience":    salience,
        })

    results.sort(key=lambda x: x["salience"], reverse=True)
    return results[:30]


# ── spaCy extraction path (English + fallback) ────────────────────────────────

def _extract_entities_spacy(
    title: str,
    combined: str,
    body: str,
    nlp,
) -> list[dict]:
    """Extract entities using a spaCy model."""
    try:
        doc = nlp(combined)
    except Exception as e:
        log.debug("spaCy processing failed: %s", e)
        return []

    mention_counts: dict[str, int] = {}
    raw_entities: dict[str, tuple[str, str]] = {}

    for ent in doc.ents:
        mapped_type = ENTITY_TYPE_MAP.get(ent.label_)
        if mapped_type is None:
            continue

        entity_text = ent.text.strip()
        if len(entity_text) < 2:
            continue

        key = entity_text.lower()
        mention_counts[key] = mention_counts.get(key, 0) + 1
        if key not in raw_entities:
            raw_entities[key] = (entity_text, mapped_type)

    title_lower = title.lower()
    results: list[dict] = []

    for key, (entity_text, entity_type) in raw_entities.items():
        salience = _compute_salience(
            entity_text, title_lower, body, mention_counts[key],
        )
        if salience < NER_MIN_SALIENCE:
            continue
        results.append({
            "entity_text": entity_text,
            "entity_type": entity_type,
            "salience":    salience,
        })

    results.sort(key=lambda x: x["salience"], reverse=True)
    return results[:30]


# ── Public API ────────────────────────────────────────────────────────────────

def extract_entities(
    title: str,
    full_text: str,
    description: str,
    language_detected: str | None = None,
) -> list[dict]:
    """
    Extract named entities from article content.

    Dispatches to the best model for the detected language:
      English (en/en-*)  → spaCy en_core_web_sm
      Indian languages   → ai4bharat/IndicNER (fallback: xx_ent_wiki_sm)
      Other languages    → spaCy xx_ent_wiki_sm

    Processes: title + first 5,000 chars of body (covers lede + key paragraphs).

    Args:
      title:             Article headline
      full_text:         Article body text
      description:       RSS description (fallback if full_text is empty)
      language_detected: ISO language code from language detection step

    Returns:
      List of entity dicts sorted by salience (descending), capped at 30:
        [{"entity_text": "Narendra Modi", "entity_type": "PERSON", "salience": 0.8}, ...]

      Returns empty list on error or no entities above threshold.
    """
    body     = (full_text or "").strip() or (description or "").strip()
    combined = f"{title}\n\n{body[:5000]}" if title else body[:5000]

    if not combined.strip():
        return []

    lang       = (language_detected or "").lower()
    use_english = lang in _ENGLISH_LANGS or lang == ""
    use_indic   = lang in _INDIC_LANGS

    # ── English path ──────────────────────────────────────────────────────────
    if use_english:
        try:
            nlp = _get_nlp_en()
        except OSError as e:
            log.warning("NER skipped (English model): %s", e)
            return []
        return _extract_entities_spacy(title or "", combined, body, nlp)

    # ── Indian language path ──────────────────────────────────────────────────
    if use_indic:
        results = _extract_entities_indic(title or "", combined, body)
        if results:
            return results
        # IndicNER unavailable or returned nothing — fall through to spaCy xx
        try:
            nlp = _get_nlp_xx()
        except OSError as e:
            log.warning("NER skipped (multilingual fallback): %s", e)
            return []
        return _extract_entities_spacy(title or "", combined, body, nlp)

    # ── Other language path (spaCy multilingual) ──────────────────────────────
    try:
        nlp = _get_nlp_xx()
    except OSError as e:
        log.warning("NER skipped (multilingual model): %s", e)
        return []
    return _extract_entities_spacy(title or "", combined, body, nlp)
