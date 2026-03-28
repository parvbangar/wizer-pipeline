"""
enrichment/steps/ner.py
════════════════════════
Named Entity Recognition — extracts people, organisations, and locations
from article text using spaCy.

MODEL DISPATCH:
  English (en, en-*):  spaCy en_core_web_sm
  All other languages: spaCy xx_ent_wiki_sm (multilingual, Wikipedia-trained)

ENTITY TYPES STORED:
  PERSON   — Narendra Modi, Virat Kohli, Mukesh Ambani
  ORG      — BJP, BCCI, Reliance, RBI, Supreme Court
  GPE      — India, Maharashtra, Delhi, Mumbai
  EVENT    — Budget 2024, IPL 2024, G20 Summit
  PRODUCT  — iPhone, WhatsApp, UPI, Aadhaar
  LAW      — CAA, RTI, Article 370, PMLA

SALIENCE SCORING:
  - Entity in title:           +0.5
  - Entity in first 200 chars: +0.3
  - Each body mention:         +0.1 (capped at 0.3)
  - Final score clamped to [0, 1], entities below NER_MIN_SALIENCE filtered out

INSTALL:
  pip install spacy
  python -m spacy download en_core_web_sm
  python -m spacy download xx_ent_wiki_sm
"""

from __future__ import annotations

import logging

from enrichment.config import (
    SPACY_MODEL, SPACY_MULTILINGUAL_MODEL,
    NER_MIN_SALIENCE, ENTITY_TYPE_MAP,
)

log = logging.getLogger(__name__)

# ── Model caches — loaded once per process ────────────────────────────────────
_nlp_en = None
_nlp_xx = None

_ENGLISH_LANGS = frozenset({"en", "en-us", "en-gb", "en-in", "en-au"})


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
                "Multilingual model '%s' not found — falling back to '%s'",
                SPACY_MULTILINGUAL_MODEL, SPACY_MODEL,
            )
            _nlp_xx = _get_nlp_en()
    return _nlp_xx


def _compute_salience(entity_text: str, title: str, text_body: str, mentions: int) -> float:
    score = 0.0
    el = entity_text.lower()
    if el in title.lower():
        score += 0.5
    if el in text_body[:200].lower():
        score += 0.3
    score += min(mentions * 0.1, 0.3)
    return min(round(score, 2), 1.0)


def _extract_entities_spacy(title: str, combined: str, body: str, nlp) -> list[dict]:
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

    results: list[dict] = []
    for key, (entity_text, entity_type) in raw_entities.items():
        salience = _compute_salience(entity_text, title, body, mention_counts[key])
        if salience < NER_MIN_SALIENCE:
            continue
        results.append({
            "entity_text": entity_text,
            "entity_type": entity_type,
            "salience":    salience,
        })

    results.sort(key=lambda x: x["salience"], reverse=True)
    return results[:30]


def extract_entities(
    title: str,
    full_text: str,
    description: str,
    language_detected: str | None = None,
) -> list[dict]:
    """
    Extract named entities from article content.

    English → spaCy en_core_web_sm
    All other languages → spaCy xx_ent_wiki_sm

    Returns list of entity dicts sorted by salience, max 30.
    Returns empty list on error.
    """
    body     = (full_text or "").strip() or (description or "").strip()
    combined = f"{title}\n\n{body[:5000]}" if title else body[:5000]

    if not combined.strip():
        return []

    lang        = (language_detected or "").lower()
    use_english = lang in _ENGLISH_LANGS or lang == ""

    if use_english:
        try:
            nlp = _get_nlp_en()
        except OSError as e:
            log.warning("NER skipped (English model): %s", e)
            return []
        return _extract_entities_spacy(title or "", combined, body, nlp)

    try:
        nlp = _get_nlp_xx()
    except OSError as e:
        log.warning("NER skipped (multilingual model): %s", e)
        return []
    return _extract_entities_spacy(title or "", combined, body, nlp)
