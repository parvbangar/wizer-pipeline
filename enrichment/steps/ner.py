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

from enrichment.config import SPACY_MODEL, NER_MIN_SALIENCE, ENTITY_TYPE_MAP

log = logging.getLogger(__name__)

# spaCy model is loaded once per process (expensive operation ~500ms)
# and reused for all articles in the batch.
_nlp = None


def _get_nlp():
    """Load and cache the spaCy model. Raises on first call if not installed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(SPACY_MODEL)
            log.info("spaCy model '%s' loaded", SPACY_MODEL)
        except ImportError:
            raise ImportError("spacy is not installed. Run: pip install spacy")
        except OSError:
            raise OSError(
                f"spaCy model '{SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {SPACY_MODEL}"
            )
    return _nlp


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


def extract_entities(title: str, full_text: str, description: str) -> list[dict]:
    """
    Extract named entities from article content.

    Processes: title + first 5000 chars of body (or description as fallback).
    Capping at 5000 chars keeps processing fast while covering the
    most entity-rich part of a news article (lede and first few paragraphs).

    Args:
      title:       Article headline
      full_text:   Article body text
      description: RSS description (fallback if full_text is empty)

    Returns:
      List of entity dicts:
        [{"entity_text": "Narendra Modi", "entity_type": "PERSON", "salience": 0.8}, ...]

      Returns empty list on error or if no entities found above threshold.
    """
    try:
        nlp = _get_nlp()
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
