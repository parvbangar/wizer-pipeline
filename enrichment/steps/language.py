"""
enrichment/steps/language.py
════════════════════════════
Detects the actual language of an article's body text.

WHY DO WE NEED THIS WHEN feeds ALREADY HAVE language_code?
  Feed-level language metadata is set once when the feed is registered.
  In practice:
    - An 'en' feed occasionally publishes a Hindi article
    - A 'hi' feed sometimes publishes Hinglish (mixed Hindi/English)
    - Many feeds have language_code = NULL (missing metadata)
    - Wire services (PTI, ANI) republish in multiple languages

  language_detected gives you the ACTUAL language of each individual article.
  This is critical for:
    - Routing articles to the right language bucket in the app
    - Sentiment analysis (VADER only works on English)
    - Future: routing to the right regional NLP model

LIBRARY: langdetect
  - Ported from Google's language-detection library
  - Supports 55 languages including Hindi (hi), Tamil (ta), Telugu (te),
    Malayalam (ml), Bengali (bn), Gujarati (gu), Marathi (mr), Punjabi (pa)
  - Fast — pure Python, no model downloads
  - Limitation: can confuse similar scripts (Urdu/Hindi, Malay/Indonesian)

INSTALL:  pip install langdetect
"""

from __future__ import annotations

import logging

from enrichment.config import LANG_DETECT_MIN_CHARS

log = logging.getLogger(__name__)


def detect_language(full_text: str, description: str, title: str) -> str | None:
    """
    Detect the language of article content.

    Tries text sources in order of reliability:
      1. full_text  — most text = most reliable detection
      2. description — fallback for RSS-only articles
      3. title      — last resort (titles are short, less reliable)

    Args:
      full_text:   Article body (may be empty)
      description: RSS description (may be empty)
      title:       Article headline (may be empty)

    Returns:
      ISO 639-1 language code string, e.g. 'en', 'hi', 'ta', 'te'
      None if detection fails or text is too short to be reliable
    """
    # Pick the richest available text source
    text = (
        (full_text or "").strip()
        or (description or "").strip()
        or (title or "").strip()
    )

    if len(text) < LANG_DETECT_MIN_CHARS:
        log.debug("Text too short for language detection (%d chars)", len(text))
        return None

    try:
        from langdetect import detect
        lang = detect(text[:2000])   # cap at 2000 chars — detection plateaus after that
        return lang
    except ImportError:
        log.warning("langdetect not installed — skipping language detection")
        return None
    except Exception as e:
        log.debug("Language detection failed: %s", e)
        return None
