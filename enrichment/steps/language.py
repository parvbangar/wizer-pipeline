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
    - Selecting the right spaCy NER model (en vs xx)
    - Sentiment analysis (VADER only works on English)
    - Displaying articles in the right language feed in the app

LIBRARY: lingua-language-detector
  Why lingua over langdetect:
    - DETERMINISTIC — same text always returns the same result (langdetect
      uses a random seed internally, so results vary between calls)
    - MORE ACCURATE — especially on short texts like news headlines
    - BUILT FOR ACCURACY over speed — uses confidence scoring internally
    - EXPLICIT INDIAN LANGUAGE SUPPORT — Hindi, Bengali, Gujarati, Marathi,
      Punjabi, Tamil, Telugu, Urdu are all first-class supported languages

  Install:  pip install lingua-language-detector

SUPPORTED INDIAN LANGUAGES:
  hi — Hindi         bn — Bengali       gu — Gujarati
  mr — Marathi       pa — Punjabi       ta — Tamil
  te — Telugu        ur — Urdu

  Note: Kannada (kn), Malayalam (ml), Odia (or), Assamese (as) are not
  supported by lingua. Articles in these languages return None.
"""

from __future__ import annotations

import logging

from enrichment.config import LANG_DETECT_MIN_CHARS

log = logging.getLogger(__name__)

# ── Lingua detector — loaded once, reused for entire batch ────────────────────
# Building the detector is expensive (~300ms). We cache it at module level.
_detector = None


def _get_detector():
    """
    Load and cache the lingua language detector.

    Uses from_all_languages() to maximise coverage across Indian regional
    languages and international languages. The detector is thread-safe and
    can be shared across calls.
    """
    global _detector
    if _detector is None:
        try:
            from lingua import LanguageDetectorBuilder
            _detector = LanguageDetectorBuilder.from_all_languages().build()
            log.info("Lingua language detector loaded")
        except ImportError:
            raise ImportError(
                "lingua-language-detector not installed. "
                "Run: pip install lingua-language-detector"
            )
    return _detector


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
      ISO 639-1 language code string, e.g. 'en', 'hi', 'ta', 'te', 'bn'
      None if detection fails or text is too short to be reliable.

      Results are DETERMINISTIC — the same text always returns the same code.
    """
    # Use the richest available text source for most reliable detection
    text = (
        (full_text or "").strip()
        or (description or "").strip()
        or (title or "").strip()
    )

    if len(text) < LANG_DETECT_MIN_CHARS:
        log.debug("Text too short for language detection (%d chars)", len(text))
        return None

    try:
        detector = _get_detector()
    except ImportError as e:
        log.warning("%s", e)
        return None

    try:
        # lingua returns None when it cannot confidently detect the language
        # (unlike langdetect which guesses and gets it wrong)
        lang = detector.detect_language_of(text[:3000])
        if lang is None:
            log.debug("Language detection returned no confident result")
            return None

        # Convert lingua Language enum to ISO 639-1 string
        # e.g. Language.HINDI → IsoCode639_1.HI → "hi"
        iso_code = lang.iso_code_639_1.name.lower()
        return iso_code

    except Exception as e:
        log.debug("Language detection failed: %s", e)
        return None
