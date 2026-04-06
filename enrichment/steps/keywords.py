"""
enrichment/steps/keywords.py
═════════════════════════════
Extracts the top keywords and key phrases from article text using YAKE.

WHY YAKE (Yet Another Keyword Extractor)?
  - Completely unsupervised — no training data or model needed
  - Language-agnostic — works on Hindi, Tamil, Telugu articles too
    (unlike KeyBERT which needs a transformer model for each language)
  - Fast: 10ms per article on average
  - No downloads: pure Python

HOW YAKE WORKS (simplified):
  It scores each candidate keyword using:
    1. Position — keywords near the start of a document score higher
    2. Frequency — terms that appear often score higher
    3. Context — terms co-occurring with many different words score higher
    4. Relatedness — terms that are spread across the document score higher
  Lower YAKE score = more important keyword (counter-intuitive but that's the API)

WHAT KEYWORDS ARE USED FOR:
  - App layer: article tags shown to users
  - Search: full-text search augmentation

STORED AS: jsonb array in articles.keywords
  e.g. ["Narendra Modi", "Budget 2024", "GST reform", "direct tax"]

INSTALL: pip install yake
"""

from __future__ import annotations

import logging

from enrichment.config import MAX_KEYWORDS, KEYWORD_MAX_NGRAM, KEYWORD_DEDUP_THRESHOLD

log = logging.getLogger(__name__)

# YAKE language code map.
# YAKE has explicit stopword lists for these codes; others fall back to a
# generic (no-stopword) mode which still works but slightly less precisely.
_YAKE_LANG_MAP: dict[str, str] = {
    "en": "en", "hi": "hi", "ar": "ar", "de": "de", "es": "es",
    "fr": "fr", "it": "it", "nl": "nl", "pt": "pt", "ru": "ru",
    "tr": "tr", "zh": "zh",
}

# Keywords that are pure editorial boilerplate — they appear in contaminated
# full_text extractions (subscribe CTAs, share prompts, nav headers) and carry
# no news value.  Filtered after YAKE extraction.
_BOILERPLATE_KW: frozenset[str] = frozenset({
    "read more", "also read", "read also", "related news", "related stories",
    "subscribe", "newsletter", "sign up", "follow us", "download app",
    "click here", "tap here", "watch live", "share this", "share on",
    "tweet", "breaking news", "latest news", "top stories", "more stories",
    "advertisement", "sponsored", "partner content", "paid post",
    "terms of use", "privacy policy", "all rights reserved", "copyright",
    "stay updated", "get latest", "find out more", "learn more",
    "read full story", "read full article",
})


def extract_keywords(
    title: str,
    full_text: str,
    description: str,
    language_detected: str | None = None,
) -> list[str]:
    """
    Extract top keywords from article content.

    Combines title + body for extraction. Title words naturally score
    higher due to position weighting, which is what we want.

    Args:
      title:             Article headline
      full_text:         Article body text
      description:       RSS description (fallback)
      language_detected: ISO language code from the language step — used to
                         give YAKE the correct stopword list (e.g. "hi" for Hindi)

    Returns:
      List of keyword strings, e.g. ["Narendra Modi", "budget 2024"]
      Returns empty list on error or if text is too short.
    """
    # Build combined text — title at top for position weighting
    body = (full_text or "").strip() or (description or "").strip()
    text = f"{title}\n\n{body[:3000]}" if title else body[:3000]

    if len(text.strip()) < 30:
        return []

    # Map detected language to the closest YAKE code; fall back to "en"
    lang_base = (language_detected or "en").split("-")[0].lower()
    yake_lang = _YAKE_LANG_MAP.get(lang_base, "en")

    try:
        import yake
        extractor = yake.KeywordExtractor(
            lan=yake_lang,                      # language-specific stopwords
            n=KEYWORD_MAX_NGRAM,                # max ngram size
            dedupLim=KEYWORD_DEDUP_THRESHOLD,   # deduplication threshold
            top=MAX_KEYWORDS * 2,               # extract extra so we can filter boilerplate
            features=None,
        )
        # Returns list of (keyword, score) tuples — lower score = more important
        raw = extractor.extract_keywords(text)

        # Filter boilerplate and return top MAX_KEYWORDS
        result: list[str] = []
        for kw, _ in raw:
            kw_lower = kw.lower().strip()
            if kw_lower in _BOILERPLATE_KW:
                continue
            if any(bp in kw_lower for bp in _BOILERPLATE_KW):
                continue
            if len(kw_lower) < 3:
                continue
            result.append(kw)
            if len(result) >= MAX_KEYWORDS:
                break

        return result

    except ImportError:
        log.warning("yake not installed — skipping keyword extraction. Run: pip install yake")
        return []
    except Exception as e:
        log.debug("Keyword extraction failed: %s", e)
        return []
