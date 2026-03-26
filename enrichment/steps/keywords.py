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
  - Layer 3 propensity scoring: keyword overlap with user interest graph
  - GNews queries: search API with top keywords instead of full title

STORED AS: jsonb array in articles.keywords
  e.g. ["Narendra Modi", "Budget 2024", "GST reform", "direct tax"]

INSTALL: pip install yake
"""

from __future__ import annotations

import logging

from enrichment.config import MAX_KEYWORDS, KEYWORD_MAX_NGRAM, KEYWORD_DEDUP_THRESHOLD

log = logging.getLogger(__name__)


def extract_keywords(title: str, full_text: str, description: str) -> list[str]:
    """
    Extract top keywords from article content.

    Combines title + body for extraction. Title words naturally score
    higher due to position weighting, which is what we want.

    Args:
      title:       Article headline
      full_text:   Article body text
      description: RSS description (fallback)

    Returns:
      List of keyword strings, e.g. ["Narendra Modi", "budget 2024"]
      Returns empty list on error or if text is too short.
    """
    # Build combined text — title at top for position weighting
    body = (full_text or "").strip() or (description or "").strip()
    text = f"{title}\n\n{body[:3000]}" if title else body[:3000]

    if len(text.strip()) < 30:
        return []

    try:
        import yake
        extractor = yake.KeywordExtractor(
            lan="en",                           # language hint (YAKE adapts regardless)
            n=KEYWORD_MAX_NGRAM,                # max ngram size
            dedupLim=KEYWORD_DEDUP_THRESHOLD,   # deduplication threshold
            top=MAX_KEYWORDS,                   # how many to return
            features=None,
        )
        # Returns list of (keyword, score) tuples — lower score = more important
        keywords_with_scores = extractor.extract_keywords(text)
        # Return just the keyword strings (not the scores)
        return [kw for kw, _ in keywords_with_scores]

    except ImportError:
        log.warning("yake not installed — skipping keyword extraction. Run: pip install yake")
        return []
    except Exception as e:
        log.debug("Keyword extraction failed: %s", e)
        return []
