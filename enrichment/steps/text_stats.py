"""
enrichment/steps/text_stats.py
═══════════════════════════════
Computes basic text statistics from article body.

WHAT IT PRODUCES:
  word_count        — number of words in full_text (or description as fallback)
  reading_time_mins — estimated reading time in minutes

WHY THESE MATTER FOR PROPENSITY SCORING:
  - Very short articles (<50 words) are often wire blurbs or aggregator stubs.
    They're lower-signal for virality because there's nothing to engage with.
  - Very long articles (>2000 words) are deep investigations.
    They get shared differently than short news items.
  - Reading time is a direct UX feature — "5 min read" drives clicks.

NO EXTERNAL LIBRARIES — only Python stdlib.
"""

from __future__ import annotations

from enrichment.config import WORDS_PER_MINUTE


def compute_text_stats(full_text: str, description: str) -> dict:
    """
    Compute word count and reading time from article content.

    Uses full_text if available (preferred — complete article body).
    Falls back to description if full_text is empty (RSS-only articles).

    Args:
      full_text:   Article body text (may be empty string)
      description: RSS description/summary (fallback)

    Returns dict with:
      word_count        (int)   — 0 if no text available
      reading_time_mins (float) — 0.0 if no text available
    """
    # Prefer full_text, fall back to description
    text = (full_text or "").strip() or (description or "").strip()

    if not text:
        return {"word_count": 0, "reading_time_mins": 0.0}

    # Simple whitespace-based word count.
    # Not perfect for CJK languages but good enough for news corpus
    # which is primarily English/Hindi (both use whitespace-delimited words).
    words = len(text.split())

    # Round to 1 decimal place: 412 words / 200 wpm = 2.1 min
    reading_time = round(words / WORDS_PER_MINUTE, 1)

    return {
        "word_count":        words,
        "reading_time_mins": reading_time,
    }
