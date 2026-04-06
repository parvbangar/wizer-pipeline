"""
enrichment/steps/summarizer.py
════════════════════════════════
Extractive article summarization — produces the ai_summary field.

STRATEGY:
  1. Use description if it is substantive (≥150 chars).
     RSS descriptions are often written by journalists as genuine editorial
     summaries — they are the highest-quality signal we have.

  2. Otherwise extract the first 3 sentences from full_text.
     News articles follow the inverted-pyramid structure: the most important
     facts (who / what / when / where / why) always appear in the opening
     sentences.  The first 3 sentences cover the entire lead for most articles.

NO MODEL REQUIRED.
  Fast, deterministic, works in any language.
  ~0ms per article (pure regex + string ops).

OUTPUT:
  A single string, max 500 chars.
  Stored in articles.ai_summary as plain text.
"""

from __future__ import annotations

import re


# Sentence boundary: punctuation followed by whitespace + uppercase letter,
# Hindi Devanagari letter (\\u0900-\\u097F), or opening quote.
# This handles English and Hindi without a tokenizer.
_SENT_SPLIT_RE = re.compile(
    r'(?<=[.!?।])\s+(?=[A-Z"\''\u0900-\u097F\u0600-\u06FF])'
)


def summarize_article(
    full_text: str,
    description: str,
    max_chars: int = 500,
) -> str | None:
    """
    Return a short extractive summary of the article.

    Args:
      full_text:   Crawled article body
      description: RSS/OG description
      max_chars:   Hard cap on summary length (default 500)

    Returns:
      Summary string, or None if no meaningful text is available.
    """
    desc = (description or "").strip()
    body = (full_text or "").strip()

    # Prefer a rich description (journalist-written) over body extraction
    if len(desc) >= 150:
        return desc[:max_chars]

    if not body:
        return desc[:max_chars] if desc else None

    # Split into sentences and take the first 3 that are substantial
    sentences = _SENT_SPLIT_RE.split(body)
    clean: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) >= 30:
            clean.append(s)
        if len(clean) >= 3:
            break

    if not clean:
        # Fallback: first max_chars of body
        return body[:max_chars]

    summary = " ".join(clean)
    return summary[:max_chars]
