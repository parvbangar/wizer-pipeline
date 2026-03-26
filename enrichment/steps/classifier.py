"""
enrichment/steps/classifier.py
════════════════════════════════
Classifies each article into an Indian-news-specific category.

WHY NOT USE THE FEED-LEVEL iab_tier1/iab_tier2?
  Feed-level categories are set once for the whole feed.
  A feed tagged "Sports > Cricket" will publish cricket articles,
  but it also publishes football, tennis, and badminton.
  Article-level classification gives you the actual topic of this article.

WHY RULE-BASED (not a classifier model)?
  - Zero setup: no model download, no GPU, no training data
  - Deterministic: you can read the rules and know exactly why an article
    was tagged 'politics'. An ML model is a black box.
  - Indian-specific: you control the keywords. No Western-trained model
    knows to look for "BCCI" for cricket or "NDA" for politics.
  - Fast: string matching is ~100x faster than inference
  - Upgradeable: the function signature stays the same if you later
    swap this for a zero-shot classifier (zero code changes in runner.py)

HOW IT WORKS:
  1. Build a search string from title + description (lower-cased)
  2. Scan each category's keyword list in priority order
  3. Return the first category that gets a keyword hit
  4. If no category matches → return 'general'

  The iab_tier1 from the feed is used as a tiebreaker hint when
  the text itself is ambiguous.

CATEGORY LIST (defined in enrichment/config.py):
  cricket, politics, business, entertainment, technology,
  sports, health, education, crime, environment, world, general
"""

from __future__ import annotations

from enrichment.config import CATEGORY_RULES, CATEGORY_DEFAULT


def classify_article(
    title: str,
    description: str,
    iab_tier1: str,
) -> str:
    """
    Return the best-matching category for an article.

    Args:
      title:       Article headline
      description: RSS description / summary
      iab_tier1:   Feed-level IAB category (used as tiebreaker hint)

    Returns:
      Category string from CATEGORY_RULES keys, or CATEGORY_DEFAULT ('general').
    """
    # Build search corpus: title is weighted by appearing twice
    # (simple way to give headlines more influence without changing the logic)
    search_text = f"{title} {title} {description}".lower()

    # Track hit counts per category to handle ties
    hit_counts: dict[str, int] = {}

    for category, keywords in CATEGORY_RULES.items():
        hits = sum(1 for kw in keywords if kw in search_text)
        if hits > 0:
            hit_counts[category] = hits

    if not hit_counts:
        return CATEGORY_DEFAULT

    # If there's a clear winner, return it
    if len(hit_counts) == 1:
        return next(iter(hit_counts))

    max_hits = max(hit_counts.values())
    top_categories = [c for c, h in hit_counts.items() if h == max_hits]

    if len(top_categories) == 1:
        return top_categories[0]

    # Tiebreaker: use feed-level IAB tier1 as a hint
    iab_lower = (iab_tier1 or "").lower()
    for cat in top_categories:
        if cat in iab_lower or iab_lower in cat:
            return cat

    # Final fallback: return the category listed first in CATEGORY_RULES
    # (ordered by importance — cricket and politics first)
    for cat in CATEGORY_RULES:
        if cat in top_categories:
            return cat

    return CATEGORY_DEFAULT
