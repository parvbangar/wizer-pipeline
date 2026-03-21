"""
circuit_breaker.py
══════════════════
Monitors feed health and disables bad/inactive feeds automatically.

WHAT IS A CIRCUIT BREAKER?
  The name comes from electrical engineering.  In your home, if too much
  current flows through a wire, the circuit breaker "opens" (trips) to
  protect the system.  It needs a manual reset before it works again.

  In software, a circuit breaker stops repeated attempts to do something
  that keeps failing — like fetching a dead RSS feed.  Without it, the
  pipeline would waste time and resources on feeds that never work.

THE TWO CIRCUITS WE MONITOR:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ CIRCUIT 1: ERROR STREAK                                             │
  │                                                                     │
  │  feed.fail_count tracks consecutive fetch/parse failures.           │
  │  fail_count=0 → all good                                            │
  │  fail_count=3 → 3 failures in a row (warning territory)             │
  │  fail_count=5 → CIRCUIT OPENS → is_active=False                     │
  │                                                                     │
  │  Reset: UPDATE feeds SET is_active=True, fail_count=0               │
  │         WHERE id='...';  (you do this manually in Supabase)         │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │ CIRCUIT 2: DORMANCY                                                 │
  │                                                                     │
  │  If a feed hasn't produced ANY new articles in 30 days, it's dead.  │
  │  Common causes: site shut down, URL changed, behind paywall now.    │
  │                                                                     │
  │  We still poll dormant feeds once/week (they might come back).      │
  │  After 30 more days of nothing → is_active=False permanently.       │
  │                                                                     │
  │  Reset: Same as above — manually re-enable in Supabase.             │
  └─────────────────────────────────────────────────────────────────────┘

STATE FLOW:
  active  →  (5 errors)      →  is_active=False  [ERROR_DISABLED]
  active  →  (30 days empty) →  is_active=False  [DORMANT]
  is_active=False  →  (manual reset)  →  active
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from pipeline.config import (
    MAX_ERRORS_BEFORE_DISABLE,
    DORMANCY_DAYS,
    DORMANT_RECHECK_INTERVAL_DAYS,
)

log = logging.getLogger(__name__)


def check_dormancy(
    feed: dict,
    last_new_article_at: datetime | None,
) -> tuple[bool, str]:
    """
    Check if a feed should be marked dormant due to prolonged inactivity.

    Args:
      feed:                The feed dict from the database
      last_new_article_at: When the most recent NEW article from this feed
                           was found (None if never found any)

    Returns:
      (should_mark_dormant, reason_string)
      If should_mark_dormant is True, caller should call db.mark_feed_dormant()
    """
    if last_new_article_at is None:
        # Never found any articles — give it a grace period
        feed_created_str = feed.get("created_at")
        if not feed_created_str:
            return False, ""

        try:
            feed_created = datetime.fromisoformat(
                feed_created_str.replace("Z", "+00:00")
            )
            age_days = (datetime.now(timezone.utc) - feed_created).days
            if age_days >= DORMANCY_DAYS:
                return True, f"No articles found in {age_days} days since creation"
        except (ValueError, AttributeError):
            pass

        return False, ""

    now = datetime.now(timezone.utc)
    days_stale = (now - last_new_article_at).days

    if days_stale >= DORMANCY_DAYS:
        return (
            True,
            f"No new articles for {days_stale} days (threshold: {DORMANCY_DAYS})"
        )

    return False, ""


def should_skip_feed(feed: dict) -> tuple[bool, str]:
    """
    Decide whether to skip this feed in the current poll cycle.

    A feed is skipped when:
      - is_active=False (disabled by error streak or dormancy)

    Note: We already filter is_active=True in db.get_due_feeds().
    This function is an extra guard for edge cases.

    Returns:
      (should_skip, reason)
    """
    if not feed.get("is_active", True):
        return True, "Feed is marked inactive (is_active=False)"

    return False, ""


def get_last_new_article_date(feed: dict) -> datetime | None:
    """
    Extract the date when this feed last produced a new article.

    We use last_success_at as a proxy — this is set whenever a poll
    finds new articles.  It's the closest column in your schema to
    "when did we last get fresh content from this feed".

    Returns None if the feed has never successfully provided articles.
    """
    last_success_str = feed.get("last_success_at")
    if not last_success_str:
        return None
    try:
        return datetime.fromisoformat(last_success_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def log_circuit_state(feed_id: str, feed_url: str, fail_count: int) -> None:
    """
    Log a warning when a feed's fail_count is getting high.
    Helps you catch problems before the circuit fully opens.
    """
    threshold = MAX_ERRORS_BEFORE_DISABLE
    if fail_count == 0:
        return
    elif fail_count < threshold:
        log.warning(
            "Feed degraded: %s (fail_count=%d/%d)",
            feed_url, fail_count, threshold,
        )
    else:
        log.error(
            "CIRCUIT OPEN: feed %s disabled (fail_count=%d). "
            "Fix the feed and run: UPDATE feeds SET is_active=True, "
            "fail_count=0 WHERE id='%s';",
            feed_url, fail_count, feed_id,
        )
