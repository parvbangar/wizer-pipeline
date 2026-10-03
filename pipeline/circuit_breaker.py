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
  │  Dormant feeds are is_active=False with disabled_reason='dormant'.  │
  │  db.get_due_feeds() re-polls them once per                          │
  │  DORMANT_RECHECK_INTERVAL_DAYS (7).  If a re-poll inserts a new     │
  │  article the feed is re-activated automatically; otherwise it stays │
  │  dormant and is retried a week later.  (Error-disabled feeds carry  │
  │  disabled_reason='errors' and are NEVER auto re-polled.)            │
  │                                                                     │
  │  Reset: UPDATE feeds SET is_active=True, fail_count=0,              │
  │         disabled_reason=NULL WHERE id='...';                        │
  └─────────────────────────────────────────────────────────────────────┘

STATE FLOW:
  active  →  (5 errors)      →  is_active=False, disabled_reason='errors'  [ERROR_DISABLED]
  active  →  (30 days empty) →  is_active=False, disabled_reason='dormant' [DORMANT]
  dormant →  (weekly re-poll finds an article) →  active   [automatic]
  is_active=False  →  (manual reset)  →  active
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from pipeline.config import (
    MAX_ERRORS_BEFORE_DISABLE,
    DORMANCY_DAYS,
    DORMANT_RECHECK_INTERVAL_DAYS,
    FEED_COL_CREATED, FEED_COL_LAST_NEW, FEED_COL_DISABLED_REASON,
    DISABLED_DORMANT,
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
        feed_created_str = feed.get(FEED_COL_CREATED)
        if not feed_created_str:
            return False, ""

        try:
            feed_created = _parse_ts(feed_created_str)
            if feed_created is None:
                return False, ""
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


def _parse_ts(value) -> datetime | None:
    """Parse an ISO timestamp from PostgREST; naive values are treated as UTC."""
    if not value or not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def is_dormant_feed(feed: dict) -> bool:
    """True for a feed parked by the dormancy circuit (NOT error-disabled)."""
    return (
        not feed.get("is_active", True)
        and feed.get(FEED_COL_DISABLED_REASON) == DISABLED_DORMANT
    )


def should_skip_feed(feed: dict) -> tuple[bool, str]:
    """
    Decide whether to skip this feed in the current poll cycle.

    A feed is skipped when:
      - is_active=False (disabled by error streak or dormancy) …
      - … EXCEPT a dormant feed that db.get_due_feeds() selected for its
        weekly re-check (it tags those with feed["_dormant_recheck"]=True).
        Error-disabled feeds are never re-checked.

    Returns:
      (should_skip, reason)
    """
    if not feed.get("is_active", True):
        if feed.get("_dormant_recheck") and is_dormant_feed(feed):
            return False, ""
        return True, "Feed is marked inactive (is_active=False)"

    return False, ""


def dormant_recheck_due(feed: dict, now: datetime | None = None) -> bool:
    """
    True if a dormant feed has gone DORMANT_RECHECK_INTERVAL_DAYS since its
    last poll and should be re-polled once.
    """
    now = now or datetime.now(timezone.utc)
    last = _parse_ts(feed.get("last_polled_at"))
    if last is None:
        return True
    return (now - last) >= timedelta(days=DORMANT_RECHECK_INTERVAL_DAYS)


def get_last_new_article_date(feed: dict) -> datetime | None:
    """
    Extract the date when this feed last produced a NEW article.

    Reads feeds.last_new_article_at, which db.update_feed_after_poll only
    advances when a poll inserts >0 articles.  (We deliberately do NOT use
    last_success_at: it advances on every successful fetch, even an empty one,
    so a feed that has been silent for months would never look stale.)

    Returns None if the feed has never produced an article, or if the column
    is absent (see has_new_article_tracking()).
    """
    return _parse_ts(feed.get(FEED_COL_LAST_NEW))


def has_new_article_tracking(feed: dict) -> bool:
    """
    True if the feed row carries the last_new_article_at column at all.
    When the column hasn't been migrated, dormancy detection must be skipped:
    there is no trustworthy "last new article" signal to judge by.
    """
    return FEED_COL_LAST_NEW in feed


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
