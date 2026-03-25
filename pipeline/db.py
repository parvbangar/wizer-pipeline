"""
db.py
═════
All database operations for the pipeline.

WHY IS ALL DB CODE IN ONE FILE?
  Centralising database calls means:
  - If a column name changes in Supabase, you fix it in ONE place
  - You can test database logic independently of the rest of the pipeline
  - Easier to spot N+1 query problems (querying in a loop = slow)

THE SUPABASE CLIENT
  supabase-py is the official Python library for Supabase.
  It wraps the PostgREST API (REST interface to your PostgreSQL database).
  All calls go over HTTPS to your Supabase project.

IMPORTANT: This module uses YOUR exact column names from your schema.
  feeds columns used:    id, feed_url, final_url, domain, publisher_name,
                         validation_tier, language_code, country_code,
                         iab_tier1, iab_tier2, has_paywall, is_active,
                         poll_interval_mins, priority_score, fail_count,
                         last_polled_at, last_success_at, articles_found
  articles columns used: all columns from your schema
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pipeline.config import (
    SUPABASE_URL, SUPABASE_KEY,
    TABLE_FEEDS, TABLE_ARTICLES, TABLE_RUNS,
    FEED_COL_ID, FEED_COL_URL, FEED_COL_FINAL_URL, FEED_COL_DOMAIN,
    FEED_COL_PUBLISHER, FEED_COL_CADENCE, FEED_COL_LANGUAGE, FEED_COL_COUNTRY,
    FEED_COL_IAB1, FEED_COL_IAB2, FEED_COL_HAS_PAYWALL, FEED_COL_IS_ACTIVE,
    FEED_COL_POLL_INTERVAL, FEED_COL_PRIORITY, FEED_COL_FAIL_COUNT,
    FEED_COL_LAST_POLLED, FEED_COL_LAST_SUCCESS, FEED_COL_ARTICLES_FOUND,
    ART_COL_URL_HASH, ART_COL_FEED_ID, ART_COL_CRAWLED,
    CADENCE_POLL_INTERVALS,
)

log = logging.getLogger(__name__)

# The client is created once and reused (singleton pattern)
_client = None


def get_client():
    """
    Return the Supabase client, creating it on first call.

    WHY LAZY INITIALISATION?
      If we created the client at import time, any test that imports db.py
      would immediately try to connect to Supabase — even offline tests.
      Lazy init means the connection only happens when actually needed.
    """
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "\n\nMissing Supabase credentials!\n"
                "Create a .env file with:\n"
                "  SUPABASE_URL=https://yourproject.supabase.co\n"
                "  SUPABASE_SERVICE_KEY=eyJh...\n"
                "Find these in Supabase Dashboard → Project Settings → API\n"
            )
        try:
            from supabase import create_client
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
            log.info("Connected to Supabase: %s", SUPABASE_URL)
        except ImportError:
            raise RuntimeError(
                "supabase package not installed.\n"
                "Run: pip install supabase\n"
            )
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# FEEDS: READING
# ─────────────────────────────────────────────────────────────────────────────

def get_due_feeds(cadence: str | None = None) -> list[dict]:
    """
    Return feeds that are due to be polled right now.

    A feed is "due" when:
      - is_active = True (not manually paused or dormant)
      - last_polled_at is NULL (never polled) OR
        (now - last_polled_at) >= poll_interval_mins

    If poll_interval_mins is NULL in the DB, we use the cadence default
    from CADENCE_POLL_INTERVALS.

    Args:
      cadence: If given, only return feeds with this update_cadence value.
               If None, return feeds from ALL cadences.
    """
    client = get_client()

    # Build the SELECT columns list — exactly what we need, nothing more
    select_cols = ", ".join([
        FEED_COL_ID,
        FEED_COL_URL,
        FEED_COL_FINAL_URL,
        FEED_COL_DOMAIN,
        FEED_COL_PUBLISHER,
        FEED_COL_CADENCE,
        FEED_COL_LANGUAGE,
        FEED_COL_COUNTRY,
        FEED_COL_IAB1,
        FEED_COL_IAB2,
        FEED_COL_HAS_PAYWALL,
        FEED_COL_POLL_INTERVAL,
        FEED_COL_PRIORITY,
        FEED_COL_FAIL_COUNT,
        FEED_COL_LAST_POLLED,
        FEED_COL_LAST_SUCCESS,
        FEED_COL_ARTICLES_FOUND,
    ])

    try:
        query = (
            client.table(TABLE_FEEDS)
            .select(select_cols)
            .eq(FEED_COL_IS_ACTIVE, True)
            .order(FEED_COL_LAST_POLLED, desc=False, nullsfirst=True)  # oldest polled first → natural rotation
            .limit(10000)
        )
        if cadence:
            query = query.eq(FEED_COL_CADENCE, cadence)

        resp = query.execute()
        all_feeds = resp.data or []

        # Filter in Python: feeds where enough time has passed since last poll
        now = datetime.now(timezone.utc)
        due = []
        for feed in all_feeds:
            if _is_feed_due(feed, now):
                due.append(feed)

        log.info(
            "Found %d feeds due for polling (cadence=%s, total_active=%d)",
            len(due), cadence or "all", len(all_feeds),
        )
        return due

    except Exception as e:
        log.error("get_due_feeds failed: %s", e)
        return []


def _is_feed_due(feed: dict, now: datetime) -> bool:
    """
    Check if a feed is due for polling based on time elapsed since last poll.

    Logic:
      - Never polled (last_polled_at is None) → always due
      - Time since last poll >= interval → due
      - Otherwise → not due yet
    """
    last_polled_str = feed.get(FEED_COL_LAST_POLLED)
    if not last_polled_str:
        return True   # Never polled → poll now

    try:
        last_polled = datetime.fromisoformat(
            last_polled_str.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError):
        return True   # Can't parse timestamp → poll to be safe

    # Get the interval: prefer DB value, fall back to cadence default
    db_interval = feed.get(FEED_COL_POLL_INTERVAL)  # poll_interval_mins from DB
    cadence = feed.get(FEED_COL_CADENCE, "unknown")

    if db_interval and isinstance(db_interval, (int, float)) and db_interval > 0:
        interval_minutes = int(db_interval)
    else:
        interval_minutes = CADENCE_POLL_INTERVALS.get(cadence, 720)

    elapsed_minutes = (now - last_polled).total_seconds() / 60
    return elapsed_minutes >= interval_minutes


# ─────────────────────────────────────────────────────────────────────────────
# FEEDS: WRITING (circuit breaker state updates)
# ─────────────────────────────────────────────────────────────────────────────

def update_feed_after_poll(
    feed_id: str,
    success: bool,
    new_articles: int,
    error_msg: str = "",
) -> None:
    """
    Update the feed row after a poll attempt.

    On SUCCESS:
      - last_polled_at = now
      - last_success_at = now
      - fail_count = 0  (reset the error counter)
      - articles_found += new_articles

    On FAILURE:
      - last_polled_at = now
      - fail_count += 1
      - If fail_count reaches MAX_ERRORS: is_active = False (circuit open)

    Args:
      feed_id:      The feed's UUID
      success:      True if the RSS fetch succeeded (even if 0 new articles)
      new_articles: Count of new articles inserted this poll
      error_msg:    Error description (for logging, not stored in DB)
    """
    from pipeline.config import MAX_ERRORS_BEFORE_DISABLE
    client = get_client()
    now = datetime.now(timezone.utc).isoformat()

    try:
        if success:
            patch = {
                FEED_COL_LAST_POLLED:    now,
                FEED_COL_LAST_SUCCESS:   now,
                FEED_COL_FAIL_COUNT:     0,   # reset on success
            }
            if new_articles > 0:
                # Increment articles_found using PostgreSQL arithmetic
                # We read the current value and add to it
                current = _get_feed_articles_found(feed_id)
                patch[FEED_COL_ARTICLES_FOUND] = current + new_articles

        else:
            # Failed poll — read current fail_count, increment it
            current_fails = _get_feed_fail_count(feed_id)
            new_fail_count = current_fails + 1

            patch = {
                FEED_COL_LAST_POLLED: now,
                FEED_COL_FAIL_COUNT:  new_fail_count,
            }

            if new_fail_count >= MAX_ERRORS_BEFORE_DISABLE:
                patch[FEED_COL_IS_ACTIVE] = False
                log.warning(
                    "CIRCUIT OPEN: feed %s disabled after %d consecutive errors. "
                    "Last error: %s",
                    feed_id, new_fail_count, error_msg,
                )

        client.table(TABLE_FEEDS).update(patch).eq(FEED_COL_ID, feed_id).execute()

    except Exception as e:
        # Don't crash the pipeline if state update fails — just log
        log.error("update_feed_after_poll(%s) failed: %s", feed_id, e)


def mark_feed_dormant(feed_id: str, reason: str) -> None:
    """
    Mark a feed as dormant (is_active=False) due to prolonged inactivity.

    Called by the circuit breaker when a feed hasn't produced new articles
    in DORMANCY_DAYS days.  The feed can be re-activated manually in Supabase:
      UPDATE feeds SET is_active=True, fail_count=0 WHERE id='...';
    """
    client = get_client()
    try:
        client.table(TABLE_FEEDS).update({
            FEED_COL_IS_ACTIVE: False,
        }).eq(FEED_COL_ID, feed_id).execute()
        log.warning("DORMANT: feed %s marked inactive. Reason: %s", feed_id, reason)
    except Exception as e:
        log.error("mark_feed_dormant(%s) failed: %s", feed_id, e)


def _get_feed_fail_count(feed_id: str) -> int:
    """Read the current fail_count for a feed."""
    try:
        resp = (
            get_client()
            .table(TABLE_FEEDS)
            .select(FEED_COL_FAIL_COUNT)
            .eq(FEED_COL_ID, feed_id)
            .single()
            .execute()
        )
        return int(resp.data.get(FEED_COL_FAIL_COUNT) or 0)
    except Exception:
        return 0


def _get_feed_articles_found(feed_id: str) -> int:
    """Read the current articles_found count for a feed."""
    try:
        resp = (
            get_client()
            .table(TABLE_FEEDS)
            .select(FEED_COL_ARTICLES_FOUND)
            .eq(FEED_COL_ID, feed_id)
            .single()
            .execute()
        )
        return int(resp.data.get(FEED_COL_ARTICLES_FOUND) or 0)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# ARTICLES: DEDUPLICATION QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def load_recent_hashes(feed_id: str, limit: int = 2000) -> set[int]:
    """
    Load recent url_hash values for this feed into a Python set.

    WHY DO THIS?
      Before inserting any article, we check if it's a duplicate.
      We could check the database for every single article URL, but that
      means hundreds of individual database queries per feed poll.

      Instead, we load the last 2000 hashes into memory once at the start
      of each poll.  Then all checks happen in Python (instant).
      Only truly new articles need a final database check.

    WHY 2000?
      A very active feed might produce 200 articles/day.
      2000 hashes covers the last 10 days of articles.
      This catches almost all recent duplicates without using much memory
      (2000 × 8 bytes per int64 = 16KB — tiny).

    Returns:
      A Python set of integers (url_hash values).
      Set lookup is O(1) — checking "is X in this set?" is instant.
    """
    try:
        resp = (
            get_client()
            .table(TABLE_ARTICLES)
            .select(ART_COL_URL_HASH)
            .eq(ART_COL_FEED_ID, feed_id)
            .order(ART_COL_CRAWLED, desc=True)
            .limit(limit)
            .execute()
        )
        hashes = {row[ART_COL_URL_HASH] for row in (resp.data or [])}
        log.debug("Loaded %d hashes for feed %s", len(hashes), feed_id)
        return hashes
    except Exception as e:
        log.warning("load_recent_hashes(%s) failed: %s — using empty set", feed_id, e)
        return set()


def load_recent_simhashes(limit: int = 10000) -> set[int]:
    """
    Load recent title_simhash values across ALL feeds for near-dedup checking.

    WHY ACROSS ALL FEEDS?
      Near-duplicate detection (same story, different source) needs to check
      across feeds.  If NDTV published a PTI story 2 hours ago, and now
      India Today publishes the same PTI story, we want to flag it.

    WHY 10000?
      News cycles change fast.  10000 recent titles covers roughly 24-48 hours
      of content across all your feeds.  Older stories shouldn't be compared.
    """
    try:
        resp = (
            get_client()
            .table(TABLE_ARTICLES)
            .select("title_simhash")
            .not_.is_("title_simhash", "null")
            .order(ART_COL_CRAWLED, desc=True)
            .limit(limit)
            .execute()
        )
        return {row["title_simhash"] for row in (resp.data or [])}
    except Exception as e:
        log.warning("load_recent_simhashes failed: %s — using empty set", e)
        return set()


def url_hash_exists(url_hash: int) -> bool:
    """
    Check the database if an article with this url_hash already exists.

    This is the AUTHORITATIVE check (layer 2 fallback after the in-memory check).
    It queries the UNIQUE index on articles.url_hash which makes it very fast.

    Returns True if the article already exists, False if it's new.
    Fails open (returns False) on database errors — the UNIQUE constraint
    will catch any actual duplicates at insert time.
    """
    try:
        resp = (
            get_client()
            .table(TABLE_ARTICLES)
            .select(ART_COL_URL_HASH, count="exact")
            .eq(ART_COL_URL_HASH, url_hash)
            .limit(1)
            .execute()
        )
        return (resp.count or 0) > 0
    except Exception as e:
        log.warning("url_hash_exists check failed: %s — failing open", e)
        return False   # fail open: the UNIQUE index is the last line of defence


# ─────────────────────────────────────────────────────────────────────────────
# ARTICLES: WRITING
# ─────────────────────────────────────────────────────────────────────────────

def upsert_articles(rows: list[dict]) -> tuple[int, int]:
    """
    Insert a batch of articles into the database.

    Uses UPSERT (insert or ignore) — if url_hash already exists,
    the row is silently skipped.  This is the final dedup safety net.

    WHY BATCH INSERT?
      Inserting rows one-by-one means one network round-trip per row.
      Batch inserting 20 rows at once uses one round-trip.
      At 20ms per round-trip, batch insert is ~20× faster.

    Args:
      rows: List of dicts, each matching the articles table schema exactly.

    Returns:
      (inserted_count, duplicate_count) tuple
    """
    if not rows:
        return 0, 0

    client = get_client()
    inserted = 0
    duplicates = 0

    try:
        resp = (
            client.table(TABLE_ARTICLES)
            .upsert(rows, on_conflict=ART_COL_URL_HASH, ignore_duplicates=True)
            .execute()
        )
        inserted = len(resp.data) if resp.data else 0
        duplicates = len(rows) - inserted

    except Exception as e:
        log.error("Batch upsert failed: %s — retrying row by row", e)
        # Fall back to individual inserts so partial batches aren't lost
        for row in rows:
            try:
                client.table(TABLE_ARTICLES).upsert(
                    row,
                    on_conflict=ART_COL_URL_HASH,
                    ignore_duplicates=True,
                ).execute()
                inserted += 1
            except Exception as row_e:
                log.error(
                    "Row insert failed url=%s: %s",
                    row.get("url", "?"), row_e,
                )
                duplicates += 1

    return inserted, duplicates


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUN LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def log_run(summary: dict) -> None:
    """
    Write a summary row to pipeline_runs after each run.
    This gives you a history of every pipeline execution for monitoring.

    If the insert fails (e.g. missing column), logs an ERROR with the full
    detail so it is visible in GitHub Actions — previously this was a silent
    warning which made DB schema mismatches invisible.
    """
    try:
        get_client().table(TABLE_RUNS).insert(summary).execute()
        log.debug("pipeline_runs row written: cadence=%s", summary.get("cadence"))
    except Exception as e:
        log.error(
            "log_run FAILED — run not recorded in pipeline_runs. "
            "Check that the 'cadence' column exists (run migration Step 7). "
            "Error: %s | summary keys: %s",
            e, list(summary.keys()),
        )
