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
  OPTIONAL feeds columns (docs/ingestion_fixes_migration.sql; code degrades
  gracefully if absent): created_at, last_new_article_at, disabled_reason
  articles columns used: all columns from your schema
"""

from __future__ import annotations

import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Callable, Iterable

from pipeline.config import (
    SUPABASE_URL, SUPABASE_KEY,
    TABLE_FEEDS, TABLE_ARTICLES, TABLE_RUNS,
    FEED_COL_ID, FEED_COL_URL, FEED_COL_FINAL_URL, FEED_COL_DOMAIN,
    FEED_COL_PUBLISHER, FEED_COL_CADENCE, FEED_COL_LANGUAGE, FEED_COL_COUNTRY,
    FEED_COL_IAB1, FEED_COL_IAB2, FEED_COL_HAS_PAYWALL, FEED_COL_IS_ACTIVE,
    FEED_COL_POLL_INTERVAL, FEED_COL_PRIORITY, FEED_COL_FAIL_COUNT,
    FEED_COL_LAST_POLLED, FEED_COL_LAST_SUCCESS, FEED_COL_ARTICLES_FOUND,
    FEED_COL_LAST_NEW, FEED_COL_DISABLED_REASON,
    FEED_OPTIONAL_COLS, DISABLED_DORMANT, DISABLED_ERRORS,
    FEED_DUE_TOLERANCE, DB_PAGE_SIZE,
    DORMANT_RECHECK_INTERVAL_DAYS,
    ART_COL_URL_HASH, ART_COL_FEED_ID, ART_COL_CRAWLED,
    CADENCE_POLL_INTERVALS,
    ARTICLE_HARD_LIMIT, ARTICLE_PRUNE_TARGET,
    MAX_ARTICLE_BODY_CHARS, FEED_POLL_BATCH, MAX_ERRORS_BEFORE_DISABLE,
)

log = logging.getLogger(__name__)

# The client is created once and reused (singleton pattern)
_client = None          # kept for backwards compatibility (tests); see _local
_local = threading.local()


def get_client():
    """
    Return this THREAD's Supabase client, creating it on first use.

    WHY ONE CLIENT PER THREAD?
      Feeds and articles are processed by up to ~30 executor threads. With a
      single shared client they all multiplexed one HTTP connection, and the
      API gateway answered bursts with "Server disconnected" — on 2026-10-03
      that turned every insert of the first production run into an error.
      A client per thread gives each thread its own connection pool.

    WHY LAZY INITIALISATION?
      If we created the client at import time, any test that imports db.py
      would immediately try to connect to Supabase — even offline tests.
    """
    client = getattr(_local, "client", None)
    if client is None:
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
        except ImportError:
            raise RuntimeError("supabase package not installed. Run: pip install supabase") from None
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        _local.client = client
        log.debug("Connected to Supabase (thread %s)", threading.current_thread().name)
    return client


def _reset_client() -> None:
    """Drop this thread's client so the next call reconnects."""
    _local.client = None


def _is_transient(e: Exception) -> bool:
    """Connection-level failures worth one reconnect + retry."""
    try:
        import httpx
        if isinstance(e, (httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectTimeout,
                          httpx.ConnectError, httpx.PoolTimeout, httpx.ReadError, httpx.WriteError)):
            return True
    except ImportError:
        pass
    msg = str(e).lower()
    return any(t in msg for t in ("server disconnected", "timed out", "connection reset",
                                  "connection refused", "broken pipe"))


def _retry(operation):
    """Run a DB call; on a transient connection error reconnect and retry once."""
    try:
        return operation()
    except Exception as e:
        if not _is_transient(e):
            raise
        log.debug("DB connection error (%s) — reconnecting and retrying once", e)
        _reset_client()
        time.sleep(0.5)
        return operation()


# ─────────────────────────────────────────────────────────────────────────────
# FEEDS: READING
# ─────────────────────────────────────────────────────────────────────────────

class FeedLoadError(RuntimeError):
    """Raised when the list of feeds cannot be loaded from the database.

    get_due_feeds() used to swallow DB errors and return [], which made a
    Supabase outage indistinguishable from "no feeds due" - the run exited 0.
    Raising lets main.py exit 1 so the GitHub Actions job goes red.
    """


def _is_missing_column_error(exc: Exception, columns: Iterable[str]) -> bool:
    """True if a PostgREST/Postgres error says one of `columns` doesn't exist."""
    text = str(exc)
    looks_missing = (
        "42703" in text or "PGRST204" in text or "does not exist" in text
        or "Could not find" in text
    )
    return looks_missing and any(c in text for c in columns)


def _paginate(
    make_query: Callable[[], object],
    limit: int | None = None,
    page_size: int = DB_PAGE_SIZE,
) -> list[dict]:
    """
    Read ALL rows of a query in `page_size` pages using .range().

    WHY?  PostgREST silently caps every response at max_rows (1000 on
    Supabase), whatever .limit() says.  A single .limit(10000) therefore
    returned at most 1000 rows with no error.

    `make_query` must return a FRESH query builder each call (builders are
    mutated by .range()) and should include a deterministic ORDER BY so pages
    don't overlap or skip rows.
    `limit` stops early once that many rows have been collected.
    """
    rows: list[dict] = []
    offset = 0
    while limit is None or len(rows) < limit:
        want = page_size if limit is None else min(page_size, limit - len(rows))
        resp = make_query().range(offset, offset + want - 1).execute()
        page = resp.data or []
        rows.extend(page)
        if len(page) < want:
            break          # short page = no more rows
        offset += want
    return rows


_FEED_BASE_COLS = [
    FEED_COL_ID, FEED_COL_URL, FEED_COL_FINAL_URL, FEED_COL_DOMAIN,
    FEED_COL_PUBLISHER, FEED_COL_CADENCE, FEED_COL_LANGUAGE, FEED_COL_COUNTRY,
    FEED_COL_IAB1, FEED_COL_IAB2, FEED_COL_HAS_PAYWALL, FEED_COL_POLL_INTERVAL,
    FEED_COL_PRIORITY, FEED_COL_FAIL_COUNT, FEED_COL_LAST_POLLED,
    FEED_COL_LAST_SUCCESS, FEED_COL_ARTICLES_FOUND,
]


def _select_feed_rows(
    cadence: str | None,
    is_active: bool,
    extra_filter: Callable[[object], object] | None = None,
) -> list[dict]:
    """
    Paginated SELECT of feeds with the right column list.

    Tries the full column list (including the OPTIONAL columns created_at /
    last_new_article_at / disabled_reason).  If the database says one of them
    doesn't exist yet (migration not applied) it retries once without them,
    so a missing migration degrades dormancy handling instead of killing the run.
    """
    client = get_client()

    def build(cols: list[str]) -> Callable[[], object]:
        def make():
            q = (
                client.table(TABLE_FEEDS)
                .select(", ".join(cols))
                .eq(FEED_COL_IS_ACTIVE, is_active)
                # oldest polled first -> natural rotation; id breaks ties so
                # pagination is stable
                .order(FEED_COL_LAST_POLLED, desc=False, nullsfirst=True)
                .order(FEED_COL_ID)
            )
            if cadence:
                q = q.eq(FEED_COL_CADENCE, cadence)
            if extra_filter:
                q = extra_filter(q)
            return q
        return make

    try:
        return _paginate(build(_FEED_BASE_COLS + list(FEED_OPTIONAL_COLS)))
    except Exception as e:
        if not _is_missing_column_error(e, FEED_OPTIONAL_COLS):
            raise
        log.warning(
            "feeds table lacks optional columns (%s) - run "
            "docs/ingestion_fixes_migration.sql. Dormancy tracking disabled.", e,
        )
        return _paginate(build(_FEED_BASE_COLS))


def _derive_language_name(feed: dict) -> None:
    """
    Fill feed["language_name"] (-> articles.language) with the feed's short
    language code ("en", "hi", "pt-br").

    PRODUCTION SCHEMA: articles.language is char(5) there (docs/migration.sql
    says text). Writing names ("English", "Malayalam") made EVERY insert fail
    with 22001 "value too long" on the first run of 2026-10-03, so the column
    gets the code, truncated to 5 characters. The full name is derivable from
    config.LANGUAGE_NAMES whenever it is needed.
    """
    if feed.get("language_name"):
        feed["language_name"] = str(feed["language_name"])[:5]
        return
    code = str(feed.get(FEED_COL_LANGUAGE) or "").lower().replace("_", "-")
    feed["language_name"] = code[:5]


def get_due_feeds(cadence: str | None = None) -> list[dict]:
    """
    Return feeds that are due to be polled right now.

    A feed is "due" when:
      - is_active = True and (last_polled_at is NULL OR
        (now - last_polled_at) >= poll_interval_mins * FEED_DUE_TOLERANCE)
      - OR it is DORMANT (is_active=False, disabled_reason='dormant') and
        DORMANT_RECHECK_INTERVAL_DAYS have passed since its last poll.
        Those rows are tagged feed["_dormant_recheck"]=True.

    If poll_interval_mins is NULL in the DB, we use the cadence default
    from CADENCE_POLL_INTERVALS.

    Args:
      cadence: If given, only return feeds with this update_cadence value.
               If None, return feeds from ALL cadences.

    Raises:
      FeedLoadError: the active-feed query failed.  (A failure of only the
      dormant re-check query is logged and ignored - it must not block the
      normal poll.)
    """
    try:
        all_feeds = _select_feed_rows(cadence, is_active=True)
    except Exception as e:
        log.error("get_due_feeds failed: %s", e)
        raise FeedLoadError(f"could not load feeds from Supabase: {e}") from e

    now = datetime.now(timezone.utc)
    due = [f for f in all_feeds if _is_feed_due(f, now)]

    # -- Weekly re-check of dormant feeds ------------------------------------
    dormant_due: list[dict] = []
    try:
        dormant = _select_feed_rows(
            cadence, is_active=False,
            extra_filter=lambda q: q.eq(FEED_COL_DISABLED_REASON, DISABLED_DORMANT),
        )
        cutoff = timedelta(days=DORMANT_RECHECK_INTERVAL_DAYS)
        for f in dormant:
            last = _parse_iso(f.get(FEED_COL_LAST_POLLED))
            if last is None or (now - last) >= cutoff:
                f["_dormant_recheck"] = True
                dormant_due.append(f)
    except Exception as e:
        log.warning("dormant re-check query failed (%s) - skipping re-checks", e)

    due.extend(dormant_due)
    for f in due:
        _derive_language_name(f)

    log.info(
        "Found %d feeds due for polling (cadence=%s, total_active=%d, "
        "dormant_rechecks=%d)",
        len(due), cadence or "all", len(all_feeds), len(dormant_due),
    )
    return due


def _parse_iso(value) -> datetime | None:
    """Parse a PostgREST timestamp; naive values are treated as UTC."""
    if not value or not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _is_feed_due(feed: dict, now: datetime) -> bool:
    """
    Check if a feed is due for polling based on time elapsed since last poll.

    Logic:
      - Never polled (last_polled_at is None) -> always due
      - Time since last poll >= interval * FEED_DUE_TOLERANCE -> due
      - Otherwise -> not due yet

    WHY A TOLERANCE?
      Each workflow's cron period equals the feed's poll interval (e.g. hourly
      cron vs 60 min).  But last_polled_at is stamped when the feed FINISHES,
      some minutes after its run started, so at the next run's start the
      elapsed time is "interval minus a few minutes" - just under the
      interval.  With a strict `elapsed >= interval` the feed was skipped and
      only polled on the run after, i.e. every 2 intervals.  Treating a feed as
      due at 90 % of its interval (FEED_DUE_TOLERANCE, default 0.9) absorbs
      run duration and GitHub-cron jitter.  Lower it if runs routinely take
      more than ~10 % of the interval.
    """
    if not feed.get(FEED_COL_LAST_POLLED):
        return True   # Never polled -> poll now
    last_polled = _parse_iso(feed.get(FEED_COL_LAST_POLLED))
    if last_polled is None:
        return True   # Can't parse timestamp -> poll to be safe

    # Get the interval: prefer DB value, fall back to cadence default
    db_interval = feed.get(FEED_COL_POLL_INTERVAL)  # poll_interval_mins from DB
    cadence = feed.get(FEED_COL_CADENCE, "unknown")

    if db_interval and isinstance(db_interval, (int, float)) and db_interval > 0:
        interval_minutes = int(db_interval)
    else:
        interval_minutes = CADENCE_POLL_INTERVALS.get(cadence, 720)

    elapsed_minutes = (now - last_polled).total_seconds() / 60
    return elapsed_minutes >= interval_minutes * FEED_DUE_TOLERANCE


# ─────────────────────────────────────────────────────────────────────────────
# FEEDS: WRITING (circuit breaker state updates)
# ─────────────────────────────────────────────────────────────────────────────

def _update_feed_row(feed_id: str, patch: dict) -> None:
    """
    UPDATE one feeds row, tolerating missing OPTIONAL columns.

    If the DB rejects the patch because last_new_article_at / disabled_reason
    don't exist yet, drop those keys and retry once so the essential state
    (last_polled_at, fail_count, is_active ...) is still saved.
    """
    # (retried once on transient connection errors — see _retry)
    try:
        _retry(lambda: get_client().table(TABLE_FEEDS).update(patch).eq(FEED_COL_ID, feed_id).execute())
    except Exception as e:
        optional_in_patch = [c for c in FEED_OPTIONAL_COLS if c in patch]
        if optional_in_patch and _is_missing_column_error(e, optional_in_patch):
            log.warning(
                "feeds update without optional columns %s (run "
                "docs/ingestion_fixes_migration.sql): %s", optional_in_patch, e,
            )
            reduced = {k: v for k, v in patch.items() if k not in FEED_OPTIONAL_COLS}
            _retry(lambda: get_client().table(TABLE_FEEDS).update(reduced).eq(FEED_COL_ID, feed_id).execute())
        else:
            raise


def update_feed_after_poll(
    feed_id: str,
    success: bool,
    new_articles: int,
    error_msg: str = "",
    reactivate: bool = False,
) -> None:
    """
    Update the feed row after a poll attempt.

    On SUCCESS:
      - last_polled_at = now
      - last_success_at = now   (= "the fetch worked", even with 0 new articles)
      - last_new_article_at = now  ONLY if new_articles > 0
        (this is the timestamp dormancy detection reads)
      - fail_count = 0  (reset the error counter)
      - articles_found += new_articles
      - reactivate=True (a dormant re-check that found articles) also sets
        is_active=True and clears disabled_reason

    On FAILURE:
      - last_polled_at = now   (so a hung/failing feed doesn't jump the queue
        again on the very next run)
      - fail_count += 1
      - If fail_count reaches MAX_ERRORS: is_active = False,
        disabled_reason = 'errors' (circuit open)

    Args:
      feed_id:      The feed's UUID
      success:      True if the RSS fetch succeeded (even if 0 new articles)
      new_articles: Count of new articles inserted this poll
      error_msg:    Error description (for logging, not stored in DB)
      reactivate:   Bring a dormant feed back to active (see above)
    """
    from pipeline.config import MAX_ERRORS_BEFORE_DISABLE
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
                patch[FEED_COL_LAST_NEW] = now
            if reactivate:
                patch[FEED_COL_IS_ACTIVE] = True
                patch[FEED_COL_DISABLED_REASON] = None
                log.info("REACTIVATED dormant feed %s (found %d new articles)",
                         feed_id, new_articles)

        else:
            # Failed poll - read current fail_count, increment it
            current_fails = _get_feed_fail_count(feed_id)
            new_fail_count = current_fails + 1

            patch = {
                FEED_COL_LAST_POLLED: now,
                FEED_COL_FAIL_COUNT:  new_fail_count,
            }

            if new_fail_count >= MAX_ERRORS_BEFORE_DISABLE:
                patch[FEED_COL_IS_ACTIVE] = False
                patch[FEED_COL_DISABLED_REASON] = DISABLED_ERRORS
                log.warning(
                    "CIRCUIT OPEN: feed %s disabled after %d consecutive errors. "
                    "Last error: %s",
                    feed_id, new_fail_count, error_msg,
                )

        _update_feed_row(feed_id, patch)

    except Exception as e:
        # Don't crash the pipeline if state update fails - just log
        log.error("update_feed_after_poll(%s) failed: %s", feed_id, e)


def mark_feed_dormant(feed_id: str, reason: str) -> None:
    """
    Mark a feed as dormant (is_active=False, disabled_reason='dormant') due to
    prolonged inactivity.

    Called by the circuit breaker when a feed hasn't produced new articles
    in DORMANCY_DAYS days.  Dormant feeds are re-polled once every
    DORMANT_RECHECK_INTERVAL_DAYS by get_due_feeds() and re-activated
    automatically if they produce an article.  Manual re-activation:
      UPDATE feeds SET is_active=True, fail_count=0, disabled_reason=NULL
      WHERE id='...';
    """
    try:
        _update_feed_row(feed_id, {
            FEED_COL_IS_ACTIVE: False,
            FEED_COL_DISABLED_REASON: DISABLED_DORMANT,
        })
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
        client = get_client()
        rows = _paginate(
            lambda: (
                client.table(TABLE_ARTICLES)
                .select(ART_COL_URL_HASH)
                .eq(ART_COL_FEED_ID, feed_id)
                .order(ART_COL_CRAWLED, desc=True)
                .order("id", desc=True)
            ),
            limit=limit,
        )
        hashes = {row[ART_COL_URL_HASH] for row in rows}
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
        client = get_client()
        # Paginated: PostgREST caps one response at max_rows (1000) so a bare
        # .limit(10000) silently returned only the newest 1000 titles.
        rows = _paginate(
            lambda: (
                client.table(TABLE_ARTICLES)
                .select("title_simhash")
                .not_.is_("title_simhash", "null")
                .order(ART_COL_CRAWLED, desc=True)
                .order("id", desc=True)
            ),
            limit=limit,
        )
        return {row["title_simhash"] for row in rows}
    except Exception as e:
        log.warning("load_recent_simhashes failed: %s — using empty set", e)
        return set()


_HASH_CHUNK = 150   # url_hash values per IN (...) lookup — keeps the URL short


def existing_hashes(hashes: Iterable[int]) -> set[int]:
    """
    Which of these url_hash values are already stored — ONE indexed lookup per
    chunk of _HASH_CHUNK hashes.

    The poller calls this once per feed with the hashes of the feed's CURRENT
    entries (new + legacy normalisation). It replaced:
      - load_recent_hashes(): the feed's last 2,000 url_hashes — 2,000 random
        heap reads per feed, ~2.7M per breaking_news pass, which saturated the
        production Micro instance's disk I/O on 2026-10-03; and
      - url_hash_exists() per entry: one round trip per RSS item.
    production stores url_hash as char(32) (space-padded); values are parsed
    back to int. Fails open (empty set) — the UNIQUE index is the final guard.
    """
    wanted = sorted({int(h) for h in hashes})
    found: set[int] = set()
    for i in range(0, len(wanted), _HASH_CHUNK):
        chunk = wanted[i:i + _HASH_CHUNK]
        try:
            resp = _retry(lambda: (
                get_client().table(TABLE_ARTICLES)
                .select(ART_COL_URL_HASH)
                .in_(ART_COL_URL_HASH, chunk)
                .execute()
            ))
        except Exception as e:
            log.warning("existing_hashes lookup failed (%d hashes): %s - failing open", len(chunk), e)
            continue
        for r in resp.data or []:
            try:
                found.add(int(str(r[ART_COL_URL_HASH]).strip()))
            except (TypeError, ValueError):
                pass
    return found


def url_hash_exists(url_hash: int | Iterable[int]) -> bool:
    """
    Check the database if an article with this url_hash - or ANY of several
    hashes - already exists.

    Passing several hashes (new + legacy normalisation, see dedup.py) is done
    with ONE .in_() query, not one query per hash.

    This is the AUTHORITATIVE check (layer 2 fallback after the in-memory check).
    It queries the UNIQUE index on articles.url_hash which makes it very fast.

    Returns True if the article already exists, False if it's new.
    Fails open (returns False) on database errors - the UNIQUE constraint
    will catch any actual duplicates at insert time.
    """
    hashes = [url_hash] if isinstance(url_hash, int) else sorted(set(url_hash))
    if not hashes:
        return False
    try:
        resp = _retry(lambda: (
            get_client()
            .table(TABLE_ARTICLES)
            .select(ART_COL_URL_HASH, count="exact")
            .in_(ART_COL_URL_HASH, hashes)
            .limit(1)
            .execute()
        ))
        return (resp.count or 0) > 0
    except Exception as e:
        log.warning("url_hash_exists check failed: %s - failing open", e)
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
      (inserted_count, duplicate_count) tuple.  Rows that FAILED for any reason
      other than a conflict are in neither count (they are logged); callers can
      derive them as len(rows) - inserted - duplicates.
    """
    inserted_rows, duplicates = upsert_articles_returning(rows)
    return len(inserted_rows), duplicates


def upsert_articles_returning(rows: list[dict]) -> tuple[list[dict], int]:
    """
    upsert_articles(), but returns the INSERTED rows as the database stored
    them (with their ids) — the hand-off to clustering / enrichment needs the
    ids (pipeline/handoff.py). Conflict-skipped rows are not returned.
    """
    if not rows:
        return [], 0

    inserted: list[dict] = []
    duplicates = 0

    try:
        resp = _retry(lambda: (
            get_client().table(TABLE_ARTICLES)
            .upsert(rows, on_conflict=ART_COL_URL_HASH, ignore_duplicates=True)
            .execute()
        ))
        inserted = list(resp.data or [])
        duplicates = len(rows) - len(inserted)

    except Exception as e:
        log.error("Batch upsert failed: %s — retrying row by row", e)
        # Fall back to individual inserts so partial batches aren't lost
        for row in rows:
            try:
                row_resp = _retry(lambda: get_client().table(TABLE_ARTICLES).upsert(
                    row,
                    on_conflict=ART_COL_URL_HASH,
                    ignore_duplicates=True,
                ).execute())
                # With ignore_duplicates the response only contains rows that
                # were actually inserted; a conflict-skipped row comes back
                # empty.  (Counting every non-raising call as "inserted"
                # inflated the stats.)
                if row_resp.data:
                    inserted.extend(row_resp.data)
                else:
                    duplicates += 1
            except Exception as row_e:
                log.error(
                    "Row insert failed url=%s: %s",
                    row.get("url", "?"), row_e,
                )

    return inserted, duplicates


class FeedPollBatch:
    """
    Feed poll outcomes of one run, written with wizer_record_feed_polls
    (docs/bulk_io_migration.sql) — one call per FEED_POLL_BATCH feeds instead
    of update_feed_after_poll's read + write per feed. Same rules; the
    counters are incremented in SQL.

    If the function is missing (migration not applied yet) the batch falls
    back to update_feed_after_poll / mark_feed_dormant per feed, so a deploy
    ahead of the migration still records every poll.
    """

    def __init__(self, batch_size: int = FEED_POLL_BATCH) -> None:
        self.items: list[dict] = []
        self.batch_size = max(int(batch_size), 1)
        self.written = 0

    def add(self, feed_id, success: bool, new_articles: int = 0,
            reactivate: bool = False, dormant: bool = False) -> None:
        self.items.append({"id": str(feed_id), "success": bool(success),
                           "new_articles": int(new_articles or 0),
                           "reactivate": bool(reactivate), "dormant": bool(dormant)})

    def flush(self) -> int:
        """Write everything recorded so far; returns the number of feeds updated."""
        pending, self.items = self.items, []
        for i in range(0, len(pending), self.batch_size):
            chunk = pending[i:i + self.batch_size]
            try:
                resp = _retry(lambda: get_client().rpc("wizer_record_feed_polls", {
                    "p_items": chunk, "p_max_errors": MAX_ERRORS_BEFORE_DISABLE,
                }).execute())
                self.written += int(resp.data or 0)
            except Exception as e:
                if "wizer_record_feed_polls" not in str(e):
                    log.error("record_feed_polls failed for %d feeds: %s — per-feed fallback", len(chunk), e)
                else:
                    log.warning("wizer_record_feed_polls missing (apply docs/bulk_io_migration.sql) "
                                "— per-feed fallback")
                for it in chunk:
                    update_feed_after_poll(it["id"], it["success"], it["new_articles"], "",
                                           it["reactivate"])
                    if it["dormant"]:
                        mark_feed_dormant(it["id"], "dormant (batched fallback)")
                    self.written += 1
        return self.written


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUN LOGGING
#
# Two-phase design so a run is recorded even if the job is killed mid-way:
#   1. log_run_start()  — called at the TOP of run_pipeline, before any feeds
#                         Inserts a row immediately and returns its id
#   2. log_run_finish() — called in a finally block at the END
#                         Updates the row with final stats
#
# Both `cadence` (new) and `tier` (legacy name of the same thing) are sent.
# docs/migration.sql adds whichever is missing, but a deployment may still lack
# either one, so _write_run_row() drops any column PostgREST reports as missing
# and retries.
# ─────────────────────────────────────────────────────────────────────────────

_RUN_OPTIONAL_COLS = ("cadence", "tier")


def _missing_run_column(exc: Exception, row: dict) -> str | None:
    """Name of the optional column the error complains about, if any."""
    text = str(exc)
    if "PGRST204" not in text and "42703" not in text and "does not exist" not in text:
        return None
    for col in _RUN_OPTIONAL_COLS:
        if col in row and col in text:
            return col
    return None


def _write_run_row(write: Callable[[dict], object], payload: dict) -> bool:
    """
    Run `write(row)` (an insert or update) and, if the DB reports a missing
    optional column, drop that column and retry.  Up to one retry per optional
    column.  Returns True on success.  Other errors are logged and swallowed -
    run logging must never crash the pipeline.
    """
    row = dict(payload)
    for _ in range(len(_RUN_OPTIONAL_COLS) + 1):
        try:
            write(row)
            return True
        except Exception as e:
            missing = _missing_run_column(e, row)
            if missing:
                log.debug("pipeline_runs.%s column missing - retrying without it", missing)
                row.pop(missing)
                continue
            log.error("pipeline_runs write FAILED: %s", e)
            return False
    return False


def log_run_start(cadence: str | None, dry_run: bool) -> str | None:
    """
    Insert a row at the START of a pipeline run.
    Returns the new row id (uuid) so log_run_finish can update it.
    Returns None if the insert fails.
    """
    client = get_client()
    cadence_val = cadence or "all"
    payload = {
        "tier":            cadence_val,
        "cadence":         cadence_val,
        "feeds_attempted": 0,
        "feeds_skipped":   0,
        "new_articles":    0,
        "near_duplicates": 0,
        "exact_duplicates": 0,
        "errors":          0,
        "duration_s":      0.0,
        "dry_run":         dry_run,
    }
    result: dict = {}

    def write(row: dict) -> None:
        resp = client.table(TABLE_RUNS).insert(row).execute()
        result["id"] = (resp.data or [{}])[0].get("id")

    if not _write_run_row(write, payload):
        log.error("log_run_start FAILED")
        return None
    log.info("pipeline_runs row created (id=%s, cadence=%s)", result.get("id"), cadence_val)
    return result.get("id")


def log_run_finish(run_id: str | None, summary: dict) -> None:
    """
    Update the pipeline_runs row with final stats.
    If run_id is None (start failed), falls back to a fresh insert.
    """
    client = get_client()
    cadence_val = summary.get("cadence", "all")
    payload = {
        "tier":            cadence_val,
        "cadence":         cadence_val,
        "feeds_attempted": summary.get("feeds_attempted", 0),
        "feeds_skipped":   summary.get("feeds_skipped", 0),
        "new_articles":    summary.get("new_articles", 0),
        "near_duplicates": summary.get("near_duplicates", 0),
        "exact_duplicates": summary.get("exact_duplicates", 0),
        "errors":          summary.get("errors", 0),
        "duration_s":      summary.get("duration_s", 0.0),
        "dry_run":         summary.get("dry_run", False),
    }

    def write(row: dict) -> None:
        if run_id:
            client.table(TABLE_RUNS).update(row).eq("id", run_id).execute()
        else:
            client.table(TABLE_RUNS).insert(row).execute()

    if _write_run_row(write, payload):
        log.info(
            "pipeline_runs updated - cadence=%s new=%d errors=%d duration=%.1fs",
            cadence_val, payload["new_articles"], payload["errors"], payload["duration_s"],
        )
    else:
        log.error("log_run_finish FAILED")


def log_run(summary: dict) -> None:
    """Backward-compat wrapper — used by tests. Prefer log_run_start/finish."""
    log_run_finish(None, summary)


# ─────────────────────────────────────────────────────────────────────────────
# ARTICLE TABLE SIZE CAP
# ─────────────────────────────────────────────────────────────────────────────

def fetch_uncrawled_articles(limit: int, max_age_days: int = 7) -> list[dict]:
    """
    Fetch articles where is_crawled=False, published within max_age_days,
    with a title (not empty stubs).  Ordered oldest-first so fresh articles
    at the front of the queue aren't starved by a flood of old failures.

    Called by recrawl.py to build the retry queue.

    Returns a list of dicts with id, url, domain, top_image_url columns.
    Returns empty list on DB error (safe to skip — next run will retry).
    """
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    try:
        resp = (
            get_client()
            .table(TABLE_ARTICLES)
            .select("id, url, domain, top_image_url, og_tags")
            .eq(ART_COL_CRAWLED, False)
            .gt("published_at", cutoff)
            .not_.is_("title", "null")
            .order("published_at", desc=False)   # oldest failing first
            .limit(limit)
            .execute()
        )
        rows = resp.data or []
        log.info("fetch_uncrawled_articles: %d articles to retry (max_age=%dd)", len(rows), max_age_days)
        return rows
    except Exception as e:
        log.error("fetch_uncrawled_articles failed: %s", e)
        return []


def update_article_crawl(
    article_id: int,
    full_text: str,
    is_crawled: bool,
    crawl_strategy: str,
    top_image_url: str = "",
    og_tags: dict | None = None,
) -> bool:
    """
    Write re-crawl results back to an article row.

    Called by recrawl.py after a successful (or permanently-failed) retry.
    Only updates fields that the re-crawl can meaningfully improve:
      - full_text / is_crawled / crawl_strategy
      - top_image_url if we found one and the article had none
      - og_tags if provided (merges into existing, doesn't overwrite)

    Returns True on success.
    """
    patch: dict = {
        "full_text":       full_text[:MAX_ARTICLE_BODY_CHARS] if full_text else "",
        "is_crawled":      is_crawled,
        "crawl_strategy":  crawl_strategy,
        "crawled_at":      datetime.now(timezone.utc).isoformat(),
    }
    if top_image_url:
        patch["top_image_url"] = top_image_url[:500]
    if og_tags:
        patch["og_tags"] = og_tags

    try:
        get_client().table(TABLE_ARTICLES).update(patch).eq("id", article_id).execute()
        return True
    except Exception as e:
        log.error("update_article_crawl(%s) failed: %s", article_id, e)
        return False


_PRUNE_CHUNK = 50_000      # rows deleted per RPC call — keeps each statement short
_PRUNE_MAX_CALLS = 100     # safety stop: at most 5M rows per pipeline run


def prune_articles_if_needed() -> int:
    """
    Delete the oldest articles once the table exceeds ARTICLE_HARD_LIMIT,
    bringing it back down to ARTICLE_PRUNE_TARGET (so it doesn't fire every run).

    Runs in the database (wizer_prune_articles, docs/ingestion_fixes_migration.sql),
    walking the primary key in chunks of _PRUNE_CHUNK until the target is met.

    WHY NOT IN PYTHON (the previous implementation):
      It fetched the `excess` oldest ids through PostgREST, which caps every
      response at max_rows (1000 on Supabase) — so above the hard limit each
      run deleted at most 1000 rows while ingestion added far more, and the
      table never came back under the limit. It also sorted the whole table by
      the unindexed created_at column.

    Returns the number of articles deleted (0 if pruning was not needed).
    Never raises — a failed prune must not fail the ingestion run.
    """
    if ARTICLE_HARD_LIMIT <= 0 or ARTICLE_PRUNE_TARGET <= 0:
        return 0      # pruning disabled (default) — see pipeline/config.py
    deleted = 0
    try:
        for _ in range(_PRUNE_MAX_CALLS):
            resp = get_client().rpc("wizer_prune_articles", {
                "p_hard_limit": ARTICLE_HARD_LIMIT if deleted == 0 else ARTICLE_PRUNE_TARGET,
                "p_target":     ARTICLE_PRUNE_TARGET,
                "p_max_delete": _PRUNE_CHUNK,
            }).execute()
            n = int(resp.data or 0)
            deleted += n
            if n < _PRUNE_CHUNK:
                break
        if deleted:
            log.info("Pruned %d oldest articles (target %d rows)", deleted, ARTICLE_PRUNE_TARGET)
        return deleted
    except Exception as e:
        log.warning("prune_articles_if_needed failed after %d deletions "
                    "(is docs/ingestion_fixes_migration.sql applied?): %s", deleted, e)
        return deleted
