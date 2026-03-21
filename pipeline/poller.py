"""
poller.py
═════════
The pipeline's execution engine — polls feeds, deduplicates, crawls, inserts.

WHAT IS ASYNCIO AND WHY DO WE USE IT?
  Normally Python runs one thing at a time.  If we fetch 20 RSS feeds
  one-by-one, and each takes 2 seconds, that's 40 seconds of waiting.

  asyncio lets Python do multiple things "at the same time" by switching
  between tasks while one is waiting for a network response.

  Think of it like this: instead of waiting for each cup of tea to brew
  one at a time, you put the kettle on for all of them, and attend to each
  one when it's ready.

  With asyncio, fetching 20 feeds concurrently takes ~2 seconds total
  instead of 40 seconds.

WHAT IS A SEMAPHORE?
  A semaphore is a counter that limits how many things run simultaneously.
  MAX_CONCURRENT_FEEDS = 20 means at most 20 feeds are being fetched at once.

  WHY LIMIT?
    - Remote servers will block you if you send too many requests (rate limiting)
    - Your VM/machine has limited RAM and CPU
    - Supabase has connection limits on the free tier

WHAT IS A THREAD POOL?
  feedparser (RSS parser) and crawlers are "blocking" — they stop Python
  entirely while waiting for network.  asyncio can't switch to other tasks
  while a blocking call is running.

  Solution: run_in_executor() puts blocking calls in a thread pool.
  Python runs them in a separate thread so the asyncio loop stays free.

POLL FLOW FOR ONE FEED:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. circuit_breaker: should we skip this feed?                      │
  │  2. db.load_recent_hashes() — warm the in-memory dedup set          │
  │  3. feedparser.parse(feed_url) — fetch and parse the RSS [thread]   │
  │  4. For each RSS entry:                                              │
  │       a. normalise_url() + url_hash()                               │
  │       b. simhash(title) for near-dedup                              │
  │       c. Check in-memory hash set (Layer 1 dedup)                   │
  │       d. db.url_hash_exists() (Layer 2 dedup — authoritative)       │
  │       e. crawler.crawl_article() — fetch full text [thread]         │
  │       f. Check near-duplicate (SimHash distance)                    │
  │  5. db.upsert_articles(batch) — bulk insert                         │
  │  6. db.update_feed_after_poll() — update fail_count, last_polled_at │
  │  7. circuit_breaker: check dormancy                                  │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from pipeline import db
from pipeline.circuit_breaker import (
    check_dormancy, should_skip_feed, get_last_new_article_date,
    log_circuit_state,
)
from pipeline.config import (
    MAX_CONCURRENT_FEEDS, MAX_CONCURRENT_ARTICLES,
    FEED_COL_ID, FEED_COL_URL, FEED_COL_FINAL_URL,
    FEED_COL_TIER, FEED_COL_FAIL_COUNT,
)
from pipeline.crawler import CrawledArticle, crawl_article
from pipeline.dedup import normalise_url, url_hash, simhash, is_near_duplicate

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RSS FEED FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_rss_blocking(feed_url: str) -> tuple[list[dict], dict, str | None]:
    """
    Fetch and parse an RSS/Atom feed.
    Returns (entries, feed_meta, error_message_or_None).

    WHY IS THIS A SEPARATE FUNCTION?
      feedparser is a synchronous (blocking) library.  It can't be used
      directly in async code without wrapping it.  We put it in a
      separate function that runs in a thread pool via run_in_executor().

    feedparser NEVER raises exceptions — it uses the "bozo" flag:
      result.bozo = True  → the feed had parse errors
      result.bozo_exception → what the error was
      result.entries → the articles it managed to parse (often non-empty
                        even with bozo=True — some feeds have minor XML errors
                        but are still usable)
    """
    try:
        import feedparser
    except ImportError:
        return [], {}, "feedparser not installed — run: pip install feedparser"

    try:
        result = feedparser.parse(
            feed_url,
            request_headers={"User-Agent": "NewsIngestBot/2.0"},
        )
    except Exception as e:
        return [], {}, f"feedparser exception: {e}"

    entries = result.get("entries", [])
    feed_meta = {
        "title":    result.feed.get("title", ""),
        "link":     result.feed.get("link", ""),
        "language": result.feed.get("language", ""),
    }

    if result.get("bozo") and not entries:
        exc = result.get("bozo_exception", "unknown parse error")
        return [], feed_meta, f"bozo feed with no entries: {exc}"

    if result.get("bozo"):
        # Has entries despite parse error — use them but log the warning
        log.debug(
            "Bozo feed (minor parse error) %s: %s — got %d entries",
            feed_url, result.get("bozo_exception"), len(entries),
        )

    return entries, feed_meta, None


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE ENTRY PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

async def _process_one_entry(
    entry:        dict,           # feedparser entry
    feed:         dict,           # feed DB row
    seen_hashes:  set[int],       # in-memory dedup set (mutated in-place)
    seen_simhashes: set[int],     # in-memory simhash set (mutated in-place)
    article_sem:  asyncio.Semaphore,
    loop:         asyncio.AbstractEventLoop,
) -> CrawledArticle | None:
    """
    Process one RSS entry — dedup check, crawl, return CrawledArticle or None.

    Returns None if the article is an exact duplicate (same URL hash).
    Returns CrawledArticle with is_duplicate=True if it's a near-duplicate.
    Returns CrawledArticle normally for new unique articles.
    """
    # Get article URL — prefer 'link' over 'id' (id is sometimes a GUID, not URL)
    raw_url = (entry.get("link") or entry.get("id") or "").strip()
    if not raw_url or not raw_url.startswith("http"):
        return None

    # ── LAYER 1: EXACT DEDUP (in-memory hash check) ─────────────────────────
    norm = normalise_url(raw_url)
    h    = url_hash(norm)

    if h in seen_hashes:
        log.debug("EXACT DUP (memory): %s", norm)
        return None   # Already seen this URL in this poll cycle

    # ── LAYER 2: EXACT DEDUP (database check) ───────────────────────────────
    # This is authoritative — checks against ALL articles ever stored
    exists = await loop.run_in_executor(None, db.url_hash_exists, h)
    if exists:
        seen_hashes.add(h)   # add to memory so we skip it next time too
        log.debug("EXACT DUP (DB): %s", norm)
        return None

    # Mark as seen for the rest of this poll cycle
    seen_hashes.add(h)

    # ── COMPUTE SIMHASH FOR NEAR-DEDUP ───────────────────────────────────────
    title_text = (entry.get("title") or "").strip()
    title_sh   = simhash(title_text) if title_text else 0

    # ── CRAWL THE ARTICLE ────────────────────────────────────────────────────
    async with article_sem:   # limit concurrent crawls
        article = await loop.run_in_executor(
            None,
            crawl_article,
            entry, feed, norm, h, title_sh,
        )

    # ── LAYER 3: NEAR-DUPLICATE DETECTION (SimHash) ──────────────────────────
    if title_sh and title_sh != 0:
        for existing_sh in seen_simhashes:
            if is_near_duplicate(title_sh, existing_sh):
                article.is_duplicate = True
                log.debug(
                    "NEAR DUP (simhash): '%s'",
                    title_text[:60],
                )
                break   # mark it but still store it (is_duplicate=True)
        seen_simhashes.add(title_sh)

    return article


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FEED POLL
# ─────────────────────────────────────────────────────────────────────────────

async def poll_one_feed(
    feed:        dict,
    feed_sem:    asyncio.Semaphore,
    article_sem: asyncio.Semaphore,
    seen_simhashes: set[int],    # shared across all feeds in this run
    loop:        asyncio.AbstractEventLoop,
) -> dict:
    """
    Poll a single feed end-to-end.
    Returns a stats dict: {new: int, duplicates: int, errors: int}
    """
    feed_id  = str(feed[FEED_COL_ID])
    feed_url = feed.get(FEED_COL_FINAL_URL) or feed[FEED_COL_URL]
    tier     = feed.get(FEED_COL_TIER, "unknown")

    # ── CIRCUIT BREAKER: should we even try? ─────────────────────────────────
    skip, reason = should_skip_feed(feed)
    if skip:
        log.debug("SKIP [%s] %s — %s", tier, feed_url, reason)
        return {"new": 0, "duplicates": 0, "errors": 0, "skipped": True}

    async with feed_sem:   # limit concurrent feed fetches
        t0 = time.perf_counter()
        log.info("POLLING [%s] %s", tier, feed_url)

        # ── WARM THE IN-MEMORY DEDUP CACHE ───────────────────────────────────
        # Load recent url_hashes for THIS feed into a Python set
        seen_hashes: set[int] = await loop.run_in_executor(
            None, db.load_recent_hashes, feed_id
        )

        # ── FETCH AND PARSE THE RSS FEED ─────────────────────────────────────
        entries, feed_meta, err_msg = await loop.run_in_executor(
            None, _fetch_rss_blocking, feed_url
        )

        if err_msg and not entries:
            # Complete failure — record it
            await loop.run_in_executor(
                None, db.update_feed_after_poll,
                feed_id, False, 0, err_msg,
            )
            # Log circuit state (warning at 3 fails, error at 5 fails)
            current_fails = int(feed.get(FEED_COL_FAIL_COUNT) or 0) + 1
            log_circuit_state(feed_id, feed_url, current_fails)
            log.warning("POLL FAILED [%s] %s — %s", tier, feed_url, err_msg)
            return {"new": 0, "duplicates": 0, "errors": 1, "skipped": False}

        log.debug("  %d entries from %s", len(entries), feed_url)

        # ── PROCESS ALL ENTRIES CONCURRENTLY ─────────────────────────────────
        tasks = [
            _process_one_entry(
                entry, feed, seen_hashes, seen_simhashes, article_sem, loop
            )
            for entry in entries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ── COLLECT VALID ARTICLES ────────────────────────────────────────────
        to_insert: list[dict] = []
        near_dups  = 0

        for r in results:
            if isinstance(r, Exception):
                log.debug("Entry processing error: %s", r)
            elif r is not None:
                to_insert.append(r.to_db_row())
                if r.is_duplicate:
                    near_dups += 1

        # ── BULK INSERT ───────────────────────────────────────────────────────
        inserted = 0
        exact_dups = len(entries) - len([r for r in results if r is not None and not isinstance(r, Exception)])

        if to_insert:
            inserted, db_dups = await loop.run_in_executor(
                None, db.upsert_articles, to_insert
            )
            exact_dups += db_dups
            if inserted:
                log.info(
                    "  → %d new | %d near-dups | %d exact-dups from %s (%.0fms)",
                    inserted, near_dups, exact_dups, feed_url,
                    (time.perf_counter() - t0) * 1000,
                )

        # ── UPDATE FEED STATE ─────────────────────────────────────────────────
        await loop.run_in_executor(
            None, db.update_feed_after_poll,
            feed_id, True, inserted, "",
        )

        # ── DORMANCY CHECK ────────────────────────────────────────────────────
        last_new = get_last_new_article_date(feed)
        dormant, reason = check_dormancy(feed, last_new)
        if dormant:
            await loop.run_in_executor(
                None, db.mark_feed_dormant, feed_id, reason,
            )

        return {
            "new":        inserted,
            "near_dups":  near_dups,
            "exact_dups": exact_dups,
            "errors":     0,
            "skipped":    False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(tier: str | None = None, dry_run: bool = False) -> dict:
    """
    Poll all due feeds for the given tier (or all tiers if tier=None).

    Args:
      tier:    'tier1_high', 'tier2_medium', 'tier3_low', or None for all
      dry_run: If True, fetch feeds but do NOT insert to DB (for testing)

    Returns a summary dict with counts of new articles, duplicates, errors.
    This summary is also written to the pipeline_runs table.
    """
    log.info("═══ Pipeline run start | tier=%s | dry_run=%s ═══", tier or "all", dry_run)
    run_start = time.perf_counter()

    loop = asyncio.get_event_loop()

    # Load feeds due for polling
    feeds = await loop.run_in_executor(None, db.get_due_feeds, tier)
    log.info("Loaded %d feeds due for polling", len(feeds))

    if not feeds:
        log.info("No feeds due — exiting early")
        return {"tier": tier, "feeds": 0, "new": 0, "errors": 0}

    # Shared semaphores
    feed_sem    = asyncio.Semaphore(MAX_CONCURRENT_FEEDS)
    article_sem = asyncio.Semaphore(MAX_CONCURRENT_ARTICLES)

    # Shared SimHash set — near-dedup across all feeds in this run
    seen_simhashes: set[int] = await loop.run_in_executor(
        None, db.load_recent_simhashes
    )
    log.debug("Loaded %d recent simhashes for near-dedup", len(seen_simhashes))

    # Launch all feed polls concurrently
    tasks = [
        poll_one_feed(feed, feed_sem, article_sem, seen_simhashes, loop)
        for feed in feeds
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate stats
    total_new      = 0
    total_near_dups = 0
    total_exact_dups = 0
    total_errors   = 0
    total_skipped  = 0

    for r in results:
        if isinstance(r, Exception):
            total_errors += 1
            log.error("Unhandled feed task error: %s", r)
        else:
            total_new       += r.get("new", 0)
            total_near_dups += r.get("near_dups", 0)
            total_exact_dups += r.get("exact_dups", 0)
            total_errors    += r.get("errors", 0)
            if r.get("skipped"):
                total_skipped += 1

    duration = round(time.perf_counter() - run_start, 2)

    summary = {
        "tier":            tier or "all",
        "feeds_attempted": len(feeds),
        "feeds_skipped":   total_skipped,
        "new_articles":    total_new,
        "near_duplicates": total_near_dups,
        "exact_duplicates": total_exact_dups,
        "errors":          total_errors,
        "duration_s":      duration,
        "run_at":          datetime.now(timezone.utc).isoformat(),
        "dry_run":         dry_run,
    }

    log.info(
        "═══ Run complete | %d new | %d near-dups | %d exact-dups | "
        "%d errors | %.1fs ═══",
        total_new, total_near_dups, total_exact_dups, total_errors, duration,
    )

    if not dry_run:
        await loop.run_in_executor(None, db.log_run, summary)

    return summary
