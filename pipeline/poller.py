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
  │  3. download the feed bytes with a timeout + size cap + wall-clock  │
  │     deadline, then feedparser.parse(bytes) [thread]                 │
  │  4. For each RSS entry:                                              │
  │       a. normalise_url() + url_hash() (new AND legacy hash)         │
  │       b. Check in-memory hash set (Layer 1 dedup)                   │
  │       c. db.url_hash_exists() (Layer 2 dedup — authoritative,       │
  │          one .in_() query for new+legacy hash)                      │
  │       d. crawler.crawl_article() — fetch full text [thread]         │
  │       e. simhash(final title) + near-duplicate check                │
  │  5. db.upsert_articles(batch) — bulk insert                         │
  │  6. db.update_feed_after_poll() — fail_count, last_polled_at,       │
  │     last_new_article_at (only if >0 new articles)                   │
  │  7. circuit_breaker: check dormancy (only if 0 new articles)        │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import html as html_module
import io
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

from pipeline import db
from pipeline.db import log_run_start, log_run_finish
from pipeline.circuit_breaker import (
    check_dormancy, should_skip_feed, get_last_new_article_date,
    has_new_article_tracking, log_circuit_state,
)
from pipeline.config import (
    MAX_CONCURRENT_FEEDS, MAX_CONCURRENT_ARTICLES, EXECUTOR_MAX_WORKERS,
    FEED_FETCH_TIMEOUT_SECONDS, FEED_FETCH_DEADLINE_SECONDS, FEED_MAX_BYTES,
    LEGACY_URL_HASH_CHECK,
    FEED_COL_ID, FEED_COL_URL, FEED_COL_FINAL_URL,
    FEED_COL_CADENCE, FEED_COL_FAIL_COUNT,
    HANDOFF_PATH, STORE_FULL_TEXT, MAX_ARTICLE_BODY_CHARS,
)
from pipeline.handoff import Handoff
from pipeline.crawler import CrawledArticle, crawl_article, reset_domain_failures
from pipeline.dedup import (
    normalise_url, url_hash, legacy_url_hash, simhash, is_near_duplicate,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RSS FEED FETCHING
# ─────────────────────────────────────────────────────────────────────────────

_FEED_USER_AGENT = "NewsIngestBot/2.0"
_FETCH_GRACE_SECONDS = 5.0   # extra asyncio-level slack on top of the download deadline


class FeedFetchError(Exception):
    """A feed could not be downloaded (timeout, HTTP error, too large ...)."""


def _download_feed(
    feed_url: str,
    timeout: float = FEED_FETCH_TIMEOUT_SECONDS,
    deadline: float = FEED_FETCH_DEADLINE_SECONDS,
    max_bytes: int = FEED_MAX_BYTES,
) -> tuple[bytes, dict, str]:
    """
    Download a feed with hard limits.  Returns (body, response_headers, final_url).

    WHY NOT feedparser.parse(url)?
      feedparser's own fetcher has NO timeout, so one server that accepts the
      connection and then stalls blocks a worker thread forever (and, with the
      run waiting on it, the whole pipeline_runs row never gets finalised).

    Three independent limits:
      timeout   - socket timeout for connect and for EACH recv.
      deadline  - wall-clock budget for the whole download.  A "slow drip"
                  server sending 1 byte every few seconds never trips the
                  socket timeout, but is cut off here (checked between chunks;
                  overshoot is at most one `timeout`).
      max_bytes - body size cap (also applied to the DECOMPRESSED size, so a
                  gzip bomb can't exhaust memory).

    Raises FeedFetchError on any failure.
    """
    parsed = urllib.parse.urlparse(feed_url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise FeedFetchError(f"unsupported URL scheme: {parsed.scheme!r}")

    request = urllib.request.Request(feed_url, headers={
        "User-Agent":      _FEED_USER_AGENT,
        "Accept":          "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
        "Accept-Encoding": "gzip, identity",
    })
    started = time.monotonic()

    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            final_url = resp.geturl() or feed_url
            chunks: list[bytes] = []
            total = 0
            while True:
                if time.monotonic() - started > deadline:
                    raise FeedFetchError(f"download exceeded {deadline:.0f}s wall-clock deadline")
                # read1(): returns as soon as ANY data arrives.  Plain read(n)
                # blocks until n bytes (or EOF) have accumulated, which would
                # let a slow-drip server hold us far past the deadline.
                chunk = resp.read1(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise FeedFetchError(f"response larger than {max_bytes} bytes")
                chunks.append(chunk)
    except FeedFetchError:
        raise
    except urllib.error.HTTPError as e:
        raise FeedFetchError(f"HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise FeedFetchError(f"URL error: {e.reason}") from e
    except TimeoutError as e:                       # socket.timeout is TimeoutError on 3.10+
        raise FeedFetchError(f"timed out after {timeout:.0f}s") from e
    except Exception as e:
        raise FeedFetchError(f"{type(e).__name__}: {e}") from e

    body = b"".join(chunks)
    if "gzip" in headers.get("content-encoding", "").lower():
        try:
            # decompressobj + max_length bounds the decompressed size
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            body = d.decompress(body, max_bytes + 1)
        except zlib.error as e:
            raise FeedFetchError(f"bad gzip body: {e}") from e
        if len(body) > max_bytes:
            raise FeedFetchError(f"decompressed response larger than {max_bytes} bytes")
        headers.pop("content-encoding", None)

    return body, headers, final_url


def _fetch_rss_blocking(feed_url: str) -> tuple[list[dict], dict, str | None]:
    """
    Fetch and parse an RSS/Atom feed.
    Returns (entries, feed_meta, error_message_or_None).

    WHY IS THIS A SEPARATE FUNCTION?
      The download and feedparser are synchronous (blocking).  They can't be
      used directly in async code without wrapping them.  We put them in a
      separate function that runs in a thread pool via run_in_executor().

    We download the bytes ourselves (_download_feed: timeout, size cap,
    wall-clock deadline) and hand them to feedparser, passing the response
    headers and the FINAL url as content-location so relative links inside the
    feed still resolve against the right base.

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
        body, headers, final_url = _download_feed(feed_url)
    except FeedFetchError as e:
        return [], {}, f"fetch failed: {e}"

    response_headers = dict(headers)
    response_headers["content-location"] = final_url   # base for relative links

    try:
        # BytesIO (a stream) so feedparser never mistakes the payload for a
        # URL or a local file path.
        result = feedparser.parse(io.BytesIO(body), response_headers=response_headers)
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

# Returned by _process_one_entry for entries with no usable http(s) URL, so they
# are NOT miscounted as exact duplicates.
_INVALID_ENTRY = object()


def _has_http_scheme(url: str) -> bool:
    """Case-insensitive http(s):// check ("HTTP://x.com" is valid)."""
    return url[:7].lower() == "http://" or url[:8].lower() == "https://"

def _entry_hash_candidates(entries: list) -> set[int]:
    """Every url_hash (new + legacy normalisation) of a feed's entries."""
    out: set[int] = set()
    for entry in entries:
        raw_url = (entry.get("link") or entry.get("id") or "").strip()
        if not raw_url or not _has_http_scheme(raw_url):
            continue
        out.add(url_hash(raw_url))
        if LEGACY_URL_HASH_CHECK:
            out.add(legacy_url_hash(raw_url))
    return out


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
    Returns _INVALID_ENTRY if the entry has no usable http(s) link.
    Returns CrawledArticle with is_duplicate=True if it's a near-duplicate.
    Returns CrawledArticle normally for new unique articles.
    Any exception propagates; poll_one_feed logs it and counts it as an error.
    """
    # Get article URL — prefer 'link' over 'id' (id is sometimes a GUID, not URL)
    raw_url = (entry.get("link") or entry.get("id") or "").strip()
    if not raw_url or not _has_http_scheme(raw_url):
        return _INVALID_ENTRY

    # ── URL IDENTITY ─────────────────────────────────────────────────────────
    # `norm` is the URL we store and fetch: tracking params stripped but the
    # ORIGINAL http/https scheme kept (some publishers are http-only).
    # `h` (stored) hashes the canonical form where http == https.
    # `legacy_h` is what the pre-change normalisation produced, so articles
    # stored before that change are still recognised (see dedup.py /
    # config.LEGACY_URL_HASH_CHECK for when this can be removed).
    norm      = normalise_url(raw_url, collapse_scheme=False)
    h         = url_hash(raw_url)
    legacy_h  = legacy_url_hash(raw_url) if LEGACY_URL_HASH_CHECK else h
    candidates = {h, legacy_h}

    # ── EXACT DEDUP ─────────────────────────────────────────────────────────
    if seen_hashes & candidates:
        log.debug("EXACT DUP: %s", norm)
        return None   # stored already, or seen earlier in this poll

    # (The database check happened up front for the whole feed:
    #  poll_one_feed seeds seen_hashes with db.existing_hashes(), so this one
    #  set test covers both "already stored" and "seen earlier in this poll".)

    # Mark as seen for the rest of this poll cycle (only the NEW hash is stored)
    seen_hashes.add(h)

    # ── ENTRY TITLE (used only for logging here) ─────────────────────────────
    # The title_simhash itself is computed in crawl_article() from the SAME
    # RSS title that is stored as `title`, so the two always agree.
    title_text = html_module.unescape((entry.get("title") or "").strip())
    title_sh   = simhash(title_text) if title_text else 0

    # ── CRAWL THE ARTICLE ────────────────────────────────────────────────────
    async with article_sem:   # limit concurrent crawls
        article = await loop.run_in_executor(
            None,
            crawl_article,
            entry, feed, norm, h, title_sh,
        )

    # ── LAYER 3: NEAR-DUPLICATE DETECTION (SimHash) ──────────────────────────
    # Use the simhash of the title that is actually STORED (the crawler only
    # changes it when the feed supplied no title).
    title_sh = article.title_simhash or 0
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
    polls:       "db.FeedPollBatch | None" = None,
    handoff:     Handoff | None = None,
) -> dict:
    """
    Poll a single feed end-to-end.
    Returns a stats dict: {new: int, duplicates: int, errors: int}

    polls:   record the feed's outcome in this batch (written once per run)
             instead of updating the feed row right away.
    handoff: collect the inserted articles WITH their body for the processing
             jobs (pipeline/handoff.py); the body is then only stored in
             Postgres if STORE_FULL_TEXT.
    """
    feed_id  = str(feed[FEED_COL_ID])
    feed_url = feed.get(FEED_COL_FINAL_URL) or feed[FEED_COL_URL]
    cadence  = feed.get(FEED_COL_CADENCE, "unknown")
    # A dormant feed selected for its weekly re-check (see db.get_due_feeds)
    dormant_recheck = bool(feed.get("_dormant_recheck"))

    # ── CIRCUIT BREAKER: should we even try? ─────────────────────────────────
    skip, reason = should_skip_feed(feed)
    if skip:
        log.debug("SKIP [%s] %s — %s", cadence, feed_url, reason)
        return {"new": 0, "duplicates": 0, "errors": 0, "skipped": True}

    async with feed_sem:   # limit concurrent feed fetches
        t0 = time.perf_counter()
        log.info("POLLING [%s] %s", cadence, feed_url)

        # ── FETCH AND PARSE THE RSS FEED ─────────────────────────────────────
        # The blocking fetch enforces its own socket timeout and wall-clock
        # deadline, but a thread can still wedge in places urllib can't bound
        # (e.g. a DNS lookup).  asyncio.wait_for is the backstop: the poll task
        # stops waiting after deadline + grace and the feed is recorded as a
        # FAILURE (fail_count++, last_polled_at updated) like any other error,
        # instead of hanging the run.  (The stuck thread itself can't be killed;
        # it just frees its pool slot whenever the OS gives up.)
        try:
            entries, feed_meta, err_msg = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_rss_blocking, feed_url),
                timeout=FEED_FETCH_DEADLINE_SECONDS + FEED_FETCH_TIMEOUT_SECONDS + _FETCH_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            entries, feed_meta = [], {}
            err_msg = "fetch timed out (hard asyncio deadline)"

        if err_msg and not entries:
            # Complete failure — record it
            if polls is not None:
                polls.add(feed_id, success=False)
            else:
                await loop.run_in_executor(
                    None, db.update_feed_after_poll,
                    feed_id, False, 0, err_msg,
                )
            # Log circuit state (warning at 3 fails, error at 5 fails)
            current_fails = int(feed.get(FEED_COL_FAIL_COUNT) or 0) + 1
            log_circuit_state(feed_id, feed_url, current_fails)
            log.warning("POLL FAILED [%s] %s — %s", cadence, feed_url, err_msg)
            return {"new": 0, "duplicates": 0, "errors": 1, "skipped": False}

        log.debug("  %d entries from %s", len(entries), feed_url)

        # ── EXACT DEDUP AGAINST THE DATABASE — one batched lookup ────────────
        # Hash every entry (new + legacy normalisation) and ask the DB which of
        # those hashes exist, in ONE indexed IN (...) lookup per 150 hashes.
        # (Previously: load the feed's last 2,000 hashes + one query per entry
        #  — see db.existing_hashes for why that was replaced.)
        seen_hashes: set[int] = await loop.run_in_executor(
            None, db.existing_hashes, _entry_hash_candidates(entries)
        )

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
        new_articles: list[CrawledArticle] = []
        near_dups      = 0
        entry_errors   = 0    # entries whose processing raised
        exact_dups     = 0    # skipped: URL already seen (memory or DB)
        invalid        = 0    # skipped: no usable http(s) link

        for r in results:
            if isinstance(r, BaseException):
                # A processing failure is NOT a duplicate: log it visibly and
                # count it as an error (it used to be DEBUG + "exact dup").
                entry_errors += 1
                log.warning(
                    "Entry processing error in %s: %s: %s",
                    feed_url, type(r).__name__, r,
                )
            elif r is _INVALID_ENTRY:
                invalid += 1
            elif r is None:
                exact_dups += 1
            else:
                to_insert.append(r.to_db_row(include_full_text=STORE_FULL_TEXT or handoff is None))
                new_articles.append(r)
                if r.is_duplicate:
                    near_dups += 1

        if invalid:
            log.debug("  %d entries without a usable link in %s", invalid, feed_url)

        # ── BULK INSERT ───────────────────────────────────────────────────────
        inserted = 0

        if to_insert:
            if handoff is not None:
                inserted_rows, db_dups = await loop.run_in_executor(
                    None, db.upsert_articles_returning, to_insert
                )
                inserted = len(inserted_rows)
                handoff.add_inserted(inserted_rows, new_articles)
            else:
                inserted, db_dups = await loop.run_in_executor(
                    None, db.upsert_articles, to_insert
                )
            exact_dups += db_dups
            # Rows that failed to insert (neither inserted nor conflict-skipped)
            entry_errors += max(len(to_insert) - inserted - db_dups, 0)
            if inserted:
                log.info(
                    "  → %d new | %d near-dups | %d exact-dups from %s (%.0fms)",
                    inserted, near_dups, exact_dups, feed_url,
                    (time.perf_counter() - t0) * 1000,
                )

        # ── UPDATE FEED STATE ─────────────────────────────────────────────────
        # A dormant feed that just produced an article wakes up again.
        reactivate = dormant_recheck and inserted > 0
        if polls is None:
            await loop.run_in_executor(
                None, db.update_feed_after_poll,
                feed_id, True, inserted, "", reactivate,
            )
        dormant = False

        # ── DORMANCY CHECK ────────────────────────────────────────────────────
        # Only when this poll found nothing: a feed that just inserted articles
        # is by definition alive (the `feed` row is the PRE-poll snapshot, so
        # its last_new_article_at is stale and would wrongly flag it).
        # Dormant re-checks are already dormant; nothing more to decide.
        # And only after a CLEAN poll: if inserts or entries failed, "0 new"
        # says nothing about the feed (on 2026-10-03 a schema mismatch made
        # every insert fail and healthy feeds were marked dormant).
        if inserted == 0 and entry_errors == 0 and not dormant_recheck:
            if not has_new_article_tracking(feed):
                log.debug("last_new_article_at not available — dormancy check "
                          "skipped (run docs/ingestion_fixes_migration.sql)")
            else:
                last_new = get_last_new_article_date(feed)
                dormant, reason = check_dormancy(feed, last_new)
                if dormant and polls is None:
                    await loop.run_in_executor(
                        None, db.mark_feed_dormant, feed_id, reason,
                    )
                elif dormant:
                    log.warning("DORMANT: feed %s marked inactive. Reason: %s", feed_id, reason)

        if polls is not None:
            polls.add(feed_id, success=True, new_articles=inserted,
                      reactivate=reactivate, dormant=dormant)

        return {
            "new":        inserted,
            "near_dups":  near_dups,
            "exact_dups": exact_dups,
            "errors":     entry_errors,
            "skipped":    False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

async def run_pipeline(cadence: str | None = None, dry_run: bool = False) -> dict:
    """
    Poll all due feeds for the given cadence (or all cadences if cadence=None).

    Args:
      cadence: 'breaking_news', 'multiple_daily', 'daily', 'several_weekly',
               'weekly', 'monthly', 'unknown', or None for all
      dry_run: If True, fetch feeds but do NOT insert to DB (for testing)

    Returns a summary dict with counts of new articles, duplicates, errors.
    This summary is also written to the pipeline_runs table.

    TWO-PHASE LOGGING:
      A pipeline_runs row is written at the START of the run (before any feeds
      are polled) so that GitHub Actions jobs killed by a timeout still produce
      a visible record.  The row is updated with final stats in a finally block.
    """
    log.info("═══ Pipeline run start | cadence=%s | dry_run=%s ═══", cadence or "all", dry_run)
    run_start = time.perf_counter()
    loop = asyncio.get_running_loop()

    # ── EXPLICITLY SIZED THREAD POOL ─────────────────────────────────────────
    # The loop's stock default executor has ~min(32, cpu+4) threads (about 6 on
    # a 2-vCPU runner), fewer than MAX_CONCURRENT_FEEDS, so blocking fetches
    # queued behind each other and a few hung servers froze the run.  Size the
    # pool from the concurrency config instead.
    executor = ThreadPoolExecutor(
        max_workers=EXECUTOR_MAX_WORKERS, thread_name_prefix="wizer",
    )
    loop.set_default_executor(executor)
    reset_domain_failures()    # crawler's per-domain fail-fast is per RUN

    # ── WRITE START ROW immediately so a killed job still leaves a record ─────
    run_id = None
    if not dry_run:
        run_id = await loop.run_in_executor(None, log_run_start, cadence, dry_run)

    # Feed outcomes are written once at the end (dry runs keep the patched
    # per-feed no-ops); inserted articles + bodies go to the hand-off file.
    polls   = db.FeedPollBatch() if not dry_run else None
    handoff = Handoff(MAX_ARTICLE_BODY_CHARS) if (HANDOFF_PATH and not dry_run) else None

    summary = {
        "cadence":         cadence or "all",
        "feeds_attempted": 0,
        "feeds_skipped":   0,
        "new_articles":    0,
        "near_duplicates": 0,
        "exact_duplicates": 0,
        "errors":          0,
        "duration_s":      0.0,
        "dry_run":         dry_run,
    }

    try:
        # Load feeds due for polling.  A DB outage raises FeedLoadError here;
        # it propagates (after the finally block records the run) so main.py
        # exits 1 instead of reporting a green "no feeds due" run.
        try:
            feeds = await loop.run_in_executor(None, db.get_due_feeds, cadence)
        except db.FeedLoadError:
            summary["errors"] += 1
            raise
        log.info("Loaded %d feeds due for polling", len(feeds))

        if not feeds:
            log.info("No feeds due — exiting early")
            return summary

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
            poll_one_feed(feed, feed_sem, article_sem, seen_simhashes, loop,
                          polls=polls, handoff=handoff)
            for feed in feeds
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate stats
        total_new        = 0
        total_near_dups  = 0
        total_exact_dups = 0
        total_errors     = 0
        total_skipped    = 0

        for r in results:
            if isinstance(r, Exception):
                total_errors += 1
                log.error("Unhandled feed task error: %s", r)
            else:
                total_new        += r.get("new", 0)
                total_near_dups  += r.get("near_dups", 0)
                total_exact_dups += r.get("exact_dups", 0)
                total_errors     += r.get("errors", 0)
                if r.get("skipped"):
                    total_skipped += 1

        duration = round(time.perf_counter() - run_start, 2)

        summary.update({
            "feeds_attempted": len(feeds),
            "feeds_skipped":   total_skipped,
            "new_articles":    total_new,
            "near_duplicates": total_near_dups,
            "exact_duplicates": total_exact_dups,
            "errors":          total_errors,
            "duration_s":      duration,
        })

        log.info(
            "═══ Run complete | cadence=%s | %d new | %d near-dups | %d exact-dups | "
            "%d errors | %.1fs ═══",
            cadence or "all", total_new, total_near_dups, total_exact_dups,
            total_errors, duration,
        )

        # ── ARTICLE TABLE SIZE CAP ─────────────────────────────────────────
        if not dry_run:
            pruned = await loop.run_in_executor(None, db.prune_articles_if_needed)
            if pruned:
                log.info("Article cap: pruned %d oldest articles to stay under 500K", pruned)

    finally:
        # Feed state and the hand-off are written even after a crash: every
        # article that WAS inserted must reach enrichment with its body.
        if polls is not None and polls.items:
            n = await loop.run_in_executor(None, polls.flush)
            log.info("Recorded poll outcomes for %d feeds", n)
        if handoff is not None:
            summary["handoff_articles"] = handoff.write(HANDOFF_PATH)
        # Always update the DB record — even if the run was killed or crashed
        if not dry_run:
            summary["duration_s"] = round(time.perf_counter() - run_start, 2)
            await loop.run_in_executor(None, log_run_finish, run_id, summary)
        # Don't wait for any wedged fetch thread; pending work is dropped.
        executor.shutdown(wait=False, cancel_futures=True)

    return summary
