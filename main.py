#!/usr/bin/env python3
"""
main.py
═══════
Command-line entry point for the News Ingestion Pipeline.

WHAT IS THIS FILE?
  This is the file you run.  Everything else in the pipeline/ folder
  is support code — this is the thing that kicks everything off.

HOW TO RUN:
  # Test it first — no database writes
  python main.py --dry-run --verbose

  # Poll only breaking news feeds (runs every 30 min in CI)
  python main.py --cadence breaking_news

  # Poll daily feeds
  python main.py --cadence daily

  # Poll all cadences (slow — use for local testing only)
  python main.py

  # See all options
  python main.py --help

WHAT HAPPENS WHEN YOU RUN IT:
  1. Reads your .env file for SUPABASE_URL and SUPABASE_SERVICE_KEY
  2. Connects to Supabase
  3. Fetches all feeds that are due for polling (based on their update_cadence)
  4. Polls each feed concurrently (up to MAX_CONCURRENT_FEEDS at once)
  5. For each article: deduplicates, crawls, inserts to articles table
  6. Updates feed state (last_polled_at, fail_count, etc.)
  7. Writes a summary row to pipeline_runs
  8. Exits with code 0 (success) or 1 (crash)

EXIT CODES:
  0 = completed successfully (even with some feed errors — that's normal)
  1 = pipeline crashed (DB unreachable, config missing, etc.)
"""

import argparse
import asyncio
import logging
import sys

from pipeline.poller import run_pipeline


def setup_logging(verbose: bool) -> None:
    """
    Configure logging for the pipeline.

    WHAT IS LOGGING?
      Instead of using print() everywhere, we use Python's logging module.
      It lets us:
        - Set different detail levels (DEBUG, INFO, WARNING, ERROR)
        - Add timestamps to every message
        - Redirect logs to files if needed
        - Silence noisy third-party libraries
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
        stream  = sys.stdout,
    )

    # Silence noisy third-party libraries
    for noisy in ("urllib3", "httpx", "httpcore", "chardet",
                  "trafilatura", "newspaper", "lxml"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="News RSS Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dry-run --verbose              Test without writing to DB
  python main.py --cadence breaking_news          Poll live news feeds (30 min cycle)
  python main.py --cadence multiple_daily         Poll high-frequency feeds (3 hr cycle)
  python main.py --cadence daily                  Poll daily feeds (12 hr cycle)
  python main.py --cadence several_weekly         Poll semi-weekly feeds (2 day cycle)
  python main.py --cadence weekly                 Poll weekly feeds (7 day cycle)
  python main.py --cadence monthly                Poll monthly feeds (30 day cycle)
  python main.py --cadence unknown                Poll unclassified feeds (12 hr cycle)
  python main.py                                  Poll ALL due feeds
        """,
    )

    parser.add_argument(
        "--cadence",
        choices=[
            "breaking_news", "multiple_daily", "daily",
            "several_weekly", "weekly", "monthly", "unknown",
        ],
        default=None,
        help=(
            "Poll only feeds with this update_cadence value. "
            "Defaults to all cadences."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch and parse feeds but do NOT insert articles or update "
            "feed state in Supabase. Use this for testing."
        ),
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging (very detailed output).",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("main")

    if args.dry_run:
        log.info("DRY RUN MODE — no database writes will happen")
        # Patch db functions to no-ops so nothing is written
        import pipeline.db as db_module
        db_module.upsert_articles = lambda rows: (
            log.info("DRY RUN: would insert %d articles", len(rows)) or (len(rows), 0)
        )
        db_module.update_feed_after_poll = lambda *a, **k: None
        db_module.mark_feed_dormant      = lambda *a, **k: None
        db_module.log_run                = lambda *a, **k: None

    log.info(
        "Starting pipeline | cadence=%s | dry_run=%s",
        args.cadence or "all", args.dry_run,
    )

    try:
        summary = asyncio.run(
            run_pipeline(cadence=args.cadence, dry_run=args.dry_run)
        )
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        log.exception("Pipeline crashed: %s", e)
        sys.exit(1)

    log.info("Summary: %s", summary)

    # Print a readable summary to console
    print("\n" + "═" * 60)
    print(f"  Run complete")
    print(f"  Cadence:         {summary.get('cadence', 'all')}")
    print(f"  Feeds polled:    {summary.get('feeds_attempted', 0)}")
    print(f"  New articles:    {summary.get('new_articles', 0)}")
    print(f"  Near-duplicates: {summary.get('near_duplicates', 0)}")
    print(f"  Exact duplicates:{summary.get('exact_duplicates', 0)}")
    print(f"  Errors:          {summary.get('errors', 0)}")
    print(f"  Duration:        {summary.get('duration_s', 0)}s")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
