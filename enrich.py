#!/usr/bin/env python3
"""
enrich.py
═════════
Command-line entry point for the Layer 2 Metadata Enrichment Pipeline.

WHAT THIS FILE DOES:
  Claims a batch of not-yet-enriched articles from the work queue
  (docs/enrichment_queue_migration.sql) and runs the enrichment pipeline on
  them. Concurrent runs never process the same article.

  For each article it runs these steps in order (enrichment/runner.py):
     1. text_stats   — word count, reading time
     2. language     — detect actual language of article body
     3. sentiment    — positive / negative / neutral (multilingual model)
     4. ner          — named entities (people, orgs, places, …)
     5. keywords     — top 10 keywords/phrases (YAKE)
     6. classifier   — Indian-news category (mDeBERTa zero-shot)
     7. tags         — fine-grained topic tags (same model)
     8. summary      — extractive summary
     9. images       — top image perceptual hash
    10. clustering   — put the article into its story cluster (all languages)

HOW TO RUN:
  # Process the next ENRICH_BATCH_SIZE articles (default 1000)
  python enrich.py

  # Process a larger batch
  python enrich.py --batch-size 500

  # Dry run — compute enrichment but do NOT write to DB
  python enrich.py --dry-run --verbose

  # Re-process already-enriched articles (e.g. after upgrading NER model)
  python enrich.py --force --batch-size 1000

  # See all options
  python enrich.py --help

  # Stop taking new articles after 100 minutes (unprocessed ones are released)
  python enrich.py --time-budget 100

PREREQUISITES:
  1. Apply the migrations in docs/MIGRATIONS.md (Supabase SQL Editor)
  2. pip install -r requirements.txt
  3. python -m spacy download en_core_web_md && python -m spacy download xx_ent_wiki_sm

EXIT CODES:
  0 = completed successfully
  1 = crashed (DB unreachable, config missing, etc.)
"""

import argparse
import logging
import os
import sys

from enrichment.runner import run_enrichment


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level   = level,
        format  = "%(asctime)s  %(levelname)-8s  %(name)-25s  %(message)s",
        datefmt = "%Y-%m-%dT%H:%M:%S",
        stream  = sys.stdout,
    )
    # Silence noisy libraries
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "chardet"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 2 — Metadata Enrichment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enrich.py                          Enrich the next batch (ENRICH_BATCH_SIZE)
  python enrich.py --batch-size 500         Enrich up to 500 articles
  python enrich.py --dry-run --verbose      Test without writing to DB
  python enrich.py --force --batch-size 200 Re-enrich 200 articles
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of articles to process in this run (default: ENRICH_BATCH_SIZE env var, 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute enrichment but do NOT write anything to the database.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process articles that have already been enriched. "
             "Useful when upgrading models or fixing bugs.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Skip first N articles. Only used with --force / --dry-run (normal runs claim "
             "from the work queue, so parallel runners need no offsets). "
             "Default: ENRICH_OFFSET env var or 0.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Minutes after which no new article is started; unprocessed claims are "
             "released back to the queue. Default: ENRICH_TIME_BUDGET_MINUTES (0 = none).",
    )
    parser.add_argument(
        "--handoff",
        default=None,
        help="Enrich the articles of an ingest hand-off (a .jsonl.gz file or a directory of "
             "them, see pipeline/handoff.py) instead of claiming from the queue.",
    )
    parser.add_argument("--shard", type=int, default=0, help="With --handoff: this runner's index.")
    parser.add_argument("--shards", type=int, default=1, help="With --handoff: number of runners.")
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=None,
        help="Sweeper mode: only claim articles ingested at least this long ago (the "
             "hand-off runners own fresher ones). Default: ENRICH_SWEEP_MIN_AGE_HOURS or 0.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging (very detailed output per article).",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger("enrich")

    if args.dry_run:
        log.info("DRY RUN MODE — no database writes will happen")
    if args.force:
        log.info("FORCE MODE — re-processing already-enriched articles")

    # Resolve batch size and offset: CLI arg > env var > default
    from enrichment.config import ENRICH_BATCH_SIZE, ENRICH_TIME_BUDGET_MINUTES
    batch_size = args.batch_size or ENRICH_BATCH_SIZE
    time_budget = args.time_budget if args.time_budget is not None else ENRICH_TIME_BUDGET_MINUTES
    offset = args.offset if args.offset is not None else int(os.getenv("ENRICH_OFFSET", "0"))

    if args.handoff:
        from enrichment.handoff_runner import run_handoff_enrichment
        from pipeline.handoff import find_files, read, shard_of
        files = find_files(args.handoff)
        records = shard_of(read(files), args.shard, args.shards)
        log.info("Hand-off: %d files, %d articles for shard %d/%d",
                 len(files), len(records), args.shard, args.shards)
        try:
            summary = run_handoff_enrichment(records, time_budget_minutes=time_budget,
                                             dry_run=args.dry_run)
        except Exception as e:
            log.exception("Hand-off enrichment crashed: %s", e)
            sys.exit(1)
        print(f"
  Hand-off shard {args.shard}/{args.shards}: {summary}
")
        return

    min_age = args.min_age_hours if args.min_age_hours is not None         else float(os.getenv("ENRICH_SWEEP_MIN_AGE_HOURS", "0"))
    try:
        summary = run_enrichment(
            batch_size=batch_size,
            dry_run=args.dry_run,
            force=args.force,
            offset=offset,
            time_budget_minutes=time_budget,
            min_age_hours=min_age,
        )
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        log.exception("Enrichment pipeline crashed: %s", e)
        sys.exit(1)

    # Print readable summary
    print("\n" + "═" * 60)
    print("  Enrichment run complete")
    print(f"  Queue depth:   {summary.get('queue_depth_start')}")
    print(f"  Claimed:       {summary.get('claimed', 0)}")
    print(f"  Processed:     {summary.get('processed', 0)}")
    print(f"  Failed:        {summary.get('failed', 0)}")
    print(f"  Released:      {summary.get('released', 0)}")
    print(f"  Clusters:      {summary.get('cluster_created', 0)} new, "
          f"{summary.get('cluster_joined', 0) + summary.get('cluster_gray_joined', 0)} joined "
          f"({summary.get('cluster_gray_joined', 0)} via evidence), "
          f"{summary.get('cluster_skipped', 0)} not clustered")
    print(f"  Stopped:       {summary.get('stop_reason')}")
    print(f"  Duration:      {summary.get('duration_s', 0)}s")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
