#!/usr/bin/env python3
"""
enrich.py
═════════
Command-line entry point for the Layer 2 Metadata Enrichment Pipeline.

WHAT THIS FILE DOES:
  Runs the enrichment pipeline on a batch of articles that haven't been
  enriched yet (enriched_at IS NULL in the articles table).

  For each article it runs these steps in order:
    1. text_stats   — word count, reading time
    2. language     — detect actual language of article body
    3. sentiment    — positive / negative / neutral (English only)
    4. ner          — extract named entities (people, orgs, places)
    5. keywords     — top 10 keywords/phrases
    6. classifier   — assign Indian-news category (cricket, politics, etc.)
    7. images       — download top image + compute perceptual hash
    8. clustering   — group articles about the same event into story clusters

HOW TO RUN:
  # Process next 100 unenriched articles (default batch size)
  python enrich.py

  # Process a larger batch
  python enrich.py --batch-size 500

  # Dry run — compute enrichment but do NOT write to DB
  python enrich.py --dry-run --verbose

  # Re-process already-enriched articles (e.g. after upgrading NER model)
  python enrich.py --force --batch-size 1000

  # See all options
  python enrich.py --help

PREREQUISITES:
  1. Run docs/enrichment_migration.sql in Supabase SQL Editor
  2. pip install -r requirements.txt
  3. python -m spacy download en_core_web_sm

EXIT CODES:
  0 = completed successfully
  1 = crashed (DB unreachable, config missing, etc.)
"""

import argparse
import logging
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
  python enrich.py                          Enrich next 100 articles
  python enrich.py --batch-size 500         Enrich next 500 articles
  python enrich.py --dry-run --verbose      Test without writing to DB
  python enrich.py --force --batch-size 200 Re-enrich 200 articles
        """,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of articles to process in this run (default: from ENRICH_BATCH_SIZE env var or 100)",
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

    # Resolve batch size: CLI arg > env var > default (handled in runner)
    from enrichment.config import ENRICH_BATCH_SIZE
    batch_size = args.batch_size or ENRICH_BATCH_SIZE

    try:
        summary = run_enrichment(
            batch_size=batch_size,
            dry_run=args.dry_run,
            force=args.force,
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
    print(f"  Processed:    {summary.get('processed', 0)}")
    print(f"  Failed:       {summary.get('failed', 0)}")
    print(f"  Clustered:    {summary.get('clustered', 0)} (joined existing)")
    print(f"  New clusters: {summary.get('new_clusters', 0)}")
    print(f"  Duration:     {summary.get('duration_s', 0)}s")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
