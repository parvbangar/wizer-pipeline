#!/usr/bin/env python3
"""
cluster.py
══════════
Operator CLI for story clustering (Layer 2, step 10).

Clustering is its own job, decoupled from NLP enrichment: embedding a
headline costs ~0.1-0.3 s on CPU, the full NLP set ~15 s, so clustering keeps
up with every ingested article (~50K/day) while NLP enrichment covers what it
can. enrich.py still clusters any article it enriches that the job has not
reached yet.

  run        Cluster every article that has no cluster yet (published in the
             last --hours), oldest first, in batches. This is the scheduled job
             (.github/workflows/cluster.yml, every 30 min). It does not wait for
             NLP enrichment, so ALL ingested articles are clustered.
  backfill   Same as `run` with a 72 h default look-back — for catching up.

  maintain   Merge twin clusters, reconcile counts with member articles, and
             prune orphaned clusters. Runs every 3 h from
             .github/workflows/cluster_maintenance.yml.

  report     Print clustering health, the top stories of the last 24 h, the
             enrichment queue status and the most recent enrichment runs.

USAGE:
  python cluster.py run                            # cluster new articles (last 12 h)
  python cluster.py backfill --hours 72            # catch up on the last 3 days
  python cluster.py backfill --hours 72 --dry-run  # count only
  python cluster.py maintain                       # merge / reconcile / prune
  python cluster.py maintain --dry-run             # show planned merges
  python cluster.py report --top 20

PREREQUISITES:
  docs/MIGRATIONS.md — clustering_v2_migration.sql must have been applied.

EXIT CODES:
  0 = success   1 = crashed (DB unreachable, migration missing, …)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

log = logging.getLogger("cluster")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("urllib3", "httpx", "httpcore", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# run / backfill
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args) -> int:
    """
    Cluster every article without a cluster published in the last --hours,
    oldest first, in pages: fetch page → embed page → assign page (batched RPC).

    This is the scheduled clustering job (.github/workflows/cluster.yml). It does
    NOT wait for NLP enrichment: every ingested article — any language, any
    length — gets its story, so outlet_count counts every outlet.
    """
    from enrichment import db
    from enrichment.clustering import assign_clusters_batch
    from enrichment.steps.embedding import build_embedding_text, embed_texts

    t0 = time.perf_counter()
    deadline = t0 + args.time_budget * 60 if args.time_budget else None
    since = (datetime.now(timezone.utc) - timedelta(hours=args.hours)).isoformat()
    counts = {"seen": 0, "seed": 0, "join": 0, "gray_join": 0, "existing": 0, "skipped": 0}
    after_pub = after_id = None

    while counts["seen"] < args.limit:
        if deadline and time.perf_counter() > deadline:
            log.info("Time budget of %.0f min reached — stopping (the next run continues)", args.time_budget)
            break
        page = db.fetch_unclustered(since, min(args.batch, args.limit - counts["seen"]),
                                    after_pub, after_id)
        if not page:
            break
        counts["seen"] += len(page)
        after_pub, after_id = page[-1]["published_at"], page[-1]["id"]
        if args.dry_run:
            continue

        vectors = embed_texts([
            build_embedding_text(a.get("title"), a.get("description"), a.get("body_lead"))
            for a in page
        ])
        entities = db.fetch_entities_for_articles([a["id"] for a in page])
        results = assign_clusters_batch([
            (a, vec, entities.get(a["id"], []), a.get("image_phash"),
             a.get("language_detected") or a.get("language_code"))
            for a, vec in zip(page, vectors)
        ])
        for res in results.values():
            key = res.action if res else "skipped"
            counts[key] = counts.get(key, 0) + 1
        log.info("clustered %d articles so far (%.0f/min) — %s", counts["seen"],
                 counts["seen"] / max((time.perf_counter() - t0) / 60, 1e-9),
                 {k: v for k, v in counts.items() if k != "seen"})

    verb = "would be clustered" if args.dry_run else "processed"
    log.info("Clustering run done in %.0fs: %d articles %s (since %s) — %s",
             time.perf_counter() - t0, counts["seen"], verb, since,
             {k: v for k, v in counts.items() if k != "seen"})
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# maintain
# ─────────────────────────────────────────────────────────────────────────────

def cmd_maintain(args) -> int:
    from enrichment import db
    from enrichment.cluster_maintenance import run_maintenance

    report = run_maintenance(db, lookback_hours=args.lookback_hours,
                             prune_after_hours=args.prune_after_hours, dry_run=args.dry_run)
    if args.dry_run:
        for winner, loser, sim in report.planned[:50]:
            print(f"  would merge {loser} → {winner}   avg-link {sim:.3f}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────

def _print_row(title: str, row: dict | None) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    if not row:
        print("  (no data)")
        return
    width = max(len(k) for k in row)
    for k, v in row.items():
        print(f"  {k:<{width}}  {v}")


def cmd_report(args) -> int:
    from enrichment import db

    health = db.fetch_view("cluster_health", 1)
    _print_row("Clustering health (24 h)", health[0] if health else None)
    queue = db.fetch_view("enrichment_queue_health", 1)
    _print_row("Enrichment queue", queue[0] if queue else None)

    print(f"\n── Top {args.top} stories (24 h) " + "─" * 40)
    for s in db.fetch_view("top_stories_24h", args.top):
        langs = ",".join(s.get("language_set") or [])
        print(f"  {s['outlet_count']:3d} outlets {s['article_count']:4d} articles  [{langs:<8}] "
              f"{(s.get('headline') or '')[:90]}")

    print("\n── Recent enrichment runs " + "─" * 35)
    runs = (db.get_client().table("enrichment_runs")
            .select("started_at, runner, claimed, processed, failed, released, cluster_created, "
                    "cluster_joined, cluster_gray_joined, duration_s, stop_reason")
            .order("started_at", desc=True).limit(args.runs).execute().data or [])
    for r in runs:
        print(f"  {str(r['started_at'])[:19]}  {str(r.get('runner') or '-'):<8} "
              f"claimed {r['claimed']:5d}  ok {r['processed']:5d}  fail {r['failed']:3d}  "
              f"rel {r['released']:4d}  new {r['cluster_created']:4d}  "
              f"join {r['cluster_joined'] + r['cluster_gray_joined']:4d}  "
              f"{r.get('duration_s') or 0:7.0f}s  {r.get('stop_reason') or ''}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    for stream in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", "-v", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, help_text in (("run", "cluster every article without a cluster (scheduled job)"),
                            ("backfill", "alias of `run` with a longer default look-back")):
        b = sub.add_parser(name, help=help_text)
        b.add_argument("--hours", type=int, default=12 if name == "run" else 72,
                       help="look back this far (default: run 12, backfill 72)")
        b.add_argument("--batch", type=int, default=500, help="articles per fetch/embed page")
        b.add_argument("--limit", type=int, default=200_000, help="stop after this many articles")
        b.add_argument("--time-budget", type=float, default=0,
                       help="minutes after which to stop cleanly (0 = no limit)")
        b.add_argument("--dry-run", action="store_true")

    m = sub.add_parser("maintain", help="merge twins, reconcile counts, prune orphans")
    m.add_argument("--lookback-hours", type=int, default=6,
                   help="probe clusters touched within this window (default 6)")
    m.add_argument("--prune-after-hours", type=int, default=168,
                   help="delete orphaned clusters idle this long (default 168)")
    m.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("report", help="print clustering and queue health")
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--runs", type=int, default=10)

    args = ap.parse_args()
    _setup_logging(args.verbose)
    try:
        return {"run": cmd_run, "backfill": cmd_run, "maintain": cmd_maintain, "report": cmd_report}[args.cmd](args)
    except KeyboardInterrupt:
        log.info("Stopped by user")
        return 0
    except Exception as e:
        log.exception("cluster.py %s crashed: %s", args.cmd, e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
