#!/usr/bin/env python3
"""
tools/e2e_local.py
══════════════════
End-to-end smoke test of Layer 2 on a laptop: the REAL enrichment runner, the
REAL models (spaCy, mDeBERTa, multilingual sentiment, E5 embeddings) and the
REAL SQL (claim queue + clustering + maintenance), against a local Postgres —
no Supabase project needed.

Only the transport is swapped: enrichment.db's Supabase calls are replaced by
direct psycopg calls (tools/pg_backend.install_into_enrichment_db).

Input: live headlines from tools/cluster_eval/fetch_headlines.py
(the RSS description doubles as the article body).

USAGE:
  python tools/cluster_eval/fetch_headlines.py
  python tools/e2e_local.py --limit 300 --runners 2
  WIZER_EVAL_DSN="host=… user=…" python tools/e2e_local.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ENRICH_MIN_WORD_COUNT", "10")   # RSS descriptions stand in for bodies

import psycopg                                          # noqa: E402

from tools.db_migrations import apply_all              # noqa: E402

DSN = os.getenv("WIZER_EVAL_DSN", "host=localhost port=55432 user=postgres")
DB = "wizer_e2e"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--runners", type=int, default=2, help="concurrent runners sharing the queue")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)-24s %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "sentence_transformers", "transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    admin = psycopg.connect(f"{DSN} dbname=postgres", autocommit=True)
    admin.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {DB}")
    conn = psycopg.connect(f"{DSN} dbname={DB}")
    apply_all(conn)

    rows = json.loads((ROOT / "tools/cluster_eval/data/headlines.json").read_text(encoding="utf-8"))
    rows = rows[: args.limit]
    now = datetime.now(timezone.utc)
    for i, r in enumerate(rows):
        # Keep the real relative order but re-base onto "now" so the 48 h queue gate admits them.
        conn.execute(
            "INSERT INTO articles (url, url_hash, title, description, full_text, published_at, domain, "
            "language_code, is_crawled) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, true)",
            (r.get("link") or f"e2e://{i}", i, r["title"], r.get("description"), r.get("description"),
             now - timedelta(minutes=i), r["source"], r["lang"]),
        )
    conn.commit()

    from enrichment import db
    from enrichment.cluster_maintenance import run_maintenance
    from enrichment.runner import run_enrichment
    from enrichment.steps.embedding import embed_texts
    from tools.pg_backend import install_into_enrichment_db

    install_into_enrichment_db(psycopg.connect(f"{DSN} dbname={DB}"))
    embed_texts(["warm-up"])            # load the embedding model before the runners start
    summaries, errors = [], []

    def runner() -> None:
        try:
            summaries.append(run_enrichment(batch_size=(args.limit // args.runners) + 1))
        except Exception as e:          # reported below
            errors.append(e)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=runner) for _ in range(args.runners)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    maint = run_maintenance(db, lookback_hours=24)

    def q(sql):
        return conn.execute(sql).fetchone()

    print("\n══ End-to-end result ══════════════════════════════════════════════")
    print(f"runners={args.runners}  errors={errors!r}  elapsed={elapsed:.0f}s "
          f"({elapsed / max(len(rows), 1):.2f}s/article)")
    for s in summaries:
        print("  run:", {k: s[k] for k in ("claimed", "processed", "failed", "cluster_created",
                                           "cluster_joined", "cluster_gray_joined", "stop_reason")})
    print(f"  maintenance: merges={maint.merges} reconciled={maint.reconciled} pruned={maint.pruned}")
    print("articles enriched / total:    %s / %s" % q("SELECT count(enriched_at), count(*) FROM articles"))
    print("claimed twice (must be 0):    %s" % q("SELECT count(*) FROM articles WHERE enrich_attempts > 1"))
    print("with cluster / category:      %s / %s" % q("SELECT count(cluster_id), count(category) FROM articles"))
    print("with sentiment / summary:     %s / %s" % q("SELECT count(sentiment), count(ai_summary) FROM articles"))
    print("entities stored:              %s" % q("SELECT count(*) FROM article_entities"))
    print("clusters (multi-outlet):      %s (%s)" % q(
        "SELECT count(*), count(*) FILTER (WHERE outlet_count > 1) FROM article_clusters WHERE status = 'active'"))
    print("count drift (must be 0):      %s" % q(
        "SELECT count(*) FROM article_clusters c WHERE status = 'active' AND article_count <> "
        "(SELECT count(*) FROM articles a WHERE a.cluster_id = c.id)"))
    print("runs logged:                  %s" % q(
        "SELECT count(*) FROM enrichment_runs WHERE finished_at IS NOT NULL"))
    print("\nTop stories:")
    for h, o, n, langs, cat in conn.execute(
        "SELECT c.headline, c.outlet_count, c.article_count, c.language_set, "
        "       (SELECT mode() WITHIN GROUP (ORDER BY a.category) FROM articles a WHERE a.cluster_id = c.id) "
        "FROM article_clusters c WHERE status = 'active' "
        "ORDER BY outlet_count DESC, article_count DESC LIMIT 10"
    ).fetchall():
        print(f"  {o:2d} outlets {n:2d} arts {','.join(langs):6s} [{cat or '-':<13}] {h[:78]}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
