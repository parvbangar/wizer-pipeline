#!/usr/bin/env python3
"""
tools/cluster_eval/simulate.py
══════════════════════════════
End-to-end clustering simulation on real headlines, through the REAL code path:

  headlines.json → enrichment.steps.embedding (production text + model)
                 → enrichment.steps.ner        (production NER)
                 → enrichment.clustering.build_assign_params (production params)
                 → wizer_assign_cluster()      (production SQL, local Postgres)

and scores the resulting clusters against the hand-labelled pairs in
labels/pairs_labeled.jsonl:

  strict P/R/F1   positive = "same specific event" (grade 2)
  loose precision a same-cluster pair counts as correct if it is the same
                  event OR the same running story (grade ≥ 1)

USAGE (needs a Postgres with pgvector ≥ 0.7; DSN via --dsn or WIZER_EVAL_DSN):
  python tools/cluster_eval/simulate.py                    # current config
  python tools/cluster_eval/simulate.py --sweep            # threshold grid
  python tools/cluster_eval/simulate.py --order desc       # newest-first (as the queue runs)
  python tools/cluster_eval/simulate.py --top 15           # print biggest stories
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import psycopg                                         # noqa: E402

from enrichment import config as cfg                   # noqa: E402
from enrichment.clustering import build_assign_params  # noqa: E402
from enrichment.steps.embedding import build_embedding_text, embed_texts  # noqa: E402
from tools.db_migrations import apply_all              # noqa: E402
from tools.pg_backend import PgBackend                  # noqa: E402
from enrichment.cluster_maintenance import merge_duplicates  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, LABELS = HERE / "data", HERE / "labels"
DEFAULT_DSN = os.getenv("WIZER_EVAL_DSN", "host=localhost port=55432 user=postgres")
SIM_DB = "wizer_cluster_sim"


# ─────────────────────────────────────────────────────────────────────────────
# Inputs (cached — embedding + NER of 1.2K articles takes a minute on CPU)
# ─────────────────────────────────────────────────────────────────────────────

def load_inputs(rows: list[dict]) -> tuple[np.ndarray, list[list[dict]]]:
    tag = cfg.CLUSTER_EMBEDDING_MODEL.split("/")[-1]
    emb_path, ent_path = DATA / f"sim_emb_{tag}.npy", DATA / "sim_entities.json"
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        texts = [build_embedding_text(r["title"], r.get("description"), "") for r in rows]
        vecs = embed_texts(texts)
        emb = np.array([v if v is not None else [0.0] * 768 for v in vecs], dtype=np.float32)
        np.save(emb_path, emb)
    if ent_path.exists():
        ents = json.loads(ent_path.read_text(encoding="utf-8"))
    else:
        from enrichment.steps.ner import extract_entities
        ents = [extract_entities(r["title"], "", r.get("description", ""), r["lang"]) for r in rows]
        ent_path.write_text(json.dumps(ents, ensure_ascii=False), encoding="utf-8")
    return emb, ents


def fresh_db(dsn: str) -> psycopg.Connection:
    admin = psycopg.connect(f"{dsn} dbname=postgres", autocommit=True)
    admin.execute(f"DROP DATABASE IF EXISTS {SIM_DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {SIM_DB}")
    admin.close()
    conn = psycopg.connect(f"{dsn} dbname={SIM_DB}")
    apply_all(conn)
    return conn


def seed_articles(conn, rows: list[dict]) -> None:
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                "INSERT INTO articles (url, url_hash, title, description, published_at, "
                "domain, language_code, is_crawled, enriched_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, true, now())",
                (r.get("link") or f"sim://{r['id']}", r["id"], r["title"], r.get("description"),
                 r.get("published_at"), r["source"], r["lang"]),
            )
    conn.commit()


def reset_clusters(conn) -> None:
    conn.execute("UPDATE articles SET cluster_id = NULL, cluster_similarity = NULL, "
                 "cluster_assignment = NULL, clustered_at = NULL")
    conn.execute("DELETE FROM article_clusters")
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# One simulation run
# ─────────────────────────────────────────────────────────────────────────────

def assign_all(conn, rows, emb, ents, overrides: dict, order: str, use_entities: bool) -> tuple[dict, float]:
    reset_clusters(conn)
    backend = PgBackend(conn)
    db_ids = {r[1]: r[0] for r in conn.execute("SELECT id, url_hash FROM articles").fetchall()}
    seq = sorted(range(len(rows)), key=lambda i: rows[i].get("published_at") or "",
                 reverse=(order == "desc"))
    actions: dict[str, int] = {}
    t0 = time.perf_counter()
    for i in seq:
        r = rows[i]
        article = {"id": db_ids[r["id"]], "title": r["title"], "domain": r["source"],
                   "published_at": r.get("published_at"), "language_code": r["lang"]}
        params = build_assign_params(article, emb[i].tolist(), ents[i] if use_entities else [],
                                     None, r["lang"])
        params.update(overrides)
        action = backend.assign_cluster(params)["action"]
        actions[action] = actions.get(action, 0) + 1
    return actions, (time.perf_counter() - t0) * 1000 / len(rows)


def evaluate(conn) -> dict:
    cluster_of = {h: c for h, c in conn.execute("SELECT url_hash, cluster_id FROM articles").fetchall()}
    labeled = [json.loads(l) for l in (LABELS / "pairs_labeled.jsonl").read_text(encoding="utf-8").splitlines() if l]
    tp = fp = fn = fp_loose = 0
    for p in labeled:
        same = cluster_of[p["a"]] == cluster_of[p["b"]]
        if same and p["grade"] == 2:
            tp += 1
        elif same:
            fp += 1
            fp_loose += p["grade"] == 0
        elif p["grade"] == 2:
            fn += 1
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 0.0
    sizes = [n for (n,) in conn.execute(
        "SELECT article_count FROM article_clusters WHERE status = 'active'").fetchall()]
    return {
        "P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0,
        "P_loose": (tp + fp - fp_loose) / (tp + fp) if tp + fp else 1.0,
        "clusters": len(sizes),
        "singleton_pct": 100.0 * sum(s == 1 for s in sizes) / max(len(sizes), 1),
        "multi_outlet": conn.execute("SELECT count(*) FROM article_clusters "
                                     "WHERE status = 'active' AND outlet_count >= 2").fetchone()[0],
    }


def run(conn, rows, emb, ents, overrides: dict, order: str, use_entities: bool,
        merge_threshold: float | None = None) -> dict:
    actions, ms = assign_all(conn, rows, emb, ents, overrides, order, use_entities)
    merges = 0
    if merge_threshold is not None:
        rep = merge_duplicates(PgBackend(conn), datetime(2000, 1, 1, tzinfo=timezone.utc),
                               threshold=merge_threshold)
        merges = rep.merges
    res = evaluate(conn)
    res.update({"actions": actions, "ms_per_article": ms, "merges": merges})
    return res


def print_top(conn, n: int) -> None:
    print(f"\nTop {n} stories by outlet count:")
    for headline, outlets, arts, langs in conn.execute(
        "SELECT headline, outlet_count, article_count, language_set FROM article_clusters "
        "ORDER BY outlet_count DESC, article_count DESC LIMIT %s", (n,)
    ).fetchall():
        print(f"  {outlets:2d} outlets / {arts:2d} articles {langs}  {headline[:95]}")


def fmt(res: dict) -> str:
    return (f"P {res['P']:.3f}  R {res['R']:.3f}  F1 {res['F1']:.3f}  P_loose {res['P_loose']:.3f} | "
            f"{res['clusters']} clusters, {res['singleton_pct']:.0f}% singletons, "
            f"{res['multi_outlet']} multi-outlet | {res['actions']} merges={res['merges']} | "
            f"{res['ms_per_article']:.1f} ms/article")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--order", choices=["asc", "desc"], default="asc")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-entities", action="store_true")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--merge", type=float, nargs="*", default=None,
                    help="run the maintenance merge after assignment, at each given threshold")
    args = ap.parse_args()

    rows = json.loads((DATA / "headlines.json").read_text(encoding="utf-8"))
    emb, ents = load_inputs(rows)
    conn = fresh_db(args.dsn)
    seed_articles(conn, rows)

    if args.merge is not None:
        base = run(conn, rows, emb, ents, {}, args.order, not args.no_entities)
        print(f"no merge          → {fmt(base)}")
        for t in args.merge or [cfg.CLUSTER_MERGE_THRESHOLD]:
            res = run(conn, rows, emb, ents, {}, args.order, not args.no_entities, merge_threshold=t)
            print(f"merge @ {t:.3f}     → {fmt(res)}", flush=True)
        if args.top:
            print_top(conn, args.top)
        return 0

    if not args.sweep:
        res = run(conn, rows, emb, ents, {}, args.order, not args.no_entities)
        print(f"config: join={cfg.CLUSTER_JOIN_THRESHOLD} gray={cfg.CLUSTER_GRAY_THRESHOLD} "
              f"anchor={cfg.CLUSTER_ANCHOR_THRESHOLD} order={args.order}\n  {fmt(res)}")
        if args.top:
            print_top(conn, args.top)
        return 0

    grid = itertools.product(
        (0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87),  # join (average-link cosine)
        (0.00, 0.02, 0.04),                          # gray margin below join (0 = no gray zone)
        (0.70, 0.76, 0.80),                          # anchor
    )
    results = []
    for join, margin, anchor in grid:
        ov = {"p_join_threshold": join, "p_gray_threshold": join - margin if margin else join,
              "p_anchor_threshold": anchor}
        res = run(conn, rows, emb, ents, ov, args.order, not args.no_entities)
        results.append((res["F1"], join, margin, anchor, res))
        print(f"join {join:.2f} gray -{margin:.2f} anchor {anchor:.2f} → {fmt(res)}", flush=True)
    results.sort(key=lambda t: t[0], reverse=True)
    print("\nBest 5 by strict F1:")
    for f1, join, margin, anchor, res in results[:5]:
        print(f"  join {join:.2f} gray -{margin:.2f} anchor {anchor:.2f} → {fmt(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
