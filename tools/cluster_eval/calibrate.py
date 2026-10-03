#!/usr/bin/env python3
"""
tools/cluster_eval/calibrate.py
═══════════════════════════════
Offline calibration of the story-clustering embedding model and thresholds.

PIPELINE:
  1. fetch_headlines.py          → data/headlines.json   (live publisher RSS)
  2. calibrate.py embed          → data/emb_<model>.npy  (one file per model)
  3. calibrate.py pool           → data/pairs_to_label.jsonl
       TREC-style pooling: every article's top-k neighbours under EVERY
       candidate model are pooled, so no model is judged only on pairs it
       chose itself. Pairs are spread across similarity bands so both the
       "obviously same", the "obviously different" and the contested middle
       are represented.
  4. (human / careful reviewer)  → labels/pairs_labeled.jsonl
       label: 1 = same real-world story/event, 0 = different story.
  5. calibrate.py evaluate       → per-model ROC-AUC, best-F1 threshold,
                                    precision/recall table by threshold

The text fed to the model is built by enrichment.steps.embedding.build_embedding_text
— the exact function the pipeline uses — so calibration and production can
never silently diverge.

USAGE:
  python tools/cluster_eval/calibrate.py embed --models labse e5-base
  python tools/cluster_eval/calibrate.py pool --k 4 --per-band 60
  python tools/cluster_eval/calibrate.py evaluate
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from enrichment.steps.embedding import MODEL_PROFILES, build_embedding_text  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LABELS = HERE / "labels"

# Short aliases → HuggingFace ids (profiles carry prefix/dimension settings).
ALIASES = {
    "labse":   "sentence-transformers/LaBSE",
    "e5-base": "intfloat/multilingual-e5-base",
}


def _load_headlines() -> list[dict]:
    return json.loads((DATA / "headlines.json").read_text(encoding="utf-8"))


def _emb_path(alias: str) -> Path:
    return DATA / f"emb_{alias}.npy"


def cmd_embed(args) -> None:
    from sentence_transformers import SentenceTransformer

    rows = _load_headlines()
    for alias in args.models:
        model_id = ALIASES[alias]
        profile = MODEL_PROFILES[model_id]
        texts = [
            profile.prefix + build_embedding_text(r["title"], r.get("description", ""), "")
            for r in rows
        ]
        print(f"[{alias}] encoding {len(texts)} texts with {model_id} …", file=sys.stderr)
        model = SentenceTransformer(model_id, device="cpu")
        emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        np.save(_emb_path(alias), emb.astype(np.float32))
        print(f"[{alias}] saved {emb.shape} → {_emb_path(alias)}", file=sys.stderr)


def cmd_pool(args) -> None:
    rows = _load_headlines()
    aliases = [a for a in ALIASES if _emb_path(a).exists()]
    embs = {a: np.load(_emb_path(a)) for a in aliases}

    pooled: dict[tuple[int, int], dict[str, float]] = {}
    for a, e in embs.items():
        sims = e @ e.T
        np.fill_diagonal(sims, -1.0)
        nn = np.argsort(-sims, axis=1)[:, : args.k]
        for i in range(len(rows)):
            for j in nn[i]:
                key = (min(i, int(j)), max(i, int(j)))
                pooled.setdefault(key, {})
    # score every pooled pair under every model
    for (i, j), d in pooled.items():
        for a, e in embs.items():
            d[a] = float(e[i] @ e[j])

    # Stratify PER MODEL on that model's own similarity deciles. Models live
    # on very different scales (E5 packs everything into ~0.7-1.0, LaBSE
    # spreads ~0.3-1.0), so fixed bands — or the max across models — would
    # sample almost entirely from one model's view of the data.
    rnd = random.Random(args.seed)
    chosen_set: set[tuple[int, int]] = set()
    pairs = list(pooled)
    for a in embs:
        scores = np.array([pooled[p][a] for p in pairs])
        edges = np.quantile(scores, np.linspace(0, 1, 11))
        for b in range(10):
            lo, hi = edges[b], edges[b + 1]
            in_band = [p for p, s in zip(pairs, scores) if lo <= s <= hi and p not in chosen_set]
            rnd.shuffle(in_band)
            take = in_band[: args.per_band]
            chosen_set.update(take)
            print(f"[{a}] decile {b} [{lo:.3f},{hi:.3f}]: {len(in_band):5d} pooled, {len(take)} sampled",
                  file=sys.stderr)
    chosen = list(chosen_set)

    out = DATA / "pairs_to_label.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for n, (i, j) in enumerate(sorted(chosen)):
            fh.write(json.dumps({
                "pair": n, "a": i, "b": j,
                "a_src": rows[i]["source"], "b_src": rows[j]["source"],
                "a_title": rows[i]["title"], "b_title": rows[j]["title"],
                "sims": {k: round(v, 4) for k, v in pooled[(i, j)].items()},
            }, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} pairs → {out}", file=sys.stderr)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann–Whitney U formulation, ties count half.
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    for v in np.unique(allv):                       # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def cmd_evaluate(args) -> None:
    rows = [json.loads(l) for l in (LABELS / "pairs_labeled.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    y = np.array([r["label"] for r in rows])
    print(f"{len(rows)} labelled pairs: {int(y.sum())} same-story, {int((1 - y).sum())} different\n")
    models = sorted({m for r in rows for m in r["sims"]})
    for m in models:
        s = np.array([r["sims"][m] for r in rows])
        auc = _roc_auc(s, y)
        best = (0.0, 0.0, 0.0, 0.0)
        table = []
        for t in np.arange(0.60, 0.96, 0.02):
            pred = s >= t
            tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
            fn = int((~pred & (y == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 1.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            table.append((t, p, r, f1))
            if f1 > best[3]:
                best = (t, p, r, f1)
        print(f"── {m}   ROC-AUC {auc:.3f}   best F1 {best[3]:.3f} @ {best[0]:.2f} "
              f"(P {best[1]:.3f} R {best[2]:.3f})")
        if args.table:
            for t, p, r, f1 in table:
                print(f"     t={t:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        # Cross-lingual subset (one side Devanagari, the other not).
        xl = [k for k, r in enumerate(rows)
              if _is_devanagari(r["a_title"]) != _is_devanagari(r["b_title"])]
        if xl:
            xs, xy = s[xl], y[xl]
            print(f"     cross-lingual subset: n={len(xl)} (pos {int(xy.sum())})  AUC {_roc_auc(xs, xy):.3f}")
        print()


def _is_devanagari(text: str) -> bool:
    return sum("ऀ" <= ch <= "ॿ" for ch in text) > len(text) * 0.3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("--models", nargs="+", default=list(ALIASES))
    p = sub.add_parser("pool")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--per-band", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    v = sub.add_parser("evaluate"); v.add_argument("--table", action="store_true")
    v.add_argument("--loose", action="store_true", help="count same-running-story pairs (grade 1) as positive")
    args = ap.parse_args()
    {"embed": cmd_embed, "pool": cmd_pool, "evaluate": cmd_evaluate}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
