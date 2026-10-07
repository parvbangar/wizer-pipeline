#!/usr/bin/env python3
"""
tools/gold/tags.py
══════════════════
Topic tags (articles.ai_tag): gold set, training and scoring of the E5 tag head.

  python tools/gold/tags.py agreement          # pass TGA vs TGB (+ writes batches/TGADJ.json)
  python tools/gold/tags.py build              # agreed sets, else adjudicated → tag_gold.json
  python tools/gold/tags.py train [--no-save]  # one-vs-rest heads + per-tag thresholds; scores on gold
  python tools/gold/tags.py score pred.json    # any {gid: [tags]} prediction file

Protocol: the 300 en/hi items of the category gold set, tagged twice independently
under TAG_RUBRIC.md (0–3 tags each); items whose two tag sets differ go to a third
pass that sees both. Training: 4,387 en/hi items (train_sample + train2_sample)
tagged once (labels/TT*.json). The gold set is never used for fitting or tuning.
Metrics: micro- and macro-averaged F1 over the 20 tags, and exact-set match.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent
LABELS = HERE / "labels"
MODEL_OUT = ROOT / "enrichment" / "models" / "tag_head.npz"
MAX_TAGS = 3
TAGS = ["government", "elections", "financial markets", "corporate", "monetary policy", "economic policy",
        "cricket", "sports", "entertainment", "public health", "healthcare", "education", "crime",
        "technology", "artificial intelligence", "environment", "foreign policy", "conflict", "startup",
        "science"]


def _labels(prefix: str) -> dict[int, dict]:
    out = {}
    for f in sorted(LABELS.glob(f"{prefix}*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            out[int(r["gid"])] = r
    return out


def _kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def prf(gold: dict[int, set], pred: dict[int, set]) -> dict:
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    exact = 0
    for g, gs in gold.items():
        ps = pred.get(g, set())
        exact += gs == ps
        for t in TAGS:
            tp[t] += t in gs and t in ps
            fp[t] += t not in gs and t in ps
            fn[t] += t in gs and t not in ps
    def f1(t, f, n): return 2 * t / (2 * t + f + n) if t + f + n else None
    per = {t: f1(tp[t], fp[t], fn[t]) for t in TAGS}
    used = [v for t, v in per.items() if v is not None and (tp[t] + fn[t]) > 0]
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    return {"micro_f1": f1(T, F, N), "precision": T / (T + F) if T + F else 0, "recall": T / (T + N) if T + N else 0,
            "macro_f1": sum(used) / len(used), "exact": exact / len(gold), "per_tag": per,
            "support": {t: tp[t] + fn[t] for t in TAGS}}


def cmd_agreement() -> int:
    A, B = _labels("TGA"), _labels("TGB")
    both = sorted(set(A) & set(B))
    exact = sum(set(A[g]["tags"]) == set(B[g]["tags"]) for g in both)
    print(f"items {len(both)}; identical tag sets {exact / len(both):.1%}")
    m = prf({g: set(A[g]["tags"]) for g in both}, {g: set(B[g]["tags"]) for g in both})
    print(f"pass B against pass A: micro-F1 {m['micro_f1']:.3f}, macro-F1 {m['macro_f1']:.3f}")
    for t in TAGS:
        a = [int(t in A[g]["tags"]) for g in both]
        b = [int(t in B[g]["tags"]) for g in both]
        if sum(a) + sum(b):
            print(f"  {t:24s} A {sum(a):3d}  B {sum(b):3d}  kappa {_kappa(a, b):.2f}")
    items = {int(r["gid"]): r for r in json.loads((HERE / "batches" / "TGA.json").read_text(encoding="utf-8"))}
    adj = [{**{k: items[g][k] for k in ("gid", "lang", "title", "description", "domain")},
            "tags_1": A[g]["tags"], "tags_2": B[g]["tags"]}
           for g in both if set(A[g]["tags"]) != set(B[g]["tags"])]
    (HERE / "batches" / "TGADJ.json").write_text(json.dumps(adj, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(adj)} items with different tag sets → batches/TGADJ.json")
    return 0


def cmd_build() -> int:
    A, B, J = _labels("TGA"), _labels("TGB"), _labels("TGADJ")
    gold, missing = {}, []
    for g in sorted(set(A) & set(B)):
        if set(A[g]["tags"]) == set(B[g]["tags"]):
            gold[g] = sorted(A[g]["tags"])
        elif g in J:
            gold[g] = sorted(J[g]["tags"])
        else:
            missing.append(g)
    (HERE / "tag_gold.json").write_text(json.dumps({str(k): v for k, v in gold.items()}, ensure_ascii=False),
                                        encoding="utf-8")
    print(f"tag gold: {len(gold)} items, {len(missing)} undecided; tag counts:",
          dict(collections.Counter(t for v in gold.values() for t in v).most_common()))
    return 0


def _gold() -> dict[int, set]:
    return {int(k): set(v) for k, v in json.loads((HERE / "tag_gold.json").read_text(encoding="utf-8")).items()}


def cmd_score(files: list[str]) -> int:
    gold = _gold()
    for f in files:
        pred = {int(k): set(v) for k, v in json.loads((HERE / f).read_text(encoding="utf-8")).items()}
        m = prf(gold, pred)
        print(f"== {f}: micro-F1 {m['micro_f1']:.3f} (P {m['precision']:.3f} R {m['recall']:.3f}), "
              f"macro-F1 {m['macro_f1']:.3f}, exact set {m['exact']:.1%}")
    return 0


def decide(P, thresholds, max_tags=MAX_TAGS) -> list[set]:
    """Tags whose probability clears their threshold, best first, at most max_tags."""
    import numpy as np
    out = []
    for row in P:
        idx = [i for i in np.argsort(-row) if row[i] >= thresholds[i]][:max_tags]
        out.append({TAGS[i] for i in idx})
    return out


def _train_set():
    import numpy as np
    lab = _labels("TT")
    X, Y, dom = [], [], []
    for sample in ("train_sample", "train2_sample"):
        items = {int(r["gid"]): r for r in json.loads((HERE / f"{sample}.json").read_text(encoding="utf-8"))}
        d = np.load(HERE / f"{sample}.npz")
        for i, g in enumerate(d["gids"]):
            g = int(g)
            if g in lab and not lab[g].get("cant_tell") and d["X"][i].any():
                X.append(d["X"][i])
                Y.append([int(t in lab[g]["tags"]) for t in TAGS])
                dom.append(items[g]["domain"])
    return np.array(X), np.array(Y), np.array(dom)


def cmd_train(save: bool, grid: list[float]) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from enrichment.steps.embedding import CLUSTER_EMBEDDING_MODEL

    X, Y, dom = _train_set()
    print(f"train: {len(X)} items; positives per tag: "
          f"{dict(zip(TAGS, Y.sum(0).tolist()))}")

    def fit(Xa, Ya, C):
        W, b = np.zeros((len(TAGS), X.shape[1]), np.float32), np.full(len(TAGS), -10.0, np.float32)
        for j in range(len(TAGS)):
            if Ya[:, j].sum() >= 3:
                m = LogisticRegression(C=C, max_iter=5000).fit(Xa, Ya[:, j])
                W[j], b[j] = m.coef_[0], m.intercept_[0]
        return W, b

    def proba(W, b, Xa):
        return 1 / (1 + np.exp(-(Xa @ W.T + b)))

    # Out-of-fold probabilities (publisher-held-out) → one threshold per tag maximising
    # its F1; C is the grid value with the best out-of-fold micro-F1.
    truth = {i: {TAGS[j] for j in range(len(TAGS)) if Y[i, j]} for i in range(len(Y))}
    best_run = None
    for C in grid:
        oof = np.zeros(Y.shape, np.float32)
        for tr, te in GroupKFold(5).split(X, Y, dom):
            W, b = fit(X[tr], Y[tr], C)
            oof[te] = proba(W, b, X[te])
        thresholds = np.full(len(TAGS), 0.5, np.float32)
        for j in range(len(TAGS)):
            best = (0.0, 0.5)
            for t in np.arange(0.04, 0.86, 0.02):
                p = oof[:, j] >= t
                tp, fp, fn = (p & (Y[:, j] == 1)).sum(), (p & (Y[:, j] == 0)).sum(), (~p & (Y[:, j] == 1)).sum()
                f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0
                if f > best[0]:
                    best = (f, float(t))
            thresholds[j] = best[1]
        cv = prf(truth, dict(enumerate(decide(oof, thresholds))))
        print(f"  C={C:g}: publisher-held-out CV micro-F1 {cv['micro_f1']:.3f}, macro-F1 {cv['macro_f1']:.3f}")
        if best_run is None or cv["micro_f1"] > best_run[0]["micro_f1"]:
            best_run = (cv, C, thresholds)
    cv, C, thresholds = best_run
    print(f"chosen C={C:g}")

    W, b = fit(X, Y, C)
    g = np.load(HERE / "category_sample.npz")
    gold = _gold()
    keep = [i for i, gid in enumerate(g["gids"]) if int(gid) in gold]
    preds = decide(proba(W, b, g["X"][keep]), thresholds)
    pred = {str(int(g["gids"][i])): sorted(p) for i, p in zip(keep, preds)}
    (HERE / "pred_tags_head.json").write_text(json.dumps(pred), encoding="utf-8")
    m = prf(gold, {int(k): set(v) for k, v in pred.items()})
    print(f"gold ({len(gold)}): micro-F1 {m['micro_f1']:.3f} (P {m['precision']:.3f} R {m['recall']:.3f}), "
          f"macro-F1 {m['macro_f1']:.3f}, exact set {m['exact']:.1%}")
    for t in TAGS:
        if m["support"][t]:
            print(f"  {t:24s} support {m['support'][t]:3d}  F1 {m['per_tag'][t] or 0:.2f}  threshold {thresholds[TAGS.index(t)]:.3f}")
    if save:
        meta = {"embedding_model": CLUSTER_EMBEDDING_MODEL, "C": C, "max_tags": MAX_TAGS,
                "cv_micro_f1": round(cv["micro_f1"], 4), "train_items": int(len(X)),
                "trained": dt.date.today().isoformat()}
        np.savez_compressed(MODEL_OUT, W=W, b=b, thresholds=thresholds, tags=np.array(TAGS), meta=json.dumps(meta))
        print(f"model → {MODEL_OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    for s in (sys.stdout,):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["agreement", "build", "train", "score"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--grid", default="1,2,4,8", help="C values tried by cross-validation")
    a = ap.parse_args()
    if a.cmd == "agreement":
        return cmd_agreement()
    if a.cmd == "build":
        return cmd_build()
    if a.cmd == "train":
        return cmd_train(not a.no_save, [float(c) for c in a.grid.split(',')])
    return cmd_score(a.files)


if __name__ == "__main__":
    sys.exit(main())
