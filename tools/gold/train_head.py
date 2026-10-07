#!/usr/bin/env python3
"""
tools/gold/train_head.py
════════════════════════
Train the category head: multinomial logistic regression on the multilingual-E5 vector
the pipeline already computes, fitted on LLM-teacher labels (labels/T*.json for
train_sample.json, labels/U*.json for train2_sample.json) and scored on the held-out 13-language gold set.

  python tools/gold/embed.py train_sample.json train2_sample.json category_sample.json
  python tools/gold/train_head.py                    # CV-select C, score on gold, write the model
  python tools/gold/train_head.py --no-save          # evaluate only

Output: enrichment/models/category_head.npz (W 13×768, b, classes, embedding model id,
training metadata) and tools/gold/pred_head.json (gold predictions for evaluate.py score).
The gold set is never used for fitting or for choosing C.
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
MODEL_OUT = ROOT / "enrichment" / "models" / "category_head.npz"


def _load_train(sets: str):
    """Training rows from every "<label prefix>:<sample name>" pair whose files exist."""
    import numpy as np
    Xs, ys, ws, doms = [], [], [], []
    for pair in sets.split(","):
        prefix, sample = pair.split(":")
        if not (HERE / f"{sample}.npz").exists():
            continue
        labels = {}
        for f in sorted((HERE / "labels").glob(f"{prefix}*.json")):
            for r in json.loads(f.read_text(encoding="utf-8")):
                labels[int(r["gid"])] = r
        items = {int(r["gid"]): r for r in json.loads((HERE / f"{sample}.json").read_text(encoding="utf-8"))}
        d = np.load(HERE / f"{sample}.npz")
        keep = [i for i, g in enumerate(d["gids"]) if int(g) in labels and not labels[int(g)].get("cant_tell")
                and d["X"][i].any()]                       # an empty text embeds to a zero vector
        gids = d["gids"][keep]
        Xs.append(d["X"][keep])
        ys.append(np.array([labels[int(g)]["category"] for g in gids]))
        ws.append(np.array([float(labels[int(g)].get("confidence", 1.0)) for g in gids]))
        doms.append(np.array([items[int(g)]["domain"] for g in gids]))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(doms)


def main() -> int:
    for s in (sys.stdout,):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="T:train_sample,U:train2_sample",
                    help="label-prefix:sample pairs (default T:train_sample,U:train2_sample)")
    ap.add_argument("--grid", default="1,2,4,8,16,32")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from enrichment.steps.embedding import CLUSTER_EMBEDDING_MODEL
    import evaluate

    X, y, w, dom = _load_train(args.sets)
    print(f"train: {len(y)} items, {len(set(dom))} publishers; "
          f"classes {dict(collections.Counter(y).most_common())}")

    def fit(C, Xa, ya, wa):
        return LogisticRegression(C=C, max_iter=5000).fit(Xa, ya, sample_weight=wa)

    best = None
    for C in [float(c) for c in args.grid.split(",")]:
        pred = np.empty_like(y)
        for tr, te in GroupKFold(5).split(X, y, dom):
            pred[te] = fit(C, X[tr], y[tr], w[tr]).predict(X[te])
        acc = float((pred == y).mean())
        print(f"  C={C:g}: publisher-held-out CV accuracy {acc:.1%}")
        if best is None or acc > best[1]:
            best = (C, acc)
    C, cv_acc = best
    model = fit(C, X, y, w)

    g = np.load(HERE / "category_sample.npz")
    probs = model.predict_proba(g["X"])
    pred = {str(int(gid)): model.classes_[i] for gid, i in zip(g["gids"], probs.argmax(1))}
    (HERE / "pred_head.json").write_text(json.dumps(pred), encoding="utf-8")
    print(f"chosen C={C:g} (CV {cv_acc:.1%}); gold:")
    evaluate.cmd_score(["pred_head.json"])

    if not args.no_save:
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        meta = {"embedding_model": CLUSTER_EMBEDDING_MODEL, "C": C, "cv_accuracy": round(cv_acc, 4),
                "train_items": int(len(y)), "trained": dt.date.today().isoformat(),
                "text": "build_embedding_text(title, description, full_text)"}
        np.savez_compressed(MODEL_OUT, W=model.coef_.astype(np.float32),
                            b=model.intercept_.astype(np.float32),
                            classes=np.array(model.classes_), meta=json.dumps(meta))
        print(f"model → {MODEL_OUT.relative_to(ROOT)} ({MODEL_OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
