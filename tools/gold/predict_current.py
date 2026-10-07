#!/usr/bin/env python3
"""
tools/gold/predict_current.py
═════════════════════════════
Run the production category classifier (enrichment/steps/classifier.py,
mDeBERTa zero-shot) over the gold sample, for the baseline in
tools/gold/evaluate.py.

  python tools/gold/predict_current.py     # writes tools/gold/pred_current.json (resumable)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).parent
OUT = HERE / "pred_current.json"


def main() -> int:
    import torch
    torch.set_num_threads(4)
    from enrichment.steps.classifier import classify_article
    items = json.loads((HERE / "category_sample.json").read_text(encoding="utf-8"))
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t0 = time.perf_counter()
    for n, it in enumerate(items, 1):
        key = str(it["gid"])
        if key in done:
            continue
        done[key] = classify_article(it["title"], it["description"], "")
        if n % 50 == 0:
            OUT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            rate = (time.perf_counter() - t0) / max(len(done), 1)
            print(f"{len(done)}/{len(items)}  {rate:.2f} s/item", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(done)} predictions → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
