#!/usr/bin/env python3
"""
tools/gold/predict_tags_current.py
══════════════════════════════════
Run the mDeBERTa zero-shot tagger (enrichment/steps/classifier.classify_tags) over the
en/hi gold items, for the tag baseline in tools/gold/tags.py.

  python tools/gold/predict_tags_current.py     # writes tools/gold/pred_tags_current.json (resumable)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).parent
OUT = HERE / "pred_tags_current.json"


def main() -> int:
    import torch
    torch.set_num_threads(4)
    from enrichment.steps.classifier import classify_tags
    items = json.loads((HERE / "batches" / "TGA.json").read_text(encoding="utf-8"))
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    t0 = time.perf_counter()
    for n, it in enumerate(items, 1):
        key = str(it["gid"])
        if key in done:
            continue
        done[key] = classify_tags(it["title"], it["description"], "") or []
        if n % 50 == 0:
            OUT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
            print(f"{len(done)}/{len(items)}  {(time.perf_counter() - t0) / max(len(done), 1):.2f} s/item", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(done)} → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
