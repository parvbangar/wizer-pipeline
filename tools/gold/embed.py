#!/usr/bin/env python3
"""
tools/gold/embed.py
═══════════════════
Embed a sample file with the production clustering embedding (multilingual-E5-base,
enrichment/steps/embedding.py), so the category head is trained and scored on exactly
the vector the pipeline already computes for every article.

  python tools/gold/embed.py category_sample.json      # → category_sample.npz  (gids, X)
  python tools/gold/embed.py train_sample.json

The text is build_embedding_text(title, description, "") — headline + description,
as for articles whose body was not crawled. Cached: an existing .npz is reused.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent


def embed_file(name: str) -> Path:
    import numpy as np
    from enrichment.steps.embedding import build_embedding_text, embed_texts

    src = HERE / name
    out = src.with_suffix(".npz")
    if out.exists():
        return out
    items = json.loads(src.read_text(encoding="utf-8"))
    texts = [build_embedding_text(it["title"], it.get("description"), "") for it in items]
    t0 = time.perf_counter()
    vecs = embed_texts(texts)
    dim = len(next(v for v in vecs if v is not None))
    X = np.array([v if v is not None else [0.0] * dim for v in vecs], dtype=np.float32)
    gids = np.array([int(it["gid"]) for it in items])
    np.savez_compressed(out, gids=gids, X=X)
    print(f"{len(items)} embedded in {time.perf_counter() - t0:.0f} s "
          f"({sum(v is None for v in vecs)} empty) → {out.name}")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    for name in sys.argv[1:]:
        embed_file(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
