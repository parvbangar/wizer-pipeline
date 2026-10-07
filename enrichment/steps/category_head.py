"""
enrichment/steps/category_head.py
═════════════════════════════════
Category from the article's multilingual-E5 vector: a 13-way linear softmax head
(enrichment/models/category_head.npz), trained on LLM-teacher labels by
tools/gold/train_head.py and scored on the 13-language gold set (docs/ACCURACY.md).

Replaces the mDeBERTa zero-shot category (classifier.classify_article) when
CATEGORY_HEAD is on and the model file is present:
  - accuracy: see docs/ACCURACY.md (gold set, per language);
  - cost: one 13×768 matrix product on a vector that is computed anyway for
    clustering, instead of 13 NLI forward passes per article.

The vector must come from the model the head was trained on (meta.embedding_model);
otherwise predict() returns None and the caller falls back to mDeBERTa.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "category_head.npz"

_head = None          # (W, b, classes, meta) once loaded; False when unavailable


def _load():
    global _head
    if _head is None:
        try:
            import numpy as np
            d = np.load(MODEL_PATH, allow_pickle=False)
            _head = (d["W"].astype(np.float32), d["b"].astype(np.float32),
                     [str(c) for c in d["classes"]], json.loads(str(d["meta"])))
            log.info("Category head loaded (%d classes, trained %s)", len(_head[2]), _head[3].get("trained"))
        except Exception as e:
            log.warning("Category head unavailable (%s) — using the zero-shot classifier", e)
            _head = False
    return _head or None


def available(embedding_model: str) -> bool:
    head = _load()
    return bool(head) and head[3].get("embedding_model") == embedding_model


def predict(vector, embedding_model: str) -> tuple[str, float] | None:
    """(category, probability) for one L2-normalised vector, or None if the head cannot be used."""
    if vector is None or not available(embedding_model):
        return None
    import numpy as np
    W, b, classes, _ = _head
    z = W @ np.asarray(vector, dtype=np.float32) + b
    z = np.exp(z - z.max())
    p = z / z.sum()
    i = int(p.argmax())
    return classes[i], float(p[i])
