"""
enrichment/steps/tag_head.py
════════════════════════════
Topic tags (articles.ai_tag) from the article's multilingual-E5 vector: one logistic
head per tag with its own threshold (enrichment/models/tag_head.npz), trained on
LLM-teacher labels by tools/gold/tags.py and scored on the en/hi tag gold set
(docs/ACCURACY.md).

Replaces the mDeBERTa zero-shot tagger (classifier.classify_tags), which took 81 %
of enrichment CPU (20 NLI passes per article), when TAG_HEAD is on and the model
file is present. Same 20 tag names, so ai_tag keeps its meaning; at most 3 tags.

The vector must come from the model the head was trained on (meta.embedding_model);
otherwise predict() returns None and the caller falls back to mDeBERTa.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "tag_head.npz"

_head = None          # (W, b, thresholds, tags, meta) once loaded; False when unavailable


def _load():
    global _head
    if _head is None:
        try:
            import numpy as np
            d = np.load(MODEL_PATH, allow_pickle=False)
            _head = (d["W"].astype(np.float32), d["b"].astype(np.float32), d["thresholds"].astype(np.float32),
                     [str(t) for t in d["tags"]], json.loads(str(d["meta"])))
            log.info("Tag head loaded (%d tags, trained %s)", len(_head[3]), _head[4].get("trained"))
        except Exception as e:
            log.warning("Tag head unavailable (%s) — using the zero-shot tagger", e)
            _head = False
    return _head or None


def available(embedding_model: str) -> bool:
    head = _load()
    return bool(head) and head[4].get("embedding_model") == embedding_model


def predict(vector, embedding_model: str) -> list[str] | None:
    """Tags for one L2-normalised vector (best first, at most meta.max_tags), or None if unusable."""
    if vector is None or not available(embedding_model):
        return None
    import numpy as np
    W, b, thr, tags, meta = _head
    p = 1 / (1 + np.exp(-(W @ np.asarray(vector, dtype=np.float32) + b)))
    order = [i for i in np.argsort(-p) if p[i] >= thr[i]]
    return [tags[i] for i in order[: int(meta.get("max_tags", 3))]]
