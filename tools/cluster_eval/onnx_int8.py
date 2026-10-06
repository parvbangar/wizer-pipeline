#!/usr/bin/env python3
"""
tools/cluster_eval/onnx_int8.py
═══════════════════════════════
Can clustering use an int8 ONNX build of multilingual-e5-base instead of the
fp32 PyTorch model? Measured, not assumed: speed, per-headline agreement, and
— what matters — whether any calibrated join decision flips.

  python tools/cluster_eval/onnx_int8.py            # export (once) + compare

Inputs: data/headlines.json + data/emb_e5-base.npy (fp32, from calibrate.py
embed) + labels/pairs_labeled.jsonl (400 graded pairs).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from enrichment.steps.embedding import MODEL_PROFILES, build_embedding_text   # noqa: E402

HERE = Path(__file__).resolve().parent
MODEL = "intfloat/multilingual-e5-base"
EXPORT = HERE / "data" / "e5-base-onnx"
JOIN = 0.85


def _auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model
    import torch
    torch.set_num_threads(4)
    rows = json.loads((HERE / "data" / "headlines.json").read_text(encoding="utf-8"))
    prefix = MODEL_PROFILES[MODEL].prefix
    texts = [prefix + build_embedding_text(r["title"], r.get("description", ""), "") for r in rows]
    fp32 = np.load(HERE / "data" / "emb_e5-base.npy")

    qfile = EXPORT / "onnx" / "model_qint8_avx2.onnx"
    if not qfile.exists():
        m = SentenceTransformer(MODEL, backend="onnx", device="cpu")
        m.save_pretrained(str(EXPORT))
        export_dynamic_quantized_onnx_model(m, "avx2", str(EXPORT))
    onnx_model = SentenceTransformer(str(EXPORT), backend="onnx", device="cpu",
                                     model_kwargs={"file_name": "onnx/model_qint8_avx2.onnx",
                                                   "provider": "CPUExecutionProvider"})
    torch_model = SentenceTransformer(MODEL, device="cpu")
    onnx_model.max_seq_length = torch_model.max_seq_length = MODEL_PROFILES[MODEL].max_seq_length

    timings = {}
    for name, m in (("fp32-torch", torch_model), ("int8-onnx", onnx_model)):
        m.encode(texts[:64], batch_size=32, normalize_embeddings=True)          # warm-up
        t = time.perf_counter()
        emb = m.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        timings[name] = (time.perf_counter() - t) / len(texts) * 1000
        if name == "int8-onnx":
            int8 = emb.astype(np.float32)

    agree = np.sum(fp32 * int8, axis=1)
    pairs = [json.loads(l) for l in (HERE / "labels" / "pairs_labeled.jsonl").read_text(encoding="utf-8").splitlines()]
    a = np.array([p["a"] for p in pairs]); b = np.array([p["b"] for p in pairs])
    labels = np.array([p["label"] for p in pairs])
    s32 = np.sum(fp32[a] * fp32[b], axis=1)
    s8 = np.sum(int8[a] * int8[b], axis=1)
    flips = int(np.sum((s32 >= JOIN) != (s8 >= JOIN)))
    print(f"speed (4 threads): fp32 torch {timings['fp32-torch']:.1f} ms/headline, "
          f"int8 onnx {timings['int8-onnx']:.1f} ms/headline  ({timings['fp32-torch'] / timings['int8-onnx']:.2f}x)")
    print(f"per-headline cosine(fp32, int8): mean {agree.mean():.4f}, min {agree.min():.4f}")
    print(f"pair similarity drift: mean |Δ| {np.abs(s32 - s8).mean():.4f}, max |Δ| {np.abs(s32 - s8).max():.4f}")
    print(f"join decisions at {JOIN} that flip: {flips} of {len(pairs)}")
    print(f"AUC (strict, grade 2): fp32 {_auc(s32, labels):.4f}  int8 {_auc(s8, labels):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
