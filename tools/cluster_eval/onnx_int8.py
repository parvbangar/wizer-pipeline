#!/usr/bin/env python3
"""
tools/cluster_eval/onnx_int8.py
═══════════════════════════════
Can clustering use an int8 ONNX build of multilingual-e5-base instead of the
fp32 PyTorch model? Measured, not assumed: speed, per-headline agreement and —
what matters — whether any calibrated join decision flips.

The ONNX models run on plain onnxruntime + the model's tokenizer.json with
e5's mean pooling (no optimum / transformers pinning in production). The fp32
ONNX export is checked first against PyTorch, which validates the pooling.

  python tools/cluster_eval/onnx_int8.py     # needs data/e5-base-onnx (export once, see EXPORT note)

EXPORT (once): sentence-transformers' export writes data/e5-base-onnx/onnx/
model.onnx and model_quint8_avx2.onnx (SentenceTransformer(MODEL,
backend="onnx").save_pretrained + export_dynamic_quantized_onnx_model).
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


class OnnxEncoder:
    """Mean-pooled, L2-normalised sentence vectors from an ONNX transformer."""

    def __init__(self, model_file: Path, tokenizer_file: Path, max_len: int, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_file), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(tokenizer_file))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(texts[i:i + batch_size])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]                      # (b, seq, dim)
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)


def _auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    rows = json.loads((HERE / "data" / "headlines.json").read_text(encoding="utf-8"))
    profile = MODEL_PROFILES[MODEL]
    texts = [profile.prefix + build_embedding_text(r["title"], r.get("description", ""), "") for r in rows]
    tok = EXPORT / "tokenizer.json"

    def timed(fn):
        fn(texts[:64])
        t = time.perf_counter()
        e = fn(texts)
        return e, (time.perf_counter() - t) / len(texts) * 1000

    st = SentenceTransformer(MODEL, device="cpu")
    st.max_seq_length = profile.max_seq_length
    torch_emb, t_torch = timed(lambda x: st.encode(x, batch_size=32, normalize_embeddings=True, convert_to_numpy=True))
    fp32_onnx, t_fp32 = timed(OnnxEncoder(EXPORT / "onnx" / "model.onnx", tok, profile.max_seq_length).encode)
    int8, t_int8 = timed(OnnxEncoder(EXPORT / "onnx" / "model_quint8_avx2.onnx", tok, profile.max_seq_length).encode)

    print(f"speed, 4 threads: torch fp32 {t_torch:.1f} ms, onnx fp32 {t_fp32:.1f} ms, onnx int8 {t_int8:.1f} ms "
          f"per headline  (int8 {t_torch / t_int8:.2f}x faster than torch)")
    check = np.sum(torch_emb * fp32_onnx, axis=1)
    print(f"pooling check — cosine(torch, onnx fp32): min {check.min():.6f}")
    agree = np.sum(torch_emb * int8, axis=1)
    print(f"cosine(torch fp32, onnx int8) per headline: mean {agree.mean():.4f}, min {agree.min():.4f}")

    pairs = [json.loads(l) for l in (HERE / "labels" / "pairs_labeled.jsonl").read_text(encoding="utf-8").splitlines()]
    a = np.array([p["a"] for p in pairs]); b = np.array([p["b"] for p in pairs])
    labels = np.array([p["label"] for p in pairs])
    s32 = np.sum(torch_emb[a] * torch_emb[b], axis=1)
    s8 = np.sum(int8[a] * int8[b], axis=1)
    flips = int(np.sum((s32 >= JOIN) != (s8 >= JOIN)))
    near = int(np.sum(np.abs(s32 - JOIN) < 0.02))
    print(f"pair similarity drift: mean |Δ| {np.abs(s32 - s8).mean():.4f}, max |Δ| {np.abs(s32 - s8).max():.4f}")
    print(f"join decisions at {JOIN} that flip: {flips} of {len(pairs)} (pairs within ±0.02 of the threshold: {near})")
    print(f"strict AUC: torch fp32 {_auc(s32, labels):.4f}   onnx int8 {_auc(s8, labels):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
