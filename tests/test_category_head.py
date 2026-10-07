"""
tests/test_category_head.py — the E5 linear category head and its fallback to mDeBERTa.

Uses a tiny synthetic model file, so it runs offline without the real model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from enrichment import runner
from enrichment.steps import category_head

MODEL_ID = "test/e5"


@pytest.fixture
def head(tmp_path, monkeypatch):
    """A 3-class head: class i wins when the vector points along axis i."""
    path = tmp_path / "category_head.npz"
    W = np.zeros((3, 768), dtype=np.float32)
    for i in range(3):
        W[i, i] = 10.0
    np.savez_compressed(path, W=W, b=np.zeros(3, dtype=np.float32),
                        classes=np.array(["politics", "cricket", "business"]),
                        meta=json.dumps({"embedding_model": MODEL_ID, "trained": "2026-10-06"}))
    monkeypatch.setattr(category_head, "MODEL_PATH", path)
    monkeypatch.setattr(category_head, "_head", None)
    return path


def _axis(i: int) -> list[float]:
    v = [0.0] * 768
    v[i] = 1.0
    return v


class TestPredict:
    def test_argmax_class_and_probability(self, head):
        cat, p = category_head.predict(_axis(1), MODEL_ID)
        assert cat == "cricket"
        assert 0.99 < p <= 1.0

    def test_other_embedding_model_is_refused(self, head):
        assert category_head.predict(_axis(0), "other/model") is None
        assert not category_head.available("other/model")

    def test_none_vector(self, head):
        assert category_head.predict(None, MODEL_ID) is None

    def test_missing_model_file_means_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(category_head, "MODEL_PATH", tmp_path / "absent.npz")
        monkeypatch.setattr(category_head, "_head", None)
        assert not category_head.available(MODEL_ID)
        assert category_head.predict(_axis(0), MODEL_ID) is None


class TestRunnerCategory:
    @pytest.fixture(autouse=True)
    def _env(self, head, monkeypatch):
        monkeypatch.setattr(runner, "CATEGORY_HEAD", True)
        monkeypatch.setattr(runner, "CLUSTER_EMBEDDING_MODEL", MODEL_ID)
        monkeypatch.setattr(runner, "classify_article", lambda *a: "zero-shot")

    def test_uses_given_vector_without_embedding(self, monkeypatch):
        monkeypatch.setattr(runner, "embed_texts", lambda t: pytest.fail("must not embed"))
        art = {"_category_vector": _axis(2)}
        assert runner._category(art, "t", "d", "") == "business"

    def test_embeds_when_no_vector(self, monkeypatch):
        seen = []
        monkeypatch.setattr(runner, "embed_texts", lambda t: seen.extend(t) or [_axis(0)])
        assert runner._category({}, "Election results", "", "") == "politics"
        assert seen == ["Election results"]

    def test_falls_back_when_embedding_fails(self, monkeypatch):
        monkeypatch.setattr(runner, "embed_texts", lambda t: [None])
        assert runner._category({}, "t", "", "") == "zero-shot"

    def test_falls_back_when_head_off(self, monkeypatch):
        monkeypatch.setattr(runner, "CATEGORY_HEAD", False)
        assert runner._category({"_category_vector": _axis(1)}, "t", "", "") == "zero-shot"

    def test_falls_back_on_model_mismatch(self, monkeypatch):
        monkeypatch.setattr(runner, "CLUSTER_EMBEDDING_MODEL", "other/model")
        assert runner._category({"_category_vector": _axis(1)}, "t", "", "") == "zero-shot"

    def test_enrich_one_sets_category_from_head(self, monkeypatch):
        for name, val in [("detect_language", lambda *a: "ta"), ("classify_tags", lambda *a: []),
                          ("summarize_article", lambda *a: None)]:
            monkeypatch.setattr(runner, name, val)
        update, _ = runner.enrich_one({"id": 1, "title": "x", "_category_vector": _axis(1),
                                       "_image_phash": None})
        assert update["category"] == "cricket"
