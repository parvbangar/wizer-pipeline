"""
tests/test_clustering.py
════════════════════════
Unit tests for the Python half of story clustering — fully offline:

  enrichment/steps/embedding.py     text construction, encoding contract
  enrichment/clustering.py          evidence selection, RPC parameter contract
  enrichment/cluster_maintenance.py merge planning and orchestration

The SQL half is covered by tests/test_clustering_sql.py (real Postgres).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from enrichment import clustering
from enrichment import cluster_maintenance as cm
from enrichment.steps import embedding

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# build_embedding_text
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildEmbeddingText:

    def test_title_plus_description(self):
        out = embedding.build_embedding_text(
            "RBI hikes repo rate", "The Reserve Bank raised the repo rate by 25 bps on Friday citing inflation.", "")
        assert out == "RBI hikes repo rate. The Reserve Bank raised the repo rate by 25 bps on Friday citing inflation."

    def test_no_double_punctuation(self):
        out = embedding.build_embedding_text("Is the rally over?", "Markets fell for the fifth session in a row on Friday.", "")
        assert out.startswith("Is the rally over? Markets")

    def test_description_repeating_title_is_replaced_by_body_lead(self):
        title = "Sensex jumps 500 points"
        out = embedding.build_embedding_text(
            title, "Sensex jumps 500 points",
            "Benchmark indices rallied on Monday. Banks led the gains. Analysts expect more volatility.")
        assert out == "Sensex jumps 500 points. Benchmark indices rallied on Monday. Banks led the gains. " \
                      "Analysts expect more volatility."

    def test_short_description_falls_back_to_lead(self):
        out = embedding.build_embedding_text("Flood alert in Assam", "Read more", "Heavy rain lashed Guwahati overnight.")
        assert out == "Flood alert in Assam. Heavy rain lashed Guwahati overnight."

    def test_html_and_entities_cleaned(self):
        out = embedding.build_embedding_text(
            "Tata &amp; Sons <b>results</b>", "<p>Profit rose&nbsp;12% to ₹4,000 crore in the September quarter.</p>", "")
        assert "<" not in out and "&amp;" not in out and "Tata & Sons results." in out

    def test_devanagari_sentence_split(self):
        out = embedding.build_embedding_text("दिल्ली में बारिश", "", "दिल्ली में तेज बारिश हुई। यातायात प्रभावित रहा।")
        assert out == "दिल्ली में बारिश. दिल्ली में तेज बारिश हुई। यातायात प्रभावित रहा।"

    def test_capped_on_word_boundary(self):
        out = embedding.build_embedding_text("Headline", "word " * 200, "", max_chars=100)
        assert len(out) <= 100 and not out.endswith("wor")

    def test_empty_inputs(self):
        assert embedding.build_embedding_text(None, None, None) == ""
        assert embedding.build_embedding_text("", "", "Only body text here.") == "Only body text here."


# ─────────────────────────────────────────────────────────────────────────────
# embed_texts — model stubbed
# ─────────────────────────────────────────────────────────────────────────────

class _FakeModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.seen = None

    def encode(self, texts, **kw):
        import numpy as np
        self.seen = texts
        assert kw["normalize_embeddings"] is True
        return np.array(self.vectors[: len(texts)], dtype=float)


class TestEmbedTexts:

    def test_order_preserved_and_empties_are_none(self, monkeypatch):
        fake = _FakeModel([[1.0] + [0.0] * 767, [0.0, 1.0] + [0.0] * 766])
        monkeypatch.setattr(embedding, "_get_model", lambda model_id: fake)
        out = embedding.embed_texts(["first", "", "  ", "second"], model_id="intfloat/multilingual-e5-base")
        assert out[1] is None and out[2] is None
        assert out[0][0] == 1.0 and out[3][1] == 1.0
        assert fake.seen == ["first", "second"]          # no prefix for E5 (calibrated)

    def test_zero_and_nan_vectors_dropped(self, monkeypatch):
        fake = _FakeModel([[0.0] * 768, [float("nan")] * 768])
        monkeypatch.setattr(embedding, "_get_model", lambda model_id: fake)
        assert embedding.embed_texts(["a", "b"], model_id="sentence-transformers/LaBSE") == [None, None]

    def test_model_failure_yields_all_none(self, monkeypatch):
        def boom(model_id):
            raise RuntimeError("OOM")
        monkeypatch.setattr(embedding, "_get_model", boom)
        assert embedding.embed_texts(["a", "b"]) == [None, None]

    def test_unknown_model_rejected(self):
        with pytest.raises(ValueError, match="Unknown CLUSTER_EMBEDDING_MODEL"):
            embedding.get_profile("some/other-model")

    def test_all_profiles_match_schema_dimension(self):
        assert all(p.dimension == embedding.EMBEDDING_DIMENSION for p in embedding.MODEL_PROFILES.values())

    def test_to_pgvector(self):
        assert embedding.to_pgvector([0.5, -0.25, 1.0]) == "[0.500000,-0.250000,1.000000]"


# ─────────────────────────────────────────────────────────────────────────────
# Evidence selection
# ─────────────────────────────────────────────────────────────────────────────

def _ent(text, typ="PERSON", sal=0.8):
    return {"entity_text": text, "entity_type": typ, "salience": sal}


class TestEvidence:

    def test_keys_are_salient_specific_and_deduped(self):
        keys = clustering.entity_keys_for_clustering([
            _ent("Smit  Machchhar", sal=0.9),
            _ent("smit machchhar", sal=0.5),          # duplicate after normalisation
            _ent("Flydubai", "ORG", 0.8),
            _ent("Mumbai", "GPE", 0.2),               # below salience floor
            _ent("India", "GPE", 0.9),                # stop-listed
            _ent("PTI", "ORG", 0.9),                  # wire agency byline
            _ent("₹500 crore", "MONEY", 0.9),         # type not used as evidence
        ])
        assert keys == ["smit machchhar", "flydubai"]

    def test_keys_ranked_by_salience_and_capped(self):
        ents = [_ent(f"Person {i}", sal=0.3 + i / 100) for i in range(30)]
        keys = clustering.entity_keys_for_clustering(ents)
        assert len(keys) == 15 and keys[0] == "person 29"

    def test_top_entities_payload(self):
        payload = clustering.top_entities_payload([_ent("A", sal=0.2), _ent("B", "ORG", 0.9)])
        assert payload == [{"text": "B", "type": "ORG"}, {"text": "A", "type": "PERSON"}]

    @pytest.mark.parametrize("phash,expected", [(None, None), (0, None), (-1, None), (12345, 12345)])
    def test_image_hash_denylist(self, phash, expected):
        assert clustering.usable_image_phash(phash) == expected


# ─────────────────────────────────────────────────────────────────────────────
# RPC parameter contract
# ─────────────────────────────────────────────────────────────────────────────

def _sql_function_params(fn: str) -> list[str]:
    sql = (ROOT / "docs" / "clustering_v2_migration.sql").read_text(encoding="utf-8")
    header = re.search(rf"CREATE OR REPLACE FUNCTION {fn}\((.*?)\)\s*RETURNS", sql, re.S).group(1)
    return re.findall(r"^\s*(p_\w+)", header, re.M)


class TestAssignParams:

    ARTICLE = {"id": 42, "title": "Headline", "domain": "ndtv.com",
               "published_at": "2026-10-03T05:00:00+00:00", "language_code": "en"}

    def test_params_match_sql_signature_exactly(self):
        """A renamed SQL parameter would make every PostgREST call fail at runtime."""
        params = clustering.build_assign_params(self.ARTICLE, [0.1] * 768, [], None, "hi")
        assert list(params) == _sql_function_params("wizer_assign_cluster")

    def test_param_values(self):
        params = clustering.build_assign_params(
            self.ARTICLE, [0.1] * 768, [_ent("Smit Machchhar")], 0, None)
        assert params["p_article_id"] == 42
        assert params["p_embedding"].startswith("[0.100000,")
        assert params["p_language"] == "en"                 # falls back to feed language
        assert params["p_image_phash"] is None              # blank-image hash dropped
        assert params["p_entity_keys"] == ["smit machchhar"]
        assert params["p_gray_threshold"] <= params["p_join_threshold"]

    def test_assign_cluster_without_embedding_is_skipped(self, monkeypatch):
        called = []
        monkeypatch.setattr(clustering.db, "assign_cluster", lambda p: called.append(p))
        assert clustering.assign_cluster(self.ARTICLE, None, [], None, "en") is None
        assert called == []

    def test_assign_cluster_maps_rpc_row(self, monkeypatch):
        monkeypatch.setattr(clustering.db, "assign_cluster", lambda p: {
            "cluster_id": "c1", "action": "join", "similarity": 0.91, "article_count": 3, "outlet_count": 2})
        res = clustering.assign_cluster(self.ARTICLE, [0.1] * 768, [], None, "en")
        assert res == clustering.ClusterResult("c1", "join", 0.91, 3, 2)

    def test_assign_cluster_rpc_failure_is_none(self, monkeypatch):
        monkeypatch.setattr(clustering.db, "assign_cluster", lambda p: None)
        assert clustering.assign_cluster(self.ARTICLE, [0.1] * 768, [], None, "en") is None


# ─────────────────────────────────────────────────────────────────────────────
# Merge planning + orchestration
# ─────────────────────────────────────────────────────────────────────────────

def _cand(a, na, b, nb, sim):
    return {"probe_id": a, "probe_count": na, "other_id": b, "other_count": nb,
            "similarity": sim, "probe_updated_at": "2026-10-03T00:00:00+00:00"}


class TestPlanMerges:

    def test_larger_cluster_wins(self):
        assert cm.plan_merges([_cand("a", 2, "b", 9, 0.9)], 0.85) == [("b", "a", 0.9)]

    def test_tie_broken_by_id(self):
        assert cm.plan_merges([_cand("z", 3, "m", 3, 0.9)], 0.85) == [("m", "z", 0.9)]

    def test_symmetric_duplicates_collapsed(self):
        plan = cm.plan_merges([_cand("a", 1, "b", 2, 0.9), _cand("b", 2, "a", 1, 0.9)], 0.85)
        assert plan == [("b", "a", 0.9)]

    def test_below_threshold_and_missing_partner_ignored(self):
        assert cm.plan_merges([_cand("a", 1, "b", 1, 0.80),
                               {"probe_id": "c", "other_id": None, "similarity": None}], 0.85) == []

    def test_each_cluster_merges_once_per_round_strongest_first(self):
        plan = cm.plan_merges([_cand("a", 5, "b", 1, 0.88), _cand("a", 5, "c", 1, 0.95),
                               _cand("d", 1, "e", 1, 0.90)], 0.85)
        assert plan == [("a", "c", 0.95), ("d", "e", 0.90)]


class _FakeBackend:
    """Two candidate pages, then nothing to merge on the second round."""

    def __init__(self, pages_per_round):
        self.rounds = list(pages_per_round)
        self.current = []
        self.calls = []
        self.merged = []

    def find_merge_candidates(self, params):
        self.calls.append(params)
        if not self.current:
            self.current = list(self.rounds.pop(0)) if self.rounds else [[]]
        return self.current.pop(0)

    def merge_clusters(self, w, l):
        self.merged.append((w, l))
        return True

    def reconcile_cluster_counts(self, since):
        return 0

    def prune_orphan_clusters(self, hours):
        return 4


class TestMaintenanceOrchestration:

    def test_pages_are_followed_with_keyset(self, monkeypatch):
        monkeypatch.setattr(cm, "_PROBE_PAGE", 2)
        page1 = [_cand("a", 1, "b", 3, 0.9), _cand("c", 1, None, None, None)]
        page1[1]["probe_updated_at"] = "2026-10-03T01:00:00+00:00"
        backend = _FakeBackend([[page1, [_cand("d", 2, "e", 1, 0.95)]], [[]]])
        rep = cm.merge_duplicates(backend, datetime(2026, 10, 3, tzinfo=timezone.utc), threshold=0.85)
        assert backend.calls[1]["p_after_id"] == "c"
        assert backend.calls[1]["p_after_ts"] == "2026-10-03T01:00:00+00:00"
        assert sorted(backend.merged) == [("b", "a"), ("d", "e")]
        assert rep.merges == 2 and rep.merge_rounds == 2

    def test_dry_run_plans_but_does_not_merge(self):
        backend = _FakeBackend([[[_cand("a", 1, "b", 3, 0.9)]]])
        rep = cm.merge_duplicates(backend, datetime(2026, 10, 3, tzinfo=timezone.utc), threshold=0.85, dry_run=True)
        assert backend.merged == [] and rep.planned == [("b", "a", 0.9)]

    def test_run_maintenance_reports_all_phases(self):
        backend = _FakeBackend([[[]]])
        rep = cm.run_maintenance(backend, lookback_hours=6)
        assert (rep.merges, rep.reconciled, rep.pruned) == (0, 0, 4)

    def test_failed_merge_is_logged_not_raised(self):
        class Flaky(_FakeBackend):
            def merge_clusters(self, w, l):
                raise RuntimeError("timeout")
        backend = Flaky([[[_cand("a", 1, "b", 3, 0.9)]]])
        rep = cm.merge_duplicates(backend, datetime(2026, 10, 3, tzinfo=timezone.utc), threshold=0.85)
        assert rep.merges == 0



# ─────────────────────────────────────────────────────────────────────────────
# Batched assignment (Python side)
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignBatch:

    def _art(self, i):
        return {"id": i, "title": f"t{i}", "domain": "x.com", "published_at": "2026-10-03T00:00:00+00:00",
                "language_code": "en"}

    def test_items_and_shared_params_cover_sql_signature(self):
        sql_batch = _sql_function_params("wizer_assign_cluster_batch")
        assert set(clustering._SHARED_KEYS) <= set(sql_batch)
        assert set(clustering._ITEM_KEYS) | set(clustering._SHARED_KEYS) ==             set(_sql_function_params("wizer_assign_cluster"))

    def test_chunks_and_maps_results(self, monkeypatch):
        calls = []

        def fake(items, shared):
            calls.append((len(items), shared["p_model"]))
            return [{"article_id": it["p_article_id"], "cluster_id": "c", "action": "join",
                     "similarity": 0.9, "article_count": 2, "outlet_count": 2, "error": None}
                    for it in items]

        monkeypatch.setattr(clustering.db, "assign_cluster_batch", fake)
        monkeypatch.setattr(clustering, "BATCH_SIZE", 2)
        items = [(self._art(i), [0.1] * 768, [], None, "en") for i in range(5)]
        items.append((self._art(99), None, [], None, "en"))               # no embedding
        out = clustering.assign_clusters_batch(items)
        assert [n for n, _ in calls] == [2, 2, 1]
        assert out[99] is None and out[0].action == "join" and len(out) == 6

    def test_item_errors_and_failed_calls_are_none(self, monkeypatch):
        monkeypatch.setattr(clustering.db, "assign_cluster_batch", lambda items, shared: [
            {"article_id": 1, "cluster_id": None, "action": None, "error": "zero or NaN embedding"}])
        assert clustering.assign_clusters_batch([(self._art(1), [0.1] * 768, [], None, "en")]) == {1: None}
        monkeypatch.setattr(clustering.db, "assign_cluster_batch", lambda items, shared: None)
        assert clustering.assign_clusters_batch([(self._art(2), [0.1] * 768, [], None, "en")]) == {2: None}
