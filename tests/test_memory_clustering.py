"""
tests/test_memory_clustering.py
═══════════════════════════════
Unit tests for enrichment/memory_clustering.py (no database).

Decision-for-decision parity with the SQL wizer_assign_cluster is tested in
tests/test_clustering_sql.py::TestMemoryParity (needs WIZER_TEST_DSN).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from enrichment import memory_clustering as mc

T0 = datetime(2026, 10, 3, 6, 0, tzinfo=timezone.utc)


def unit(seed):
    v = np.random.default_rng(seed).normal(size=768)
    return v / np.linalg.norm(v)


def at_cos(base, cos, seed):
    r = unit(seed)
    orth = r - (r @ base) * base
    orth /= np.linalg.norm(orth)
    return cos * base + math.sqrt(1 - cos * cos) * orth


P = mc._Params(join=0.85, anchor=0.75, gap_s=18 * 3600, span_s=120 * 3600, candidates=5)


def add(st, aid, v, *, t=T0, domain="a.com", title="t", lang="en"):
    return st.assign(aid, v, t, domain, title, lang, P)


class TestDecision:

    def test_first_article_seeds(self):
        st = mc.ClusterState(model="m")
        a = add(st, 1, unit(1))
        assert a.action == "seed" and a.article_count == 1 and len(st) == 1

    def test_similar_joins_dissimilar_seeds(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        s = add(st, 1, base)
        j = add(st, 2, at_cos(base, 0.95, 2), domain="b.com")
        n = add(st, 3, unit(3))
        assert j.action == "join" and j.cluster_id == s.cluster_id and j.outlet_count == 2
        assert n.action == "seed" and n.cluster_id != s.cluster_id

    def test_similarity_is_average_link(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        add(st, 1, base)
        m2 = at_cos(base, 0.95, 2)
        add(st, 2, m2)
        q = at_cos(base, 0.93, 3)
        a = add(st, 3, q)
        assert a.similarity == pytest.approx((q @ base + q @ m2) / 2, abs=1e-5)

    def test_anchor_guard(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        add(st, 1, base)
        # Pull the centroid away from the anchor with members at 0.9 to a drift point.
        drift = at_cos(base, 0.8, 9)
        for i in range(2, 12):
            add(st, i, (base + drift) / np.linalg.norm(base + drift))
        far = at_cos(drift, 0.97, 10)          # close to the centroid, far from the anchor
        assert float(far @ base) < 0.75 + 0.05
        a = add(st, 99, far)
        anchor_sim = float(mc._cos16(st.anchor16[0], far.astype(np.float16)))
        assert (a.action == "seed") == (anchor_sim < 0.75)

    def test_gap_window(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        add(st, 1, base)
        late = add(st, 2, at_cos(base, 0.95, 2), t=T0 + timedelta(hours=19))
        assert late.action == "seed"

    def test_span_cap(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        for h in range(0, 121, 12):
            add(st, h + 1, at_cos(base, 0.97, h + 100), t=T0 + timedelta(hours=h))
        a = add(st, 999, at_cos(base, 0.97, 999), t=T0 + timedelta(hours=130))
        assert a.action == "seed"

    def test_representative_moves_to_most_central(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        add(st, 1, at_cos(base, 0.9, 11), title="first")
        add(st, 2, at_cos(base, 0.9, 12), title="second")
        add(st, 3, base, title="central")             # exactly the story's direction
        assert st.rep_ids[0] == 3 and st.headlines[0] == "central"

    @pytest.mark.parametrize("bad", [np.zeros(768), np.full(768, np.nan)])
    def test_rejects_zero_and_nan(self, bad):
        with pytest.raises(ValueError):
            add(mc.ClusterState(model="m"), 1, bad)

    def test_rejects_wrong_dimension(self):
        with pytest.raises(ValueError):
            add(mc.ClusterState(model="m"), 1, np.ones(10))

    def test_domain_and_language_normalised(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        add(st, 1, base, domain=" A.com ", lang="EN")
        add(st, 2, at_cos(base, 0.95, 2), domain="a.com", lang=" en")
        add(st, 3, at_cos(base, 0.95, 3), domain="", lang=None)
        assert st.outlets[0] == ["a.com", "(unknown)"] and st.languages[0] == ["en"]


class TestBookkeeping:

    def test_pending_changes_and_flush(self):
        st = mc.ClusterState(model="m")
        base = unit(1)
        add(st, 1, base)
        add(st, 2, at_cos(base, 0.95, 2))
        clusters, articles = st.pending_changes()
        assert len(clusters) == 1 and clusters[0]["is_new"] and clusters[0]["article_count"] == 2
        assert "anchor" in clusters[0] and "representative" in clusters[0]   # 2 members: not derivable
        assert [a["action"] for a in articles] == ["seed", "join"]
        st.mark_flushed("2026-10-05T00:00:00+00:00")
        add(st, 3, base)                                    # moves the representative
        clusters, _ = st.pending_changes()
        assert not clusters[0]["is_new"] and "representative" in clusters[0]

    def test_vector_text_round_trips_float32(self):
        v = unit(5).astype(np.float32)
        assert np.array_equal(np.asarray(mc._parse_vec(mc._fmt_vec(v)), np.float32), v)

    def test_save_load_round_trip(self, tmp_path):
        st = mc.ClusterState(model="m")
        for i in range(5):
            add(st, i, unit(i))
        st.mark_flushed("2026-10-05T00:00:00+00:00")
        st.save(tmp_path / "s.npz")
        back = mc.ClusterState.load(tmp_path / "s.npz", model="m")
        assert back is not None and len(back) == 5 and back.synced_at == st.synced_at
        assert np.array_equal(back.sums[:5], st.sums[:5]) and back.ids == st.ids
        q = at_cos(unit(3), 0.95, 50)
        assert add(back, 50, q).cluster_id == st.ids[3]

    def test_load_rejects_other_model_and_missing_file(self, tmp_path):
        st = mc.ClusterState(model="m")
        st.save(tmp_path / "s.npz")
        assert mc.ClusterState.load(tmp_path / "s.npz", model="other") is None
        assert mc.ClusterState.load(tmp_path / "nope.npz", model="m") is None

    def test_save_refuses_unflushed_changes(self, tmp_path):
        st = mc.ClusterState(model="m")
        add(st, 1, unit(1))
        with pytest.raises(RuntimeError):
            st.save(tmp_path / "s.npz")

    def test_remove_and_prune_keep_index_consistent(self):
        st = mc.ClusterState(model="m")
        for i in range(6):
            add(st, i, unit(i), t=T0 + timedelta(hours=i))
        st.mark_flushed(None)
        ids = list(st.ids)
        st.remove(ids[1])
        assert len(st) == 5 and all(st._index[c] == i for i, c in enumerate(st.ids))
        dropped = st.prune((T0 + timedelta(hours=3)).timestamp())
        assert dropped == 2 and set(st.ids) == set(ids[3:])
        assert all(st._index[c] == i for i, c in enumerate(st.ids))

    def test_upsert_row_tombstone_removes(self):
        st = mc.ClusterState(model="m")
        add(st, 1, unit(1))
        st.mark_flushed(None)
        st.upsert_row({"id": st.ids[0], "status": "merged", "embedding_model": "m"})
        assert len(st) == 0

    def test_gray_zone_config_refused(self, monkeypatch):
        monkeypatch.setattr(mc, "CLUSTER_GRAY_THRESHOLD", 0.7)
        monkeypatch.setattr(mc, "CLUSTER_JOIN_THRESHOLD", 0.85)
        with pytest.raises(RuntimeError):
            mc.check_supported_config()
