"""
tests/test_clustering_sql.py
════════════════════════════
Integration tests for the SQL half of the pipeline — the clustering and work-
queue functions in docs/*.sql — against a REAL PostgreSQL + pgvector.

These cannot be meaningfully mocked: the behaviour under test (atomicity under
concurrency, exact vector maths, NaN ordering, SKIP LOCKED, idempotent
migrations) lives in the database.

HOW TO RUN:
  CI:     tests.yml starts a pgvector/pgvector:pg17 service and sets WIZER_TEST_DSN.
  Local:  any Postgres ≥ 14 with pgvector ≥ 0.7, e.g.
            docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
            export WIZER_TEST_DSN="host=localhost port=5432 user=postgres password=postgres"
            pytest tests/test_clustering_sql.py -v
  Without WIZER_TEST_DSN the whole module is skipped (unit tests still run).

Each test gets a freshly TRUNCATED schema; the migrations are applied once per
session into a dedicated database (wizer_test), which is dropped afterwards.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest

DSN = os.getenv("WIZER_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="WIZER_TEST_DSN not set — SQL integration tests skipped")

if DSN:
    import numpy as np
    import psycopg
    from psycopg.rows import dict_row

    from enrichment.cluster_maintenance import merge_duplicates, run_maintenance
    from tools.db_migrations import apply_all
    from tools.pg_backend import PgBackend

TEST_DB = "wizer_test"
MODEL = "test-model"
T0 = datetime(2026, 10, 3, 6, 0, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def database():
    admin = psycopg.connect(f"{DSN} dbname=postgres", autocommit=True)
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {TEST_DB}")
    conn = psycopg.connect(f"{DSN} dbname={TEST_DB}")
    apply_all(conn)
    conn.close()
    yield f"{DSN} dbname={TEST_DB}"
    admin.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
    admin.close()


@pytest.fixture
def conn(database):
    c = psycopg.connect(database)
    c.execute("TRUNCATE articles, article_clusters, article_entities, enrichment_runs, article_archive_log RESTART IDENTITY CASCADE")
    c.commit()
    yield c
    c.close()


@pytest.fixture
def backend(conn):
    return PgBackend(conn)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_rng = np.random.default_rng(1234) if DSN else None


def unit(seed: int | None = None) -> "np.ndarray":
    rng = np.random.default_rng(seed) if seed is not None else _rng
    v = rng.normal(size=768)
    return v / np.linalg.norm(v)


def at_cos(base: "np.ndarray", cos: float, seed: int | None = None) -> "np.ndarray":
    """A unit vector with EXACT cosine `cos` to `base`."""
    r = unit(seed)
    orth = r - (r @ base) * base
    orth /= np.linalg.norm(orth)
    return cos * base + math.sqrt(1 - cos * cos) * orth


def vec(v) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def add_article(conn, *, domain="a.com", title="t", published=T0, crawled=True,
                enriched=False, lang="en", ingested=None) -> int:
    row = conn.execute(
        "INSERT INTO articles (url, url_hash, title, published_at, domain, is_crawled, "
        "language_code, enriched_at, crawled_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, coalesce(%s, now())) RETURNING id",
        (f"https://{domain}/{uuid.uuid4()}", uuid.uuid4().int >> 65, title, published, domain,
         crawled, lang, T0 if enriched else None, ingested),
    ).fetchone()
    conn.commit()
    return row[0]


def assign(backend, article_id, v, *, published=T0, domain="a.com", title="t", model=MODEL,
           entities=None, keys=None, phash=None, join=0.85, gray=None, anchor=0.75,
           gap=18, span=120, lang="en") -> dict:
    return backend.assign_cluster({
        "p_article_id": article_id, "p_embedding": vec(v), "p_model": model,
        "p_published_at": published.isoformat(), "p_domain": domain, "p_title": title,
        "p_language": lang, "p_entities": entities or [], "p_entity_keys": keys or [],
        "p_image_phash": phash, "p_join_threshold": join,
        "p_gray_threshold": join if gray is None else gray, "p_anchor_threshold": anchor,
        "p_min_shared_entities": 2, "p_image_max_distance": 6,
        "p_max_gap_hours": gap, "p_max_span_hours": span, "p_candidates": 5,
    })


def cluster(conn, cid) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute("SELECT * FROM article_clusters WHERE id = %s", (cid,)).fetchone()


def new_and_assign(conn, backend, v, **kw) -> dict:
    aid = add_article(conn, domain=kw.get("domain", "a.com"), title=kw.get("title", "t"),
                      published=kw.get("published", T0))
    res = assign(backend, aid, v, **kw)
    res["article_id"] = aid
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Migrations
# ─────────────────────────────────────────────────────────────────────────────

class TestMigrations:

    def test_chain_is_idempotent(self, database):
        """Re-running every migration on a populated database must succeed."""
        c = psycopg.connect(database)
        add_article(c)
        apply_all(c)
        assert c.execute("SELECT count(*) FROM articles").fetchone()[0] == 1
        c.close()

    def test_service_role_only(self, conn):
        """anon/authenticated must not be able to call the queue or cluster RPCs."""
        for fn in ("wizer_assign_cluster", "wizer_claim_enrichment_batch", "wizer_merge_clusters"):
            assert conn.execute(
                "SELECT bool_or(has_function_privilege('anon', p.oid, 'EXECUTE')) "
                "FROM pg_proc p WHERE proname = %s", (fn,)
            ).fetchone()[0] is False
            assert conn.execute(
                "SELECT bool_or(has_function_privilege('service_role', p.oid, 'EXECUTE')) "
                "FROM pg_proc p WHERE proname = %s", (fn,)
            ).fetchone()[0] is True

    @pytest.mark.parametrize("table", ["enrichment_runs", "article_archive_log"])
    def test_new_tables_granted_explicitly(self, conn, table):
        """Production's default privileges give service_role no SELECT/INSERT/UPDATE."""
        for priv in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            assert conn.execute("SELECT has_table_privilege('service_role', %s, %s)", (table, priv)).fetchone()[0]
        for priv in ("SELECT", "DELETE", "TRUNCATE"):
            assert not conn.execute("SELECT has_table_privilege('anon', %s, %s)", (table, priv)).fetchone()[0]

    @pytest.mark.parametrize("view", ["cluster_health", "top_stories_24h", "enrichment_queue_health"])
    def test_views_granted_to_service_role_only(self, conn, view):
        assert conn.execute("SELECT has_table_privilege('service_role', %s, 'SELECT')", (view,)).fetchone()[0]
        assert not conn.execute("SELECT has_table_privilege('anon', %s, 'SELECT')", (view,)).fetchone()[0]

    def test_monitoring_views_query(self, conn):
        for view in ("cluster_health", "top_stories_24h", "enrichment_queue_health"):
            conn.execute(f"SELECT * FROM {view}").fetchall()


# ─────────────────────────────────────────────────────────────────────────────
# jsonb / hash helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_merge_top_entities_sums_case_insensitively(self, conn):
        out = conn.execute(
            "SELECT wizer_merge_top_entities(%s::jsonb, %s::jsonb, 30)",
            ('[{"text":"Modi","type":"PERSON","count":2}]',
             '[{"text":"MODI","type":"PERSON"},{"text":"BJP","type":"ORG"}]'),
        ).fetchone()[0]
        assert out[0] == {"text": "Modi", "type": "PERSON", "count": 3}
        assert {"text": "BJP", "type": "ORG", "count": 1} in out

    def test_merge_top_entities_limit_and_blank_text(self, conn):
        out = conn.execute(
            "SELECT wizer_merge_top_entities('[]', %s::jsonb, 2)",
            ('[{"text":"a"},{"text":"b"},{"text":"c"},{"text":""}]',),
        ).fetchone()[0]
        assert len(out) == 2

    def test_jsonb_text_union_keeps_first_order(self, conn):
        out = conn.execute("SELECT wizer_jsonb_text_union('[\"b\",\"a\"]', '[\"a\",\"c\"]', 10)").fetchone()[0]
        assert out == ["b", "a", "c"]

    def test_image_hash_merge_keeps_newest(self, conn):
        out = conn.execute(
            "SELECT wizer_merge_image_hashes(%s::jsonb, %s::jsonb, 2)",
            ('[{"h":1,"d":"x"},{"h":2,"d":"x"}]', '[{"h":3,"d":"y"}]'),
        ).fetchone()[0]
        assert [e["h"] for e in out] == [2, 3]

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (0, 1, 1), (-1, 0, 64), (-(2 ** 63), 0, 1), (-(2 ** 63), 1, 2), (5, 6, 2),
    ])
    def test_phash_distance_handles_signed_values(self, conn, a, b, expected):
        assert conn.execute("SELECT wizer_phash_distance(%s, %s)", (a, b)).fetchone()[0] == expected


# ─────────────────────────────────────────────────────────────────────────────
# wizer_assign_cluster
# ─────────────────────────────────────────────────────────────────────────────

class TestAssign:

    def test_first_article_seeds_cluster(self, conn, backend):
        r = new_and_assign(conn, backend, unit(1), title="Budget announced")
        assert r["action"] == "seed" and r["article_count"] == 1
        c = cluster(conn, r["cluster_id"])
        assert c["headline"] == "Budget announced"
        assert c["canonical_article_id"] == r["article_id"] == c["representative_article_id"]
        art = conn.execute("SELECT cluster_id, cluster_assignment FROM articles WHERE id = %s",
                           (r["article_id"],)).fetchone()
        assert str(art[0]) == str(r["cluster_id"]) and art[1] == "seed"

    def test_similar_article_joins(self, conn, backend):
        base = unit(1)
        a = new_and_assign(conn, backend, base, domain="a.com")
        b = new_and_assign(conn, backend, at_cos(base, 0.95), domain="b.com")
        assert b["action"] == "join" and b["cluster_id"] == a["cluster_id"]
        assert b["article_count"] == 2 and b["outlet_count"] == 2
        assert b["similarity"] == pytest.approx(0.95, abs=1e-3)

    def test_dissimilar_article_seeds_new_cluster(self, conn, backend):
        base = unit(1)
        a = new_and_assign(conn, backend, base)
        b = new_and_assign(conn, backend, at_cos(base, 0.6))
        assert b["action"] == "seed" and b["cluster_id"] != a["cluster_id"]

    def test_same_outlet_counted_once(self, conn, backend):
        base = unit(1)
        new_and_assign(conn, backend, base, domain="ndtv.com")
        r = new_and_assign(conn, backend, at_cos(base, 0.97), domain="NDTV.com")
        assert r["article_count"] == 2 and r["outlet_count"] == 1

    def test_idempotent_on_retry(self, conn, backend):
        base = unit(1)
        a = new_and_assign(conn, backend, base)
        again = assign(backend, a["article_id"], base)
        assert again["action"] == "existing" and again["cluster_id"] == a["cluster_id"]
        assert cluster(conn, a["cluster_id"])["article_count"] == 1

    def test_similarity_is_exact_average_link(self, conn, backend):
        """Returned similarity = mean cosine between the article and every member."""
        base = unit(1)
        members = [base, at_cos(base, 0.97, seed=2), at_cos(base, 0.96, seed=3)]
        for m in members:
            new_and_assign(conn, backend, m)
        q = at_cos(base, 0.95, seed=4)
        r = new_and_assign(conn, backend, q)
        expected = float(np.mean([q @ m for m in members]))
        assert r["action"] == "join"
        assert r["similarity"] == pytest.approx(expected, abs=2e-4)

    def test_average_link_resists_chaining(self, conn, backend):
        """
        A chain a→b→c where each step is similar but a and c are not:
        c is close to the CENTROID of {a, b} but its average similarity to
        the members is below the threshold, so it must not join.
        """
        # Orthonormal basis e1, e2, e3; a = e1, b at cosine 0.86 from a.
        e1, e2, e3 = (np.eye(768)[i] for i in range(3))
        s = math.sqrt(1 - 0.86 ** 2)
        a = e1
        b = 0.86 * e1 + s * e2
        # c: cos(c, a) = 0.80, cos(c, b) = 0.86  →  average 0.83 (< 0.85)
        y = (0.86 - 0.86 * 0.80) / s
        c = 0.80 * e1 + y * e2 + math.sqrt(1 - 0.80 ** 2 - y ** 2) * e3
        centroid = (a + b) / np.linalg.norm(a + b)
        assert c @ centroid > 0.85          # a CENTROID rule would accept c …
        assert (c @ a + c @ b) / 2 < 0.85   # … average-link must not
        new_and_assign(conn, backend, a)
        assert new_and_assign(conn, backend, b)["action"] == "join"
        r_c = new_and_assign(conn, backend, c, anchor=0.0)
        assert r_c["action"] == "seed"

    def test_anchor_guard_blocks_drift(self, conn, backend):
        base = unit(1)
        new_and_assign(conn, backend, base)
        r = new_and_assign(conn, backend, at_cos(base, 0.9), anchor=0.95)
        assert r["action"] == "seed"

    def test_different_model_never_matches(self, conn, backend):
        base = unit(1)
        a = new_and_assign(conn, backend, base, model="model-a")
        b = new_and_assign(conn, backend, base, model="model-b")
        assert b["action"] == "seed" and b["cluster_id"] != a["cluster_id"]

    def test_gap_window(self, conn, backend):
        base = unit(1)
        new_and_assign(conn, backend, base, published=T0)
        late = new_and_assign(conn, backend, at_cos(base, 0.97), published=T0 + timedelta(hours=19), gap=18)
        assert late["action"] == "seed"
        ok = new_and_assign(conn, backend, at_cos(base, 0.97, seed=9), published=T0 + timedelta(hours=17), gap=18)
        assert ok["action"] == "join"

    def test_span_cap(self, conn, backend):
        base = unit(1)
        first = new_and_assign(conn, backend, base, published=T0)
        for h in (12, 24):     # keep the story alive within the 18 h gap
            new_and_assign(conn, backend, at_cos(base, 0.97, seed=h), published=T0 + timedelta(hours=h))
        beyond = new_and_assign(conn, backend, at_cos(base, 0.97, seed=99),
                                published=T0 + timedelta(hours=36), span=30)
        assert beyond["cluster_id"] != first["cluster_id"]

    def test_out_of_order_arrival_keeps_true_time_span(self, conn, backend):
        """Newest-first processing must widen, never shrink, the cluster's span."""
        base = unit(1)
        new_and_assign(conn, backend, base, published=T0 + timedelta(hours=5))
        r = new_and_assign(conn, backend, at_cos(base, 0.97), published=T0)
        c = cluster(conn, r["cluster_id"])
        assert c["first_seen_at"] == T0 and c["last_seen_at"] == T0 + timedelta(hours=5)

    def test_gray_zone_joins_with_shared_entities(self, conn, backend):
        base = unit(1)
        ents = [{"text": "Smit Machchhar", "type": "PERSON"}, {"text": "flydubai", "type": "ORG"}]
        new_and_assign(conn, backend, base, entities=ents)
        no_evidence = new_and_assign(conn, backend, at_cos(base, 0.83, seed=2), gray=0.80)
        assert no_evidence["action"] == "seed"
        r = new_and_assign(conn, backend, at_cos(base, 0.83, seed=3), gray=0.80,
                           keys=["smit machchhar", "flydubai"])
        assert r["action"] == "gray_join"
        assert cluster(conn, r["cluster_id"])["gray_join_count"] == 1

    def test_gray_zone_image_evidence_needs_another_outlet(self, conn, backend):
        base = unit(1)
        new_and_assign(conn, backend, base, domain="a.com", phash=0x0F0F0F0F)
        same_outlet = new_and_assign(conn, backend, at_cos(base, 0.83, seed=2), domain="a.com",
                                     phash=0x0F0F0F0F, gray=0.80)
        assert same_outlet["action"] == "seed"
        other_outlet = new_and_assign(conn, backend, at_cos(base, 0.83, seed=3), domain="b.com",
                                      phash=0x0F0F0F0E, gray=0.80)
        assert other_outlet["action"] == "gray_join"

    def test_gray_zone_off_when_thresholds_equal(self, conn, backend):
        base = unit(1)
        new_and_assign(conn, backend, base, entities=[{"text": "X", "type": "ORG"}, {"text": "Y", "type": "ORG"}])
        r = new_and_assign(conn, backend, at_cos(base, 0.83, seed=2), keys=["x", "y"])  # gray = join
        assert r["action"] == "seed"

    def test_representative_is_most_central_member(self, conn, backend):
        base = unit(1)
        seed = new_and_assign(conn, backend, at_cos(base, 0.93, seed=2), title="seed headline")
        new_and_assign(conn, backend, at_cos(base, 0.93, seed=3), title="other")
        central = new_and_assign(conn, backend, base, title="the central one")
        c = cluster(conn, seed["cluster_id"])
        assert c["representative_article_id"] == central["article_id"]
        assert c["headline"] == "the central one"
        assert c["canonical_article_id"] == seed["article_id"]   # seed never changes

    def test_centroid_sum_is_exact(self, conn, backend):
        base = unit(1)
        vs = [base, at_cos(base, 0.95, seed=2), at_cos(base, 0.9, seed=3)]
        r = None
        for v in vs:
            r = new_and_assign(conn, backend, v)
        stored = np.array(conn.execute(
            "SELECT centroid_sum::text FROM article_clusters WHERE id = %s", (r["cluster_id"],)
        ).fetchone()[0].strip("[]").split(","), dtype=float)
        assert np.allclose(stored, np.sum(vs, axis=0), atol=1e-5)

    def test_entities_and_languages_accumulate(self, conn, backend):
        base = unit(1)
        new_and_assign(conn, backend, base, entities=[{"text": "Modi", "type": "PERSON"}], lang="en")
        r = new_and_assign(conn, backend, at_cos(base, 0.97),
                           entities=[{"text": "modi", "type": "PERSON"}], lang="hi")
        c = cluster(conn, r["cluster_id"])
        assert c["top_entities"][0]["count"] == 2
        assert sorted(c["language_set"]) == ["en", "hi"]

    @pytest.mark.parametrize("bad", [[0.0] * 768, [float("nan")] + [0.1] * 767])
    def test_rejects_zero_and_nan_embeddings(self, conn, backend, bad):
        """NaN sorts above every number in Postgres — must never reach the join test."""
        new_and_assign(conn, backend, unit(1))
        aid = add_article(conn)
        # pgvector itself rejects NaN on input; our guard catches the zero vector.
        with pytest.raises(psycopg.Error, match="zero or NaN|NaN not allowed"):
            assign(backend, aid, bad)
        conn.rollback()

    def test_rejects_wrong_dimension(self, conn, backend):
        aid = add_article(conn)
        with pytest.raises(psycopg.Error, match="dimensions"):
            backend.assign_cluster({"p_article_id": aid, "p_embedding": "[1,2,3]", "p_model": MODEL,
                                    "p_published_at": T0.isoformat(), "p_domain": "a", "p_title": "t",
                                    "p_language": "en"})
        conn.rollback()

    def test_missing_article_raises(self, conn, backend):
        with pytest.raises(psycopg.Error, match="does not exist"):
            assign(backend, 999_999, unit(1))
        conn.rollback()

    def test_concurrent_runners_create_one_cluster(self, database):
        """
        Eight runners assign near-duplicate articles at the same instant.
        Without the advisory lock each would see "no cluster yet" and seed its
        own; with it they must all end up in ONE cluster with exact counts.
        """
        setup = psycopg.connect(database)
        setup.execute("TRUNCATE articles, article_clusters RESTART IDENTITY CASCADE")
        setup.commit()
        base = unit(1)
        ids = [add_article(setup, domain=f"d{i}.com") for i in range(8)]
        vectors = [at_cos(base, 0.97, seed=100 + i) for i in range(8)]
        barrier = threading.Barrier(8)
        results, errors = [], []

        def worker(i):
            try:
                c = psycopg.connect(database)
                barrier.wait()
                results.append(assign(PgBackend(c), ids[i], vectors[i], domain=f"d{i}.com"))
                c.close()
            except Exception as e:     # pragma: no cover - surfaced below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len({r["cluster_id"] for r in results}) == 1
        c = cluster(setup, results[0]["cluster_id"])
        assert c["article_count"] == 8 and c["outlet_count"] == 8
        assert sorted(r["action"] for r in results).count("seed") == 1
        setup.close()


# ─────────────────────────────────────────────────────────────────────────────
# Merge / reconcile / prune
# ─────────────────────────────────────────────────────────────────────────────

class TestMaintenance:

    def _twins(self, conn, backend):
        """Two clusters about the same story that online assignment kept apart."""
        base = unit(1)
        a1 = new_and_assign(conn, backend, at_cos(base, 0.93, seed=2), domain="a.com", join=0.99)
        a2 = new_and_assign(conn, backend, at_cos(base, 0.93, seed=3), domain="b.com", join=0.99)
        a3 = new_and_assign(conn, backend, at_cos(base, 0.93, seed=4), domain="c.com", join=0.99)
        assert len({a1["cluster_id"], a2["cluster_id"], a3["cluster_id"]}) == 3
        return a1, a2, a3

    def test_merge_clusters_moves_members_and_sums(self, conn, backend):
        base = unit(1)
        a = new_and_assign(conn, backend, base, domain="a.com")
        new_and_assign(conn, backend, at_cos(base, 0.97, seed=2), domain="b.com")
        lone = new_and_assign(conn, backend, at_cos(base, 0.8, seed=3), domain="c.com")
        assert backend.merge_clusters(a["cluster_id"], lone["cluster_id"]) is True
        w = cluster(conn, a["cluster_id"])
        assert w["article_count"] == 3 and w["outlet_count"] == 3
        loser = cluster(conn, lone["cluster_id"])
        assert loser["status"] == "merged" and str(loser["merged_into"]) == str(a["cluster_id"])
        assert conn.execute("SELECT count(*) FROM articles WHERE cluster_id = %s",
                            (a["cluster_id"],)).fetchone()[0] == 3

    def test_merge_refuses_inactive_or_cross_model(self, conn, backend):
        a = new_and_assign(conn, backend, unit(1))
        b = new_and_assign(conn, backend, unit(2))
        other = new_and_assign(conn, backend, unit(3), model="other-model")
        assert backend.merge_clusters(a["cluster_id"], a["cluster_id"]) is False
        assert backend.merge_clusters(a["cluster_id"], other["cluster_id"]) is False
        assert backend.merge_clusters(a["cluster_id"], b["cluster_id"]) is True
        assert backend.merge_clusters(a["cluster_id"], b["cluster_id"]) is False   # b is now a tombstone

    def test_merge_repoints_tombstone_chain(self, conn, backend):
        a, b, c = (new_and_assign(conn, backend, unit(i)) for i in (1, 2, 3))
        backend.merge_clusters(b["cluster_id"], c["cluster_id"])
        backend.merge_clusters(a["cluster_id"], b["cluster_id"])
        assert str(cluster(conn, c["cluster_id"])["merged_into"]) == str(a["cluster_id"])

    def test_merge_candidates_report_every_probe(self, conn, backend):
        a1, a2, a3 = self._twins(conn, backend)
        rows = backend.find_merge_candidates({
            "p_model": MODEL, "p_since": (T0 - timedelta(days=3650)).isoformat(),
            "p_threshold": 0.8, "p_anchor_threshold": 0.5, "p_max_gap_hours": 18,
            "p_max_span_hours": 120, "p_probe_limit": 200,
        })
        assert len(rows) == 3                                   # one row per probe
        assert all(r["other_id"] is not None for r in rows)
        assert all(r["similarity"] == pytest.approx(0.93 * 0.93, abs=0.05) for r in rows)

    def test_merge_duplicates_end_to_end(self, conn, backend):
        self._twins(conn, backend)
        rep = merge_duplicates(backend, T0 - timedelta(days=3650), threshold=0.8, model=MODEL)
        assert rep.merges == 2
        active = conn.execute("SELECT article_count, outlet_count FROM article_clusters "
                              "WHERE status = 'active'").fetchall()
        assert active == [(3, 3)]

    def test_merge_duplicates_dry_run_changes_nothing(self, conn, backend):
        self._twins(conn, backend)
        rep = merge_duplicates(backend, T0 - timedelta(days=3650), threshold=0.8, model=MODEL, dry_run=True)
        assert rep.merges == 0 and len(rep.planned) == 1
        assert conn.execute("SELECT count(*) FROM article_clusters WHERE status = 'active'").fetchone()[0] == 3

    def test_reconcile_repairs_counts_after_article_deletion(self, conn, backend):
        base = unit(1)
        a = new_and_assign(conn, backend, base, domain="a.com")
        b = new_and_assign(conn, backend, at_cos(base, 0.97), domain="b.com")
        conn.execute("DELETE FROM articles WHERE id = %s", (b["article_id"],))   # Layer 1 pruning
        conn.commit()
        fixed = backend.reconcile_cluster_counts((T0 - timedelta(days=3650)).isoformat())
        assert fixed == 1
        c = cluster(conn, a["cluster_id"])
        assert c["article_count"] == 1 and c["outlet_count"] == 1 and c["outlet_set"] == ["a.com"]
        assert backend.reconcile_cluster_counts((T0 - timedelta(days=3650)).isoformat()) == 0

    def test_prune_deletes_only_idle_orphans(self, conn, backend):
        a = new_and_assign(conn, backend, unit(1))
        b = new_and_assign(conn, backend, unit(2))
        conn.execute("DELETE FROM articles WHERE id = %s", (b["article_id"],))
        conn.execute("UPDATE article_clusters SET updated_at = now() - interval '10 days'")
        conn.commit()
        assert backend.prune_orphan_clusters(168) == 1
        assert cluster(conn, b["cluster_id"]) is None and cluster(conn, a["cluster_id"]) is not None

    def test_run_maintenance_full_pass(self, conn, backend):
        rep = run_maintenance(backend, lookback_hours=24)
        assert rep.merges == 0 and rep.reconciled == 0 and rep.pruned == 0


# ─────────────────────────────────────────────────────────────────────────────
# Work queue
# ─────────────────────────────────────────────────────────────────────────────

def _claim(conn, limit=10, age=0, lease=150, attempts=3, retry=24, retries=7, min_age=0) -> list[int]:
    rows = conn.execute("SELECT id FROM wizer_claim_enrichment_batch(%s, %s, %s, %s, %s, %s, %s)",
                        (limit, age, lease, attempts, retry, retries, min_age)).fetchall()
    conn.commit()
    return sorted(r[0] for r in rows)


def _health(conn, col):
    return conn.execute(f"SELECT {col} FROM enrichment_queue_health").fetchone()[0]


class TestQueue:

    def test_claims_are_disjoint_across_runners(self, database):
        c1, c2 = psycopg.connect(database), psycopg.connect(database)
        c1.execute("TRUNCATE articles RESTART IDENTITY CASCADE")
        c1.commit()
        now = datetime.now(timezone.utc)
        for i in range(10):
            add_article(c1, published=now - timedelta(minutes=i))
        first, second = _claim(c1, 6), _claim(c2, 6)
        assert len(first) == 6 and len(second) == 4
        assert not set(first) & set(second)
        c1.close()
        c2.close()

    def test_oldest_ingested_first(self, conn):
        now = datetime.now(timezone.utc)
        newest = add_article(conn, ingested=now - timedelta(minutes=1))
        oldest = add_article(conn, ingested=now - timedelta(days=5), published=now)
        middle = add_article(conn, ingested=now - timedelta(hours=3))
        rows = conn.execute("SELECT id FROM wizer_claim_enrichment_batch(1, 0, 150, 3, 24, 7, 0)").fetchall()
        assert [r[0] for r in rows] == [oldest]           # published_at plays no part
        assert _claim(conn, 1) == [middle]
        assert _claim(conn, 1) == [newest]

    def test_every_unenriched_article_is_eligible(self, conn):
        """v1 never claimed these: aged out, undated, stale-dated or not crawled."""
        now = datetime.now(timezone.utc)
        ids = [
            add_article(conn, ingested=now - timedelta(days=10)),             # backlog
            add_article(conn, published=None),                                # no published_at
            add_article(conn, published=now - timedelta(days=400)),           # republished old item
            add_article(conn, published=now + timedelta(days=2)),             # future-dated
            add_article(conn, crawled=False),                                 # crawl failed
        ]
        add_article(conn, enriched=True)                                      # already done
        assert _claim(conn) == sorted(ids)
        assert conn.execute("SELECT wizer_enrichment_queue_depth(0, 3)").fetchone()[0] == len(ids)

    def test_sweeper_min_age_leaves_fresh_articles_alone(self, conn):
        now = datetime.now(timezone.utc)
        old = add_article(conn, ingested=now - timedelta(hours=7))
        add_article(conn, ingested=now - timedelta(minutes=30))           # hand-off runner's
        assert _claim(conn, min_age=6 * 60) == [old]

    def test_optional_age_gate_uses_ingestion_time(self, conn):
        now = datetime.now(timezone.utc)
        fresh = add_article(conn, published=now - timedelta(days=30))        # old date, just ingested
        add_article(conn, ingested=now - timedelta(hours=72), published=now)
        assert _claim(conn, age=48) == [fresh]

    def test_dead_letter_is_retried_daily_then_given_up(self, conn):
        aid = add_article(conn)
        assert _claim(conn) == [aid]
        assert _claim(conn) == []                                # leased
        for _ in range(2):
            conn.execute("UPDATE articles SET enrich_claimed_at = now() - interval '3 hours'")
            conn.commit()
            assert _claim(conn) == [aid]                         # lease expired → reclaimed
        conn.execute("UPDATE articles SET enrich_claimed_at = now() - interval '3 hours'")
        conn.commit()
        assert _claim(conn) == []                                # 3 attempts → dead letter …
        assert _health(conn, "dead_letter") == 1 and _health(conn, "pending") == 0
        for _ in range(7):                                       # … retried once a day, 7 times
            conn.execute("UPDATE articles SET enrich_claimed_at = now() - interval '25 hours'")
            conn.commit()
            assert _claim(conn) == [aid]
            assert _claim(conn) == []                            # not again the same day
        conn.execute("UPDATE articles SET enrich_claimed_at = now() - interval '25 hours'")
        conn.commit()
        assert _claim(conn) == []                                # 10 attempts → given up
        assert _health(conn, "given_up") == 1 and _health(conn, "dead_letter") == 0
        assert conn.execute("SELECT enriched_at FROM articles WHERE id = %s",
                            (aid,)).fetchone()[0] is None        # never counted as enriched

    def test_release_refunds_the_attempt(self, conn):
        aid = add_article(conn, published=datetime.now(timezone.utc))
        _claim(conn)
        released = conn.execute("SELECT wizer_release_enrichment_claims(%s)", ([aid],)).fetchone()[0]
        conn.commit()
        assert released == 1
        attempts, claimed = conn.execute(
            "SELECT enrich_attempts, enrich_claimed_at FROM articles WHERE id = %s", (aid,)).fetchone()
        assert attempts == 0 and claimed is None
        assert _claim(conn) == [aid]

    def test_queue_depth(self, conn):
        now = datetime.now(timezone.utc)
        add_article(conn)
        add_article(conn, ingested=now - timedelta(hours=100))
        assert conn.execute("SELECT wizer_enrichment_queue_depth(48, 3)").fetchone()[0] == 1
        assert conn.execute("SELECT wizer_enrichment_queue_depth(0, 3)").fetchone()[0] == 2

    def test_health_flags_backlog_and_crawl_failures(self, conn):
        now = datetime.now(timezone.utc)
        add_article(conn, ingested=now - timedelta(hours=30))
        add_article(conn, crawled=False)
        add_article(conn, enriched=True, ingested=now - timedelta(days=3))
        assert _health(conn, "unenriched_over_24h") == 1
        assert _health(conn, "uncrawled_pending") == 1
        assert _health(conn, "pending") == 2
        assert 30 * 60 - 5 <= _health(conn, "oldest_pending_minutes") <= 30 * 60 + 5


# ─────────────────────────────────────────────────────────────────────────────
# PostgREST argument decoding
#
# In production every call goes Python → supabase-py (JSON body) → PostgREST,
# which decodes the body with json_to_record() against the function's declared
# argument types and then calls the function with named arguments. The risky
# conversions are exactly the ones our RPCs rely on: a pgvector literal sent
# as a JSON string, JSON arrays becoming text[] / bigint[], nested JSON
# becoming jsonb. These tests push the real payloads through that same
# decoding step.
# ─────────────────────────────────────────────────────────────────────────────

def postgrest_call(conn, fn: str, payload: dict) -> list[dict]:
    import json as _json
    arg_rows = conn.execute(
        "SELECT unnest(proargnames) AS name, format_type(unnest(proargtypes::oid[]), NULL) AS type, "
        "       generate_series(1, pronargs) AS pos "
        "FROM pg_proc WHERE proname = %s AND pronargs > 0", (fn,)
    ).fetchall()
    in_args = [(n, t) for n, t, _ in arg_rows if n in payload]
    coldefs = ", ".join(f'"{n}" {t}' for n, t in in_args)
    call = ", ".join(f'"{n}" := a."{n}"' for n, _ in in_args)
    sql = (f"SELECT r.* FROM json_to_record(%s::json) AS a({coldefs}), "
           f"LATERAL {fn}({call}) AS r")
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql, (_json.dumps(payload),)).fetchall()
    conn.commit()
    return rows


class TestPostgrestDecoding:

    def test_assign_cluster_payload_from_python(self, conn):
        from enrichment.clustering import build_assign_params
        aid = add_article(conn, domain="thehindu.com")
        article = {"id": aid, "title": "Headline", "domain": "thehindu.com",
                   "published_at": T0.isoformat(), "language_code": "en"}
        ents = [{"entity_text": "Smit Machchhar", "entity_type": "PERSON", "salience": 0.9}]
        payload = build_assign_params(article, list(unit(5)), ents, -1234567890123, "en")
        rows = postgrest_call(conn, "wizer_assign_cluster", payload)
        assert rows[0]["action"] == "seed"
        c = cluster(conn, rows[0]["cluster_id"])
        assert c["top_entities"] == [{"text": "Smit Machchhar", "type": "PERSON", "count": 1}]
        assert c["entity_set"] == ["smit machchhar"]
        assert c["image_hashes"] == [{"h": -1234567890123, "d": "thehindu.com"}]
        assert c["embedding_model"] == payload["p_model"]

    def test_claim_and_release_payloads(self, conn):
        now = datetime.now(timezone.utc)
        ids = [add_article(conn, published=now - timedelta(minutes=i)) for i in range(3)]
        rows = postgrest_call(conn, "wizer_claim_enrichment_batch", {
            "p_limit": 2, "p_max_age_hours": 0, "p_lease_minutes": 150, "p_max_attempts": 3,
            "p_retry_hours": 24, "p_max_retries": 7, "p_min_age_minutes": 0})
        assert sorted(r["id"] for r in rows) == sorted(ids[:2])
        assert "full_text" in rows[0] and "enrich_attempts" in rows[0]     # SETOF articles
        released = postgrest_call(conn, "wizer_release_enrichment_claims", {"p_ids": ids})
        assert list(released[0].values())[0] == 2

    def test_merge_candidates_payload_from_python(self, conn, backend):
        base = unit(1)
        new_and_assign(conn, backend, at_cos(base, 0.93, seed=2), join=0.99)
        new_and_assign(conn, backend, at_cos(base, 0.93, seed=3), join=0.99)
        payload = {
            "p_model": MODEL, "p_since": (T0 - timedelta(days=1)).isoformat(), "p_threshold": 0.8,
            "p_anchor_threshold": 0.5, "p_max_gap_hours": 18, "p_max_span_hours": 120,
            "p_probe_limit": 200, "p_after_ts": "-infinity",
            "p_after_id": "00000000-0000-0000-0000-000000000000",
        }
        rows = postgrest_call(conn, "wizer_find_cluster_merge_candidates", payload)
        assert len(rows) == 2 and all(r["other_id"] for r in rows)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 table-size cap (docs/ingestion_fixes_migration.sql)
# ─────────────────────────────────────────────────────────────────────────────

class TestArticlePruning:

    def test_prunes_oldest_down_to_target_in_chunks(self, conn):
        ids = [add_article(conn) for _ in range(25)]
        assert conn.execute("SELECT wizer_prune_articles(30, 10, 5)").fetchone()[0] == 0  # under limit
        first = conn.execute("SELECT wizer_prune_articles(20, 10, 5)").fetchone()[0]
        conn.commit()
        assert first == 5                                   # chunk-bounded
        total = 5
        while True:                                         # caller loop, as in pipeline/db.py
            n = conn.execute("SELECT wizer_prune_articles(10, 10, 5)").fetchone()[0]
            conn.commit()
            total += n
            if n < 5:
                break
        remaining = [r[0] for r in conn.execute("SELECT id FROM articles ORDER BY id").fetchall()]
        assert total == 15 and remaining == ids[-10:]       # the NEWEST 10 survive


# ─────────────────────────────────────────────────────────────────────────────
# Clustering job feed + batched assignment (cluster.py run)
# ─────────────────────────────────────────────────────────────────────────────

class TestClusteringJob:

    def test_fetch_unclustered_keyset_oldest_first(self, conn):
        ids = [add_article(conn, published=T0 + timedelta(minutes=m)) for m in (30, 10, 20)]
        add_article(conn, published=T0 - timedelta(days=5))                     # outside window
        conn.execute("UPDATE articles SET full_text = repeat('x', 5000) WHERE id = %s", (ids[0],))
        conn.commit()
        since = (T0 - timedelta(hours=1)).isoformat()
        page1 = postgrest_call(conn, "wizer_fetch_unclustered", {"p_since": since, "p_limit": 2})
        assert [r["id"] for r in page1] == [ids[1], ids[2]]
        page2 = postgrest_call(conn, "wizer_fetch_unclustered", {
            "p_since": since, "p_limit": 2,
            "p_after_published": page1[-1]["published_at"].isoformat(), "p_after_id": page1[-1]["id"]})
        assert [r["id"] for r in page2] == [ids[0]]
        assert len(page2[0]["body_lead"]) == 1500                              # trimmed for the wire

    def test_fetch_skips_clustered_articles(self, conn, backend):
        r = new_and_assign(conn, backend, unit(1))
        other = add_article(conn)
        rows = postgrest_call(conn, "wizer_fetch_unclustered",
                              {"p_since": (T0 - timedelta(hours=1)).isoformat()})
        assert [x["id"] for x in rows] == [other] and r["article_id"] != other

    def test_batch_assign_matches_single_assign_and_isolates_failures(self, conn):
        from enrichment.clustering import _ITEM_KEYS, _SHARED_KEYS, build_assign_params
        base = unit(1)
        good = [add_article(conn, domain=f"d{i}.com") for i in range(3)]
        items, shared = [], None
        for i, aid in enumerate(good + [987654321]):           # last id does not exist
            art = {"id": aid, "title": f"t{i}", "domain": f"d{i}.com",
                   "published_at": T0.isoformat(), "language_code": "en"}
            p = build_assign_params(art, list(at_cos(base, 0.97, seed=200 + i)), [], None, "en", model_id=MODEL)
            shared = shared or {k: p[k] for k in _SHARED_KEYS}
            items.append({k: p[k] for k in _ITEM_KEYS})
        rows = postgrest_call(conn, "wizer_assign_cluster_batch", {"p_items": items, **shared})
        assert [r.get("error") for r in rows[:3]] == [None, None, None], rows
        assert [r["action"] for r in rows[:3]] == ["seed", "join", "join"]
        assert len({r["cluster_id"] for r in rows[:3]}) == 1 and rows[2]["outlet_count"] == 3
        assert rows[3]["cluster_id"] is None and "does not exist" in rows[3]["error"]
        again = postgrest_call(conn, "wizer_assign_cluster_batch", {"p_items": items[:1], **shared})
        assert again[0]["action"] == "existing"                                # idempotent


# ─────────────────────────────────────────────────────────────────────────────
# Article archive (docs/archive_migration.sql + archiver/)
# ─────────────────────────────────────────────────────────────────────────────

def _old_articles(conn, day, n, with_entity=True):
    ids = []
    for i in range(n):
        aid = conn.execute(
            "INSERT INTO articles (url, url_hash, title, full_text, published_at, crawled_at, domain, "
            "is_crawled, keywords) VALUES (%s, %s, %s, %s, %s, %s, 'x.com', true, %s::jsonb) RETURNING id",
            (f"https://x.com/{uuid.uuid4()}", uuid.uuid4().int >> 65, f"old {i}", "body " * 50,
             day, datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=1, minutes=i),
             '["k"]'),
        ).fetchone()[0]
        if with_entity:
            conn.execute("INSERT INTO article_entities (article_id, entity_text, entity_type, salience) "
                         "VALUES (%s, 'Modi', 'PERSON', 0.9)", (aid,))
        ids.append(aid)
    conn.commit()
    return ids


class TestArchiveSQL:
    OLD = (datetime.now(timezone.utc) - timedelta(days=60)).date()

    def _log(self, conn, status="verified", rows=3, pruned=0):
        conn.execute("INSERT INTO article_archive_log (day, status, article_rows, bucket, pruned_rows) "
                     "VALUES (%s, %s, %s, 'b', %s)", (self.OLD, status, rows, pruned))
        conn.commit()

    def test_days_listing_respects_hot_window(self, conn):
        _old_articles(conn, self.OLD, 1)
        add_article(conn)                                           # crawled now → hot
        days = [r[0] for r in conn.execute("SELECT day FROM wizer_archive_days(30)").fetchall()]
        assert days[0] == self.OLD
        assert max(days) < (datetime.now(timezone.utc) - timedelta(days=30)).date()

    def test_prune_refuses_unverified_day(self, conn):
        _old_articles(conn, self.OLD, 3)
        self._log(conn, status="uploaded")
        with pytest.raises(psycopg.Error, match="not verified"):
            conn.execute("SELECT wizer_prune_archived_day(%s, 30, 100)", (self.OLD,))
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM articles").fetchone()[0] == 3

    def test_prune_refuses_day_without_log(self, conn):
        _old_articles(conn, self.OLD, 1)
        with pytest.raises(psycopg.Error, match="no archive log"):
            conn.execute("SELECT wizer_prune_archived_day(%s, 30, 100)", (self.OLD,))
        conn.rollback()

    def test_prune_refuses_hot_window(self, conn):
        recent = datetime.now(timezone.utc).date() - timedelta(days=5)
        conn.execute("INSERT INTO article_archive_log (day, status, article_rows, bucket) "
                     "VALUES (%s, 'verified', 0, 'b')", (recent,))
        conn.commit()
        with pytest.raises(psycopg.Error, match="hot window"):
            conn.execute("SELECT wizer_prune_archived_day(%s, 30, 100)", (recent,))
        conn.rollback()

    def test_prune_refuses_if_day_changed_since_export(self, conn):
        _old_articles(conn, self.OLD, 4)
        self._log(conn, rows=3)                                     # one row was never archived
        with pytest.raises(psycopg.Error, match="changed since export"):
            conn.execute("SELECT wizer_prune_archived_day(%s, 30, 100)", (self.OLD,))
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM articles").fetchone()[0] == 4

    def test_prune_chunks_cascades_entities_and_closes_day(self, conn):
        _old_articles(conn, self.OLD, 5)
        self._log(conn, rows=5)
        assert conn.execute("SELECT wizer_prune_archived_day(%s, 30, 2)", (self.OLD,)).fetchone()[0] == 2
        conn.commit()
        assert conn.execute("SELECT status, pruned_rows FROM article_archive_log").fetchone() == ("verified", 2)
        while conn.execute("SELECT wizer_prune_archived_day(%s, 30, 2)", (self.OLD,)).fetchone()[0]:
            conn.commit()
        conn.commit()
        assert conn.execute("SELECT count(*) FROM articles").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM article_entities").fetchone()[0] == 0
        assert conn.execute("SELECT status, pruned_rows FROM article_archive_log").fetchone() == ("pruned", 5)

    def test_run_archive_end_to_end_on_real_sql(self, conn, monkeypatch):
        """Full export → verify → prune through the real SQL, with an in-memory bucket."""
        from archiver import runner
        from archiver.parquet import parquet_ids
        from tests.test_archive import FakeStore
        from tools.pg_backend import PgArchiveDB
        monkeypatch.setattr(runner, "PART_ROWS", 2)
        monkeypatch.setattr(runner, "PAGE", 2)
        ids = _old_articles(conn, self.OLD, 5)
        hot = add_article(conn)                                     # inside the hot window: must survive
        store = FakeStore()
        rep = runner.run_archive(PgArchiveDB(conn), store, "b", hot_days=30)
        assert rep.days_waiting == 1 and rep.rows_pruned == 0       # unenriched: the archive waits
        conn.execute("UPDATE articles SET enriched_at = now()")
        conn.commit()
        rep = runner.run_archive(PgArchiveDB(conn), store, "b", hot_days=30)
        assert not rep.failures and rep.rows_pruned == 5
        archived = sorted(i for p, d in store.objects.items() if p.startswith("articles/")
                          for i in parquet_ids(d))
        assert archived == sorted(ids)
        assert conn.execute("SELECT array_agg(id) FROM articles").fetchone()[0] == [hot]
        status = conn.execute("SELECT status, article_rows, entity_rows FROM article_archive_log "
                              "WHERE day = %s", (self.OLD,)).fetchone()
        assert status == ("pruned", 5, 5)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory clustering ≡ wizer_assign_cluster (enrichment/memory_clustering.py)
# ─────────────────────────────────────────────────────────────────────────────

def _story_stream(n=240, seed=7):
    """Articles from 12 stories (similarities spread across the 0.85 join line) + noise,
    over 30 h, mostly in time order with some late arrivals."""
    rng = np.random.default_rng(seed)
    bases = [unit(1000 + s) for s in range(12)]
    out = []
    for i in range(n):
        if rng.random() < 0.2:
            v = unit(5000 + i)
        else:
            v = at_cos(bases[int(rng.integers(12))], float(rng.uniform(0.84, 0.98)), 6000 + i)
        late = float(rng.uniform(-3, 0)) if rng.random() < 0.1 else 0.0
        t = T0 + timedelta(hours=30 * i / n + late)
        out.append({"v": v, "t": t, "domain": f"site{int(rng.integers(15))}.com",
                    "title": f"title {i}", "lang": "en" if rng.random() < 0.7 else "hi"})
    return out


def _mem_params():
    from enrichment.memory_clustering import _Params
    return _Params(join=0.85, anchor=0.75, gap_s=18 * 3600, span_s=120 * 3600, candidates=5)


def _apply(conn, state):
    clusters, articles = state.pending_changes()
    row = postgrest_call(conn, "wizer_apply_cluster_changes",
                         {"p_model": MODEL, "p_clusters": clusters, "p_articles": articles})[0]
    state.mark_flushed(row["synced_at"].isoformat())
    return row


def _vec(text):
    return [float(x) for x in str(text).strip("[]").split(",")]


def _fetcher(conn):
    return lambda **kw: postgrest_call(conn, "wizer_cluster_state", kw)


class TestMemoryParity:

    def test_decisions_match_sql(self, conn, backend):
        from enrichment.memory_clustering import ClusterState
        st = ClusterState(model=MODEL)
        mapping, actions = {}, []
        for a in _story_stream():
            aid = add_article(conn, domain=a["domain"], title=a["title"], published=a["t"])
            sql = assign(backend, aid, a["v"], published=a["t"], domain=a["domain"],
                         title=a["title"], lang=a["lang"])
            mem = st.assign(aid, a["v"], a["t"], a["domain"], a["title"], a["lang"], _mem_params())
            assert mem.action == sql["action"], f"article {aid}"
            assert mapping.setdefault(sql["cluster_id"], mem.cluster_id) == mem.cluster_id
            if sql["action"] == "join":
                assert mem.similarity == pytest.approx(sql["similarity"], abs=1e-4)
            actions.append(sql["action"])
        assert actions.count("join") >= 60 and actions.count("seed") >= 30   # the test bites

    def test_bulk_write_reproduces_sql_cluster_rows(self, conn, backend):
        from enrichment.memory_clustering import ClusterState
        stream = _story_stream(120, seed=11)
        ids = [add_article(conn, domain=a["domain"], title=a["title"], published=a["t"]) for a in stream]
        for aid, a in zip(ids, stream):                                   # reference: the SQL path
            assign(backend, aid, a["v"], published=a["t"], domain=a["domain"], title=a["title"], lang=a["lang"])
        q = ("SELECT c.*, (SELECT array_agg(id ORDER BY id) FROM articles WHERE cluster_id = c.id) AS members "
             "FROM article_clusters c")
        with conn.cursor(row_factory=dict_row) as cur:
            ref = {r["canonical_article_id"]: r for r in cur.execute(q).fetchall()}
        conn.execute("UPDATE articles SET cluster_id = NULL, cluster_similarity = NULL, "
                     "cluster_assignment = NULL, clustered_at = NULL")
        conn.execute("TRUNCATE article_clusters CASCADE")
        conn.commit()

        st = ClusterState(model=MODEL)
        written = 0
        for k, (aid, a) in enumerate(zip(ids, stream)):
            st.assign(aid, a["v"], a["t"], a["domain"], a["title"], a["lang"], _mem_params())
            if k in (40, 80):                                           # flush mid-stream: exercises updates
                written += _apply(conn, st)["articles_written"]
        written += _apply(conn, st)["articles_written"]
        assert written == len(stream)
        with conn.cursor(row_factory=dict_row) as cur:
            got = {r["canonical_article_id"]: r for r in cur.execute(q).fetchall()}
        assert set(got) == set(ref)
        for key, r in ref.items():
            g = got[key]
            assert g["members"] == r["members"]
            for col in ("article_count", "outlet_count", "outlet_set", "language_set",
                        "first_seen_at", "last_seen_at", "representative_article_id", "headline", "status"):
                assert g[col] == r[col], (key, col)
            assert np.allclose(_vec(g["centroid_sum"]), _vec(r["centroid_sum"]), atol=1e-5)
            assert np.allclose(_vec(g["representative"]), _vec(r["representative"]), atol=2e-3)
            assert np.allclose(_vec(g["anchor"]), _vec(r["anchor"]), atol=2e-3)

    def test_state_from_db_continues_identically(self, conn):
        """Cold start (full load from the DB) gives the same decisions as staying in memory."""
        from enrichment.memory_clustering import ClusterState
        from enrichment.cluster_state_sync import apply_delta, load_full
        stream = _story_stream(160, seed=13)
        ids = [add_article(conn, domain=a["domain"], title=a["title"], published=a["t"]) for a in stream]
        live = ClusterState(model=MODEL)
        for aid, a in list(zip(ids, stream))[:100]:
            live.assign(aid, a["v"], a["t"], a["domain"], a["title"], a["lang"], _mem_params())
        _apply(conn, live)
        cold = load_full(_fetcher(conn), MODEL, window_since=T0 - timedelta(days=1),
                         db_now=lambda: live.synced_at)
        assert len(cold) == len(live)
        assert apply_delta(_fetcher(conn), cold, MODEL) == 0        # nothing changed since
        for aid, a in list(zip(ids, stream))[100:]:
            x = live.assign(aid, a["v"], a["t"], a["domain"], a["title"], a["lang"], _mem_params())
            y = cold.assign(aid, a["v"], a["t"], a["domain"], a["title"], a["lang"], _mem_params())
            assert x.action == y.action
            assert (x.cluster_id == y.cluster_id) or x.action == "seed"   # new seeds get fresh uuids
            assert x.similarity == pytest.approx(y.similarity, abs=1e-6)

    def test_delta_drops_merged_and_picks_up_changes(self, conn):
        from enrichment.memory_clustering import ClusterState
        from enrichment.cluster_state_sync import apply_delta
        st = ClusterState(model=MODEL)
        a1 = add_article(conn)
        a2 = add_article(conn, domain="b.com")
        st.assign(a1, unit(1), T0, "a.com", "t", "en", _mem_params())
        st.assign(a2, unit(2), T0, "b.com", "t", "en", _mem_params())
        _apply(conn, st)
        c1, c2 = st.ids
        assert conn.execute("SELECT wizer_merge_clusters(%s, %s)", (c1, c2)).fetchone()[0]
        conn.commit()
        assert apply_delta(_fetcher(conn), st, MODEL) == 2
        assert st.ids == [c1] and st.counts[0] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Bulk I/O (docs/bulk_io_migration.sql)
# ─────────────────────────────────────────────────────────────────────────────

class TestBulkIO:

    def test_apply_is_idempotent_and_never_revives_tombstones(self, conn):
        from enrichment.memory_clustering import ClusterState
        st = ClusterState(model=MODEL)
        aid = add_article(conn)
        st.assign(aid, unit(1), T0, "a.com", "t", "en", _mem_params())
        clusters, articles = st.pending_changes()
        for _ in range(2):                                              # retry of the same batch
            row = postgrest_call(conn, "wizer_apply_cluster_changes",
                                 {"p_model": MODEL, "p_clusters": clusters, "p_articles": articles})[0]
        assert conn.execute("SELECT count(*) FROM article_clusters").fetchone()[0] == 1
        assert row["articles_written"] == 0                             # already stamped
        conn.execute("UPDATE article_clusters SET status = 'merged'")
        conn.commit()
        clusters[0]["is_new"] = False
        clusters[0]["article_count"] = 99
        postgrest_call(conn, "wizer_apply_cluster_changes",
                       {"p_model": MODEL, "p_clusters": clusters, "p_articles": []})
        assert conn.execute("SELECT status, article_count FROM article_clusters").fetchone() == ("merged", 1)

    def test_save_enrichment_batch(self, conn, backend):
        a1 = add_article(conn)
        a2 = add_article(conn)
        seeded = assign(backend, a1, unit(1))
        conn.execute("UPDATE articles SET enrich_error = 'old failure' WHERE id = %s", (a1,))
        conn.execute("INSERT INTO article_entities (article_id, entity_text, entity_type, salience) "
                     "VALUES (%s, 'stale', 'ORG', 0.5)", (a1,))
        conn.commit()
        before = cluster(conn, seeded["cluster_id"])["updated_at"]
        items = [
            {"id": a1, "update": {"category": "politics", "word_count": 120, "ai_tag": ["elections"],
                                  "keywords": ["vote"], "sentiment_score": 0.4, "image_phash": -5},
             "entities": [{"entity_text": "Modi", "entity_type": "PERSON", "salience": 0.9}],
             "cluster_entities": [{"text": "Modi", "type": "PERSON"}], "entity_keys": ["modi"]},
            {"id": a2, "update": {"category": "sports"}, "entities": []},
            {"id": 987654321, "update": {"category": "ghost"}},             # deleted meanwhile
        ]
        n = postgrest_call(conn, "wizer_save_enrichment_batch", {"p_items": items})
        assert list(n[0].values())[0] == 2
        with conn.cursor(row_factory=dict_row) as cur:
            r = cur.execute("SELECT * FROM articles WHERE id = %s", (a1,)).fetchone()
        assert r["category"] == "politics" and r["ai_tag"] == ["elections"] and r["image_phash"] == -5
        assert r["enriched_at"] is not None and r["enrich_error"] is None
        ents = conn.execute("SELECT entity_text FROM article_entities WHERE article_id = %s", (a1,)).fetchall()
        assert ents == [("Modi",)]
        c = cluster(conn, seeded["cluster_id"])
        assert c["top_entities"][0]["text"] == "Modi" and c["entity_set"] == ["modi"]
        assert c["updated_at"] == before                                # delta sync untouched

    def test_record_feed_polls(self, conn):
        conn.execute("TRUNCATE feeds CASCADE")
        specs = [(2, True, None), (4, True, None), (0, True, None), (0, False, "dormant"), (1, True, None)]
        ids = [conn.execute("INSERT INTO feeds (feed_url, fail_count, articles_found, is_active, disabled_reason) "
                            "VALUES (%s, %s, 10, %s, %s) RETURNING id",
                            (f"https://f{i}.com/rss", fc, act, rsn)).fetchone()[0]
               for i, (fc, act, rsn) in enumerate(specs)]
        conn.commit()
        items = [
            {"id": str(ids[0]), "success": True, "new_articles": 3},
            {"id": str(ids[1]), "success": False},                       # 5th failure → circuit opens
            {"id": str(ids[2]), "success": True, "new_articles": 0, "dormant": True},
            {"id": str(ids[3]), "success": True, "new_articles": 2, "reactivate": True},
            {"id": str(ids[4]), "success": False},
        ]
        n = postgrest_call(conn, "wizer_record_feed_polls", {"p_items": items, "p_max_errors": 5})
        assert list(n[0].values())[0] == 5
        with conn.cursor(row_factory=dict_row) as cur:
            f = {str(r["id"]): r for r in cur.execute("SELECT * FROM feeds").fetchall()}
        r0, r1, r2, r3, r4 = (f[str(i)] for i in ids)
        assert r0["fail_count"] == 0 and r0["articles_found"] == 13 and r0["last_new_article_at"] is not None
        assert r1["fail_count"] == 5 and not r1["is_active"] and r1["disabled_reason"] == "errors"
        assert not r2["is_active"] and r2["disabled_reason"] == "dormant" and r2["articles_found"] == 10
        assert r3["is_active"] and r3["disabled_reason"] is None and r3["articles_found"] == 12
        assert r4["fail_count"] == 2 and r4["is_active"] and r4["last_success_at"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Hand-off jobs end to end (enrichment/cluster_job.py, enrichment/handoff_runner.py)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def via_postgrest(conn, monkeypatch):
    """Route enrichment.db's bulk RPCs through PostgREST-style argument decoding."""
    from enrichment import db as edb

    def rpc_row(fn):
        return lambda **kw: postgrest_call(conn, fn, kw)

    monkeypatch.setattr(edb, "fetch_cluster_state", rpc_row("wizer_cluster_state"))
    monkeypatch.setattr(edb, "db_now", lambda: conn.execute("SELECT now()").fetchone()[0].isoformat())
    monkeypatch.setattr(edb, "apply_cluster_changes", lambda model, clusters, articles: postgrest_call(
        conn, "wizer_apply_cluster_changes",
        {"p_model": model, "p_clusters": clusters, "p_articles": articles})[0])
    monkeypatch.setattr(edb, "save_enrichment_batch", lambda items: list(postgrest_call(
        conn, "wizer_save_enrichment_batch", {"p_items": items})[0].values())[0])
    monkeypatch.setattr(edb, "log_run_start", lambda row: None)
    monkeypatch.setattr(edb, "log_run_finish", lambda rid, row: None)
    return edb


def _records(conn, stream):
    out = []
    for a in stream:
        aid = add_article(conn, domain=a["domain"], title=a["title"], published=a["t"])
        out.append({"id": aid, "title": a["title"], "description": "", "full_text": "body " * 60,
                    "published_at": a["t"].isoformat(), "crawled_at": a["t"].isoformat(),
                    "domain": a["domain"], "language_code": a["lang"], "_v": a["v"]})
    return out


class TestHandoffJobs:

    def test_cluster_job_two_runs_with_cached_state(self, conn, via_postgrest, monkeypatch, tmp_path):
        """Work list from the DB; a second run resumes from the cached state; nothing is skipped."""
        from enrichment import cluster_job
        from enrichment import db as edb
        monkeypatch.setattr(edb, "fetch_unclustered_ingested",
                            lambda since, limit, after_ts, after_id: postgrest_call(
                                conn, "wizer_fetch_unclustered_ingested",
                                {"p_since": since, "p_limit": limit,
                                 **({"p_after_ts": after_ts.isoformat() if hasattr(after_ts, "isoformat") else after_ts,
                                     "p_after_id": after_id} if after_ts is not None else {})}))
        monkeypatch.setattr(cluster_job, "CLUSTER_STATE_RETAIN_HOURS", 10_000)   # synthetic T0 is in the past
        stream = _story_stream(150, seed=21)
        recs = _records(conn, stream[:80])
        vec_of = {r["id"]: r["_v"] for r in recs}
        seen_texts = []

        def fake_embed(texts):
            seen_texts.extend(texts)
            return [vec_of[i] for i in pending.pop(0)]
        pending = []
        real_fetch = edb.fetch_unclustered_ingested

        def fetch_and_note(*a):
            rows = real_fetch(*a)
            if rows:
                pending.append([r["id"] for r in rows])
            return rows
        monkeypatch.setattr(edb, "fetch_unclustered_ingested", fetch_and_note)
        monkeypatch.setattr(cluster_job, "embed_texts", fake_embed)

        s1 = cluster_job.run_memory_clustering(tmp_path / "state.npz", {recs[0]["id"]: "BODY"},
                                               since_hours=100_000, page=50, model=MODEL)
        assert (tmp_path / "state.npz").exists() and s1["written_articles"] == 80 and s1["with_body"] == 1
        more = _records(conn, stream[80:])
        vec_of.update({r["id"]: r["_v"] for r in more})
        s2 = cluster_job.run_memory_clustering(tmp_path / "state.npz", None, since_hours=100_000,
                                               page=50, model=MODEL)
        assert s2["seen"] == 70 and s2["written_articles"] == 70
        bad = conn.execute("""
            SELECT count(*) FROM article_clusters c
             WHERE c.article_count <> (SELECT count(*) FROM articles a WHERE a.cluster_id = c.id)""").fetchone()[0]
        assert bad == 0
        assert conn.execute("SELECT count(*) FROM articles WHERE cluster_id IS NULL").fetchone()[0] == 0

    def test_handoff_enrichment_saves_in_bulk(self, conn, backend, via_postgrest, monkeypatch):
        from enrichment import handoff_runner
        recs = _records(conn, _story_stream(5, seed=3))
        seeded = assign(backend, recs[0]["id"], unit(1))
        calls = []

        def fake_enrich(article):
            if article["id"] == recs[4]["id"]:
                raise RuntimeError("poison")
            calls.append(article["_image_phash"])
            return ({"category": "politics", "word_count": 300},
                    [{"entity_text": "Smit Machchhar", "entity_type": "PERSON", "salience": 0.8}])
        monkeypatch.setattr(handoff_runner, "enrich_one", fake_enrich)
        monkeypatch.setattr(handoff_runner, "_hash_image", lambda url: 42)
        monkeypatch.setattr(handoff_runner, "SAVE_EVERY", 2)
        s = handoff_runner.run_handoff_enrichment(recs)
        assert s["processed"] == 4 and s["saved"] == 4 and s["failed"] == 1 and calls == [42] * 4
        rows = conn.execute("SELECT id, category, enriched_at IS NOT NULL FROM articles ORDER BY id").fetchall()
        assert [r[2] for r in rows] == [True, True, True, True, False]    # the crash stays unenriched
        c = cluster(conn, seeded["cluster_id"])
        assert c["top_entities"][0]["text"] == "Smit Machchhar" and c["entity_set"] == ["smit machchhar"]
        assert c["top_entities"][0]["count"] == 1               # only article 0 is in this cluster


class TestSaveCrawlColumns:

    def test_crawl_merge_rules(self, conn):
        keep = add_article(conn, title="Feed title", published=T0)
        conn.execute("UPDATE articles SET description = 'rss desc', is_crawled = false WHERE id = %s", (keep,))
        blank = add_article(conn, title="", published=None)
        conn.commit()
        items = [
            {"id": keep, "update": {"category": "politics"},
             "crawl": {"title": "Page title", "title_simhash": 7, "description": "page desc",
                       "top_image_url": "https://x/i.jpg", "author": "A", "published_at": "2020-01-01T00:00:00+00:00",
                       "og_tags": {"og:title": "Page title"}, "is_crawled": True, "crawl_strategy": "default"}},
            {"id": blank, "update": {}, "crawl": {"title": "Page title", "title_simhash": 7,
                                                  "published_at": "2026-10-04T00:00:00+00:00",
                                                  "is_crawled": False, "crawl_strategy": "failed"}},
        ]
        postgrest_call(conn, "wizer_save_enrichment_batch", {"p_items": items})
        with conn.cursor(row_factory=dict_row) as cur:
            a = cur.execute("SELECT * FROM articles WHERE id = %s", (keep,)).fetchone()
            b = cur.execute("SELECT * FROM articles WHERE id = %s", (blank,)).fetchone()
        assert a["title"] == "Feed title"                       # feed title stays canonical
        assert a["published_at"] == T0                          # an existing date is never replaced
        assert a["description"] == "page desc" and a["author"] == "A" and a["og_tags"] == {"og:title": "Page title"}
        assert a["is_crawled"] and a["crawl_strategy"] == "default" and a["enriched_at"] is not None
        assert b["title"] == "Page title" and b["title_simhash"] == 7    # filled because empty
        assert b["published_at"] == datetime(2026, 10, 4, tzinfo=timezone.utc)
        assert not b["is_crawled"] and b["crawl_strategy"] == "failed"


class TestEnrichBeforeCluster:

    def test_entities_reach_the_story_in_either_order(self, conn):
        """Enrichment shards do not wait for clustering; the story still gets the entities."""
        from enrichment.memory_clustering import ClusterState
        a1 = add_article(conn)
        postgrest_call(conn, "wizer_save_enrichment_batch", {"p_items": [
            {"id": a1, "update": {"category": "politics"},
             "entities": [{"entity_text": f"E{i}", "entity_type": "PERSON", "salience": 1 - i / 20} for i in range(12)]}]})
        st = ClusterState(model=MODEL)
        st.assign(a1, unit(1), T0, "a.com", "t", "en", _mem_params())
        _apply(conn, st)                                           # clustered AFTER enrichment
        c = cluster(conn, st.ids[0])
        assert sorted(e["text"] for e in c["top_entities"]) == sorted(f"E{i}" for i in range(10))   # top 10 only
        a2 = add_article(conn, domain="b.com")
        st.assign(a2, at_cos(unit(1), 0.95, 2), T0, "b.com", "t", "en", _mem_params())
        _apply(conn, st)                                           # not enriched yet: nothing to merge
        postgrest_call(conn, "wizer_save_enrichment_batch", {"p_items": [
            {"id": a2, "update": {}, "entities": [{"entity_text": "E0", "entity_type": "PERSON", "salience": 0.9}],
             "cluster_entities": [{"text": "E0", "type": "PERSON"}], "entity_keys": ["e0"]}]})
        c = cluster(conn, st.ids[0])
        assert next(e for e in c["top_entities"] if e["text"] == "E0")["count"] == 2   # enrichment second


class TestArchiveGuard:

    def test_day_unenriched_counts_waiting_articles_only(self, conn):
        day = T0.date()
        waiting = add_article(conn, ingested=T0)
        done = add_article(conn, ingested=T0, enriched=True)
        gave_up = add_article(conn, ingested=T0)
        add_article(conn, ingested=T0 + timedelta(days=1))                 # another day
        conn.execute("UPDATE articles SET enrich_attempts = 10 WHERE id = %s", (gave_up,))
        conn.commit()
        n = conn.execute("SELECT wizer_archive_day_unenriched(%s)", (day,)).fetchone()[0]
        assert n == 1 and waiting and done


class TestInMemoryTwins:
    """ClusterState.merge_candidates ≡ wizer_find_cluster_merge_candidates; merge_twins end to end."""

    def _stream_into_sql(self, conn, backend, n=120, seed=31):
        stream = _story_stream(n, seed=seed)
        # join 0.99 keeps near-duplicates apart → plenty of twins to find
        for a in stream:
            aid = add_article(conn, domain=a["domain"], title=a["title"], published=a["t"])
            assign(backend, aid, a["v"], published=a["t"], domain=a["domain"], title=a["title"],
                   lang=a["lang"], join=0.99)

    def test_candidates_match_sql(self, conn, backend):
        from enrichment.cluster_state_sync import load_full
        from enrichment.memory_clustering import _Params
        self._stream_into_sql(conn, backend)
        sql_rows = backend.find_merge_candidates({
            "p_model": MODEL, "p_since": (T0 - timedelta(days=3650)).isoformat(),
            "p_threshold": 0.8, "p_anchor_threshold": 0.75, "p_max_gap_hours": 18,
            "p_max_span_hours": 120, "p_probe_limit": 10_000,
        })
        state = load_full(_fetcher(conn), MODEL, window_since=T0 - timedelta(days=1),
                          db_now=lambda: conn.execute("SELECT now()").fetchone()[0].isoformat())
        mem_rows = state.merge_candidates([r["probe_id"] for r in sql_rows], 0.8,
                                          _Params(join=0.85, anchor=0.75, gap_s=18 * 3600,
                                                  span_s=120 * 3600, candidates=5))
        sql = {str(r["probe_id"]): (str(r["other_id"]) if r["other_id"] else None, r["similarity"]) for r in sql_rows}
        mem = {r["probe_id"]: (r["other_id"], r["similarity"]) for r in mem_rows}
        assert set(sql) == set(mem) and len(sql) > 20
        assert sum(1 for v in sql.values() if v[0]) >= 5                         # the test bites
        for pid, (other, sim) in sql.items():
            assert mem[pid][0] == other, pid
            if other:
                assert mem[pid][1] == pytest.approx(sim, abs=1e-5)

    def test_merge_twins_end_to_end(self, conn, backend, via_postgrest, monkeypatch):
        from enrichment import cluster_job
        from enrichment import db as edb
        from enrichment.cluster_state_sync import load_full
        self._stream_into_sql(conn, backend, seed=33)
        monkeypatch.setattr(edb, "merge_clusters", lambda w, l: conn.execute(
            "SELECT wizer_merge_clusters(%s, %s)", (w, l)).fetchone()[0] or conn.commit() or True)
        monkeypatch.setattr("enrichment.config.CLUSTER_MERGE_THRESHOLD", 0.8)
        state = load_full(_fetcher(conn), MODEL, window_since=T0 - timedelta(days=1),
                          db_now=lambda: conn.execute("SELECT now()").fetchone()[0].isoformat())
        before = len(state)
        merged = cluster_job.merge_twins(state, set(state.ids), MODEL)
        conn.commit()
        active = conn.execute("SELECT count(*) FROM article_clusters WHERE status = 'active'").fetchone()[0]
        assert merged > 0 and len(state) == before - merged == active
        bad = conn.execute("""SELECT count(*) FROM article_clusters c WHERE c.status = 'active'
            AND c.article_count <> (SELECT count(*) FROM articles a WHERE a.cluster_id = c.id)""").fetchone()[0]
        assert bad == 0


# ─────────────────────────────────────────────────────────────────────────────
# Language purge (tools/purge_languages.py, 2026-10-07: English + Hindi only)
# ─────────────────────────────────────────────────────────────────────────────

class TestLanguagePurge:

    def test_purge_keeps_clusters_consistent(self, conn, backend, database, tmp_path):
        import gzip
        import sys as _sys
        from tools import purge_languages

        base = unit(101)
        en1 = add_article(conn, domain="a.com", title="Election results announced")
        en2 = add_article(conn, domain="b.com", title="Poll results declared today")
        ta = add_article(conn, domain="c.com", title="தேர்தல் முடிவுகள் அறிவிப்பு", lang="ta")
        solo = add_article(conn, domain="d.com", title="சென்னையில் கனமழை", lang="ta")
        for aid, v, d in ((en1, base, "a.com"), (en2, at_cos(base, 0.97, 5), "b.com"),
                          (ta, at_cos(base, 0.96, 6), "c.com")):
            assign(backend, aid, v, domain=d)
        assign(backend, solo, unit(202), domain="d.com")
        conn.commit()
        cid = conn.execute("SELECT cluster_id FROM articles WHERE id = %s", (en1,)).fetchone()[0]
        solo_cid = conn.execute("SELECT cluster_id FROM articles WHERE id = %s", (solo,)).fetchone()[0]
        before = cluster(conn, cid)
        assert before["article_count"] == 3

        _sys.argv = ["purge", "--dsn", database, "--backup", str(tmp_path / "bk")]
        assert purge_languages.main() == 0
        conn.rollback()

        left = {r[0] for r in conn.execute("SELECT id FROM articles").fetchall()}
        assert left == {en1, en2}
        after = cluster(conn, cid)
        assert after["article_count"] == 2
        assert sorted(after["outlet_set"]) == ["a.com", "b.com"] and after["outlet_count"] == 2
        s0 = np.array(eval(str(before["centroid_sum"])), dtype=np.float64)
        s1 = np.array(eval(str(after["centroid_sum"])), dtype=np.float64)
        assert np.allclose(s1, s0 * 2 / 3, atol=1e-4)                       # same average, consistent sum
        assert after["updated_at"] > before["updated_at"]                    # delta sync picks it up
        assert cluster(conn, solo_cid) is None                               # emptied cluster removed
        with gzip.open(tmp_path / "bk" / "articles.jsonl.gz", "rt", encoding="utf-8") as f:
            assert sorted(json.loads(line)["id"] for line in f) == sorted([ta, solo])
