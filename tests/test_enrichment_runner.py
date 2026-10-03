"""
tests/test_enrichment_runner.py
═══════════════════════════════
Behaviour of the Layer 2 runner and its DB layer — fully offline. Every NLP
step, the embedding model and the Supabase client are replaced with fakes, so
these tests pin down orchestration semantics:

  - work comes from the claim queue (dry runs / --force never claim)
  - each batch is processed oldest → newest
  - persistence order: entities → cluster → enrichment (enriched_at LAST)
  - per-step failures never lose the article; a crash is retried and parked
    only on its final attempt
  - time budget and signals release unprocessed claims
  - integer article ids (bigint) are handled everywhere (regression)
"""

from __future__ import annotations

import signal

import pytest

from enrichment import db, runner
from enrichment.clustering import ClusterResult


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────

class FakeDB:
    def __init__(self, articles=None):
        self.articles = articles or []
        self.calls: list[tuple] = []
        self.saved: dict = {}
        self.released: list = []
        self.run_rows: list = []

    # queue
    def queue_depth(self):
        return len(self.articles)

    def claim_batch(self, limit):
        self.calls.append(("claim", limit))
        return list(self.articles[:limit])

    def fetch_unenriched_batch(self, limit, offset=0):
        self.calls.append(("fetch", limit, offset))
        return list(self.articles[:limit])

    def fetch_unenriched_batch_forced(self, limit, offset=0):
        self.calls.append(("fetch_forced", limit, offset))
        return list(self.articles[:limit])

    def release_claims(self, ids):
        self.released.extend(ids)
        return len(ids)

    # writes
    def save_entities(self, article_id, entities):
        self.calls.append(("entities", article_id))
        return True

    def save_article_enrichment(self, article_id, update):
        self.calls.append(("save", article_id))
        self.saved[article_id] = dict(update)
        return True

    def mark_enrichment_failed(self, article_id, error):
        self.calls.append(("dead_letter", article_id, error))
        return True

    def log_run_start(self, row):
        self.run_rows.append(("start", row))
        return "run-1"

    def log_run_finish(self, run_id, row):
        if run_id:                      # same contract as enrichment.db
            self.run_rows.append(("finish", run_id, row))


def _article(i, published):
    return {"id": i, "title": f"Title {i}", "description": "d", "full_text": "word " * 80,
            "published_at": published, "domain": f"site{i}.com", "language_code": "en",
            "enrich_attempts": 1}


@pytest.fixture
def fake(monkeypatch):
    fdb = FakeDB([
        _article(1, "2026-10-03T09:00:00+00:00"),
        _article(2, "2026-10-03T07:00:00+00:00"),
        _article(3, "2026-10-03T08:00:00+00:00"),
    ])
    for name in ("queue_depth", "claim_batch", "fetch_unenriched_batch", "fetch_unenriched_batch_forced",
                 "release_claims", "save_entities", "save_article_enrichment", "mark_enrichment_failed",
                 "log_run_start", "log_run_finish"):
        monkeypatch.setattr(runner.db, name, getattr(fdb, name))

    # Cheap deterministic stand-ins for every model-backed step.
    monkeypatch.setattr(runner, "detect_language", lambda *a: "en")
    monkeypatch.setattr(runner, "analyse_sentiment", lambda *a: {"sentiment": "neutral", "sentiment_score": 0.0})
    monkeypatch.setattr(runner, "extract_entities",
                        lambda *a: [{"entity_text": "Modi", "entity_type": "PERSON", "salience": 0.9}])
    monkeypatch.setattr(runner, "extract_keywords", lambda *a: ["k"])
    monkeypatch.setattr(runner, "classify_article", lambda *a: "politics")
    monkeypatch.setattr(runner, "classify_tags", lambda *a: ["government"])
    monkeypatch.setattr(runner, "download_and_hash_image", lambda *a: 99)
    monkeypatch.setattr(runner, "embed_texts", lambda texts: [[0.1] * 768 for _ in texts])

    actions = iter(["seed", "join", "gray_join"])

    def fake_assign(article, embedding, entities, image_phash, language):
        fdb.calls.append(("cluster", article["id"], image_phash, language))
        return ClusterResult(f"c{article['id']}", next(actions), 0.9, 2, 2)

    monkeypatch.setattr(runner, "assign_cluster", fake_assign)
    return fdb


# ─────────────────────────────────────────────────────────────────────────────
# Batch semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestRunEnrichment:

    def test_claims_from_queue_and_processes_oldest_first(self, fake):
        summary = runner.run_enrichment(batch_size=10)
        assert fake.calls[0] == ("claim", 10)
        saves = [c[1] for c in fake.calls if c[0] == "save"]
        assert saves == [2, 3, 1]                     # published 07:00, 08:00, 09:00
        assert summary["processed"] == 3 and summary["failed"] == 0
        assert (summary["cluster_created"], summary["cluster_joined"], summary["cluster_gray_joined"]) == (1, 1, 1)
        assert summary["stop_reason"] == "drained" and fake.released == []

    def test_persistence_order_enriched_at_last(self, fake):
        runner.run_enrichment(batch_size=1)
        kinds = [c[0] for c in fake.calls if c[0] in ("entities", "cluster", "save")]
        assert kinds == ["entities", "cluster", "save"]

    def test_cluster_receives_image_hash_and_language(self, fake):
        runner.run_enrichment(batch_size=1)
        assert ("cluster", 1, 99, "en") in fake.calls

    def test_run_is_logged(self, fake):
        runner.run_enrichment(batch_size=10)
        start, finish = fake.run_rows
        assert start[1]["queue_depth_start"] == 3
        assert finish[1] == "run-1" and finish[2]["processed"] == 3 and finish[2]["claimed"] == 3

    def test_dry_run_never_claims_or_writes(self, fake):
        summary = runner.run_enrichment(batch_size=10, dry_run=True)
        assert ("fetch", 10, 0) in fake.calls
        assert not [c for c in fake.calls if c[0] in ("claim", "save", "entities", "cluster")]
        assert fake.run_rows == [] and summary["processed"] == 3

    def test_force_mode_reads_without_claiming(self, fake):
        runner.run_enrichment(batch_size=10, force=True, offset=5)
        assert ("fetch_forced", 10, 5) in fake.calls
        assert not [c for c in fake.calls if c[0] == "claim"]

    def test_empty_queue(self, fake):
        fake.articles = []
        summary = runner.run_enrichment(batch_size=10)
        assert summary["claimed"] == 0 and summary["processed"] == 0

    def test_clustering_failure_does_not_lose_enrichment(self, fake, monkeypatch):
        monkeypatch.setattr(runner, "assign_cluster", lambda *a, **k: None)
        summary = runner.run_enrichment(batch_size=10)
        assert summary["processed"] == 3 and summary["cluster_skipped"] == 3

    def test_save_failure_counts_as_failed(self, fake, monkeypatch):
        monkeypatch.setattr(runner.db, "save_article_enrichment", lambda aid, upd: False)
        summary = runner.run_enrichment(batch_size=10)
        assert summary["processed"] == 0 and summary["failed"] == 3


class TestAlreadyClustered:

    def test_articles_placed_by_clustering_job_skip_embedding_and_rpc(self, fake, monkeypatch):
        fake.articles[0]["cluster_id"] = "c-existing"
        embedded = []
        monkeypatch.setattr(runner, "embed_texts", lambda texts: embedded.extend(texts) or [[0.1] * 768 for _ in texts])
        summary = runner.run_enrichment(batch_size=10)
        assert len(embedded) == 2                                   # article 1 not re-embedded
        assert not [c for c in fake.calls if c[0] == "cluster" and c[1] == 1]
        assert summary["cluster_existing"] == 1 and summary["cluster_skipped"] == 0
        assert summary["processed"] == 3


class TestDbIdleReconnect:

    def test_client_recreated_after_idle_gap(self, monkeypatch):
        created = []
        monkeypatch.setattr(db, "create_client", lambda *a, **k: created.append(1) or object())
        monkeypatch.setattr(db, "SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setattr(db, "SUPABASE_KEY", "key")
        clock = {"t": 1000.0}
        monkeypatch.setattr(db.time, "monotonic", lambda: clock["t"])
        monkeypatch.setattr(db, "_client", None)
        monkeypatch.setattr(db, "_last_used", 0.0)
        db.get_client()
        clock["t"] += 10
        db.get_client()                                             # busy: reuse
        assert len(created) == 1
        clock["t"] += db._IDLE_RESET_SECONDS + 1
        db.get_client()                                             # idle: fresh connection
        assert len(created) == 2


class TestCrashHandling:

    def test_crash_before_final_attempt_is_left_for_retry(self, fake, monkeypatch):
        def boom(article):
            raise RuntimeError("segfault-ish")
        monkeypatch.setattr(runner, "enrich_one", boom)
        summary = runner.run_enrichment(batch_size=1)
        assert summary["failed"] == 1
        assert not [c for c in fake.calls if c[0] in ("dead_letter", "save")]

    def test_crash_on_final_attempt_is_parked(self, fake, monkeypatch):
        fake.articles[0]["enrich_attempts"] = 3
        monkeypatch.setattr(runner, "enrich_one", lambda a: (_ for _ in ()).throw(ValueError("bad html")))
        runner.run_enrichment(batch_size=1)
        dead = [c for c in fake.calls if c[0] == "dead_letter"]
        assert dead == [("dead_letter", 1, "ValueError: bad html")]

    def test_crash_mid_batch_releases_remaining_claims(self, fake, monkeypatch):
        calls = {"n": 0}

        def save(aid, upd):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt      # escapes the per-article handling
            return True

        monkeypatch.setattr(runner.db, "save_article_enrichment", save)
        with pytest.raises(KeyboardInterrupt):
            runner.run_enrichment(batch_size=10)
        assert fake.released == [3, 1]       # the article in flight and the one after it
        assert fake.run_rows[-1][2]["stop_reason"] == "signal"

    def test_unexpected_error_mid_batch_is_recorded(self, fake, monkeypatch):
        def explode(*a, **k):
            raise RuntimeError("disk full")
        monkeypatch.setattr(runner, "_process_article", explode)
        with pytest.raises(RuntimeError):
            runner.run_enrichment(batch_size=10)
        assert sorted(fake.released) == [1, 2, 3]
        finish = fake.run_rows[-1][2]
        assert finish["stop_reason"] == "error" and finish["error"] == "RuntimeError: disk full"


class TestStopping:

    def test_time_budget_releases_unprocessed_claims(self, fake, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr(runner.time, "perf_counter", lambda: clock["t"])
        original = runner._process_article

        def slow_article(*a, **k):
            original(*a, **k)
            clock["t"] += 120.0          # each article "takes" 2 minutes

        monkeypatch.setattr(runner, "_process_article", slow_article)
        summary = runner.run_enrichment(batch_size=10, time_budget_minutes=1)
        assert summary["stop_reason"] == "time_budget"
        assert summary["processed"] == 1
        assert sorted(fake.released) == [1, 3]

    def test_signal_stops_after_current_article(self, fake, monkeypatch):
        original = runner._process_article

        def process_then_signal(*a, **k):
            original(*a, **k)
            signal.raise_signal(signal.SIGINT)

        monkeypatch.setattr(runner, "_process_article", process_then_signal)
        summary = runner.run_enrichment(batch_size=10)
        assert summary["stop_reason"] == "signal" and summary["processed"] == 1
        assert sorted(fake.released) == [1, 3]
        # handlers restored: a later SIGINT is a KeyboardInterrupt again
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# ─────────────────────────────────────────────────────────────────────────────
# enrich_one
# ─────────────────────────────────────────────────────────────────────────────

class TestEnrichOne:

    def test_bigint_id_with_failing_step_does_not_crash(self, fake, monkeypatch):
        """Regression: log lines used article_id[:8] — TypeError on an int id."""
        def broken(*a):
            raise RuntimeError("model missing")
        monkeypatch.setattr(runner, "classify_article", broken)
        update, entities = runner.enrich_one(_article(123456789, "2026-10-03T00:00:00+00:00"))
        assert "category" not in update and update["keywords"] == ["k"]

    def test_short_article_skips_nlp(self, fake):
        art = _article(5, None)
        art["full_text"] = "too short"
        art["description"] = ""
        update, entities = runner.enrich_one(art)
        assert set(update) == {"word_count", "reading_time_mins"} and entities == []

    def test_unsupported_language_skips_rich_steps(self, fake, monkeypatch):
        monkeypatch.setattr(runner, "detect_language", lambda *a: "ta")
        update, entities = runner.enrich_one(_article(6, None))
        assert update["language_detected"] == "ta"
        assert "sentiment" not in update and entities == []
        assert update["category"] == "politics"          # multilingual classifier still runs


# ─────────────────────────────────────────────────────────────────────────────
# enrichment.db — queue + clustering wrappers over a fake Supabase client
# ─────────────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, data):
        self.data = data


class _RPC:
    def __init__(self, client, name, params):
        self.client, self.name, self.params = client, name, params

    def execute(self):
        self.client.rpc_calls.append((self.name, self.params))
        result = self.client.responses[self.name]
        if isinstance(result, Exception):
            raise result
        return _Resp(result(self.params) if callable(result) else result)


class _Client:
    def __init__(self, responses):
        self.responses = responses
        self.rpc_calls = []

    def rpc(self, name, params):
        return _RPC(self, name, params)


@pytest.fixture
def client(monkeypatch):
    def install(responses):
        c = _Client(responses)
        monkeypatch.setattr(db, "get_client", lambda: c)
        monkeypatch.setattr(db.time, "sleep", lambda s: None)
        return c
    return install


class TestDbQueue:

    def test_claim_pages_under_postgrest_cap(self, client):
        def claim(p):
            return [{"id": i, "published_at": f"2026-10-03T0{i % 10}:00"} for i in range(p["p_limit"])]
        c = client({"wizer_claim_enrichment_batch": claim})
        rows = db.claim_batch(1200)
        assert [p["p_limit"] for _, p in c.rpc_calls] == [500, 500, 200]
        assert len(rows) == 1200
        assert rows[0]["published_at"] >= rows[-1]["published_at"]

    def test_claim_stops_on_short_page(self, client):
        c = client({"wizer_claim_enrichment_batch": [{"id": 1, "published_at": "x"}]})
        assert len(db.claim_batch(1000)) == 1 and len(c.rpc_calls) == 1

    def test_claim_failure_raises(self, client):
        client({"wizer_claim_enrichment_batch": RuntimeError("function does not exist")})
        with pytest.raises(RuntimeError):
            db.claim_batch(10)

    def test_release_chunks_and_tolerates_failure(self, client):
        c = client({"wizer_release_enrichment_claims": lambda p: len(p["p_ids"])})
        assert db.release_claims(list(range(2500))) == 2500
        assert [len(p["p_ids"]) for _, p in c.rpc_calls] == [1000, 1000, 500]
        client({"wizer_release_enrichment_claims": RuntimeError("down")})
        assert db.release_claims([1]) == 0

    def test_assign_cluster_returns_row_or_none(self, client):
        client({"wizer_assign_cluster": [{"cluster_id": "c", "action": "seed"}]})
        assert db.assign_cluster({"p_article_id": 1})["action"] == "seed"
        client({"wizer_assign_cluster": RuntimeError("boom")})
        assert db.assign_cluster({"p_article_id": 1}) is None

    def test_assign_cluster_retries_once_on_timeout(self, client):
        import httpx
        attempts = []

        def flaky(p):
            attempts.append(1)
            if len(attempts) == 1:
                raise httpx.ReadTimeout("idle connection dropped")
            return [{"cluster_id": "c", "action": "existing"}]

        client({"wizer_assign_cluster": flaky})
        assert db.assign_cluster({"p_article_id": 1})["action"] == "existing"
        assert len(attempts) == 2

    def test_queue_depth_failure_is_none(self, client):
        client({"wizer_enrichment_queue_depth": RuntimeError("down")})
        assert db.queue_depth() is None



class TestEntityLookup:

    def test_chunks_of_33_without_order_or_paging(self, monkeypatch):
        """Production incident: IN(200 ids) + ORDER BY + LIMIT walked the whole index."""
        calls = []

        class Q:
            def __init__(self):
                self.ids = None

            def select(self, *a):
                return self

            def in_(self, col, ids):
                self.ids = list(ids)
                return self

            def order(self, *a, **k):
                raise AssertionError("no ORDER BY: it defeats the article_id index")

            def range(self, *a):
                raise AssertionError("no paging needed: a chunk is < 1000 rows")

            def execute(self):
                calls.append(len(self.ids))
                return type("R", (), {"data": [{"article_id": i, "entity_text": "X"} for i in self.ids]})()

        monkeypatch.setattr(db, "get_client", lambda: type("C", (), {"table": lambda self, n: Q()})())
        out = db.fetch_entities_for_articles(list(range(100)))
        assert calls == [33, 33, 33, 1]
        assert len(out) == 100 and out[5] == [{"article_id": 5, "entity_text": "X"}]
