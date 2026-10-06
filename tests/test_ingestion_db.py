"""
tests/test_ingestion_db.py
══════════════════════════
Regression tests for pipeline/db.py and pipeline/circuit_breaker.py
(items 2, 3, 4, 5, 13, 14, S1).  A fake Supabase client is injected, so these
run fully offline.
"""

from datetime import datetime, timezone, timedelta

import pytest

from pipeline import db
from pipeline.circuit_breaker import (
    check_dormancy, get_last_new_article_date, has_new_article_tracking,
    should_skip_feed, dormant_recheck_due,
)


# ─────────────────────────────────────────────────────────────────────────────
# FAKE SUPABASE CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class FakeResp:
    def __init__(self, data=None, count=None):
        self.data = data
        self.count = count


class FakeQuery:
    """Records every builder call; `execute()` delegates to the client handler."""

    def __init__(self, client, table):
        self.client = client
        self.table = table
        self.ops: list[tuple] = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            self.ops.append((name, args, kwargs))
            return self
        return method

    @property
    def not_(self):          # supabase's `.not_.is_(...)`
        return self

    def op(self, name):
        return [o for o in self.ops if o[0] == name]

    def execute(self):
        self.client.queries.append(self)
        return self.client.handler(self)


class FakeClient:
    def __init__(self, handler):
        self.handler = handler
        self.queries: list[FakeQuery] = []

    def table(self, name):
        return FakeQuery(self, name)


@pytest.fixture
def install(monkeypatch):
    def _install(handler):
        client = FakeClient(handler)
        monkeypatch.setattr(db, "get_client", lambda: client)
        return client
    return _install


def _iso(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# S1: PAGINATION
# ─────────────────────────────────────────────────────────────────────────────

def _range_of(q):
    (_, args, _), = q.op("range")
    return args


def _paged_handler(rows):
    """Behaves like PostgREST with max_rows=1000 and honours .range()."""
    def handler(q):
        start, end = _range_of(q)
        return FakeResp(rows[start:min(end, start + 999) + 1])
    return handler


class TestPagination:

    def test_get_due_feeds_reads_all_pages(self, install):
        rows = [{"id": str(i), "update_cadence": "daily", "last_polled_at": None}
                for i in range(2500)]
        client = install(_paged_handler(rows))
        # active query returns all 2500, dormant query (is_active=False) too —
        # distinguish by the eq filter
        def handler(q):
            if any(a == ("is_active", False) for _, a, _ in q.op("eq")):
                return FakeResp([])
            return _paged_handler(rows)(q)
        client.handler = handler
        due = db.get_due_feeds("daily")
        assert len(due) == 2500
        ranges = [_range_of(q) for q in client.queries
                  if not any(a == ("is_active", False) for _, a, _ in q.op("eq"))]
        assert ranges == [(0, 999), (1000, 1999), (2000, 2999)]

    def test_paginate_respects_limit(self, install):
        rows = [{"title_simhash": i} for i in range(5000)]
        client = install(_paged_handler(rows))
        out = db.load_recent_simhashes(limit=2500)
        assert len(out) == 2500
        assert [_range_of(q) for q in client.queries] == [(0, 999), (1000, 1999), (2000, 2499)]

    def test_simhashes_default_limit_gets_more_than_one_page(self, install):
        rows = [{"title_simhash": i} for i in range(10000)]
        install(_paged_handler(rows))
        assert len(db.load_recent_simhashes()) == 10000

    def test_load_recent_hashes_paginates(self, install):
        rows = [{"url_hash": i} for i in range(2000)]
        install(_paged_handler(rows))
        assert len(db.load_recent_hashes("f1", limit=2000)) == 2000


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 5: DB OUTAGE MUST NOT LOOK LIKE "NO FEEDS DUE"
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDueFeedsErrors:

    def test_db_error_raises(self, install):
        def boom(q):
            raise ConnectionError("supabase down")
        install(boom)
        with pytest.raises(db.FeedLoadError):
            db.get_due_feeds("daily")

    def test_missing_optional_columns_degrade(self, install):
        calls = []

        def handler(q):
            cols = q.op("select")[0][1][0]
            calls.append(cols)
            if any(a == ("is_active", False) for _, a, _ in q.op("eq")):
                return FakeResp([])
            if "last_new_article_at" in cols:
                raise Exception("{'code': '42703', 'message': 'column feeds.last_new_article_at does not exist'}")
            return FakeResp([{"id": "1", "last_polled_at": None}])
        install(handler)
        due = db.get_due_feeds()
        assert [f["id"] for f in due] == ["1"]
        assert "last_new_article_at" not in due[0]
        assert not has_new_article_tracking(due[0])


# ─────────────────────────────────────────────────────────────────────────────
# ITEMS 2 + 13: SELECTED COLUMNS, LANGUAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectedColumns:

    def test_created_at_and_dormancy_columns_selected(self, install):
        client = install(lambda q: FakeResp([]))
        db.get_due_feeds("daily")
        cols = client.queries[0].op("select")[0][1][0]
        for c in ("created_at", "last_new_article_at", "disabled_reason", "language_code"):
            assert c in cols

    def test_language_name_not_selected_but_derived(self, install):
        def handler(q):
            if any(a == ("is_active", False) for _, a, _ in q.op("eq")):
                return FakeResp([])
            return FakeResp([{"id": "1", "language_code": "hi", "last_polled_at": None},
                             {"id": "2", "language_code": "xx-YY", "last_polled_at": None},
                             {"id": "3", "language_code": "en-IN", "last_polled_at": None}])
        client = install(handler)
        due = {f["id"]: f for f in db.get_due_feeds()}
        assert "language_name" not in client.queries[0].op("select")[0][1][0]
        # Production's articles.language is char(5): the short code, never a name
        # (names made every insert fail with 22001 on 2026-10-03).
        assert due["1"]["language_name"] == "hi"
        assert due["2"]["language_name"] == "xx-yy"
        assert due["3"]["language_name"] == "en-in"
        assert all(len(f["language_name"]) <= 5 for f in due.values())


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 3: DUE TOLERANCE
# ─────────────────────────────────────────────────────────────────────────────

class TestIsFeedDue:

    NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def _feed(self, minutes_ago, **kw):
        return {"last_polled_at": (self.NOW - timedelta(minutes=minutes_ago)).isoformat(),
                "update_cadence": "breaking_news", **kw}

    def test_hourly_feed_polled_55_min_ago_is_due(self):
        """Cron fires hourly; the feed finished ~5 min into the previous run.
        With a strict >= interval check this was skipped (2-interval polling)."""
        assert db._is_feed_due(self._feed(55), self.NOW)

    def test_not_due_when_far_from_interval(self):
        assert not db._is_feed_due(self._feed(20), self.NOW)

    def test_tolerance_boundary(self):
        assert not db._is_feed_due(self._feed(50), self.NOW)    # 0.9 * 60 = 54
        assert db._is_feed_due(self._feed(54.5), self.NOW)

    def test_never_polled_and_unparseable(self):
        assert db._is_feed_due({"last_polled_at": None}, self.NOW)
        assert db._is_feed_due({"last_polled_at": "garbage"}, self.NOW)

    def test_db_interval_overrides_cadence(self):
        assert not db._is_feed_due(self._feed(100, poll_interval_mins=1440), self.NOW)

    def test_naive_timestamp_treated_as_utc(self):
        f = {"last_polled_at": "2026-01-01T10:00:00", "update_cadence": "daily"}
        assert not db._is_feed_due(f, self.NOW)


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 2: DORMANCY
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateFeedAfterPoll:

    def _patch_of(self, client):
        upd = [q for q in client.queries if q.op("update")]
        return upd[-1].op("update")[0][1][0]

    def test_zero_new_does_not_advance_last_new_article_at(self, install):
        client = install(lambda q: FakeResp([{"fail_count": 0, "articles_found": 5}]))
        db.update_feed_after_poll("f1", True, 0, "")
        patch = self._patch_of(client)
        assert "last_success_at" in patch and "last_polled_at" in patch
        assert "last_new_article_at" not in patch

    def test_new_articles_advance_last_new_article_at(self, install):
        client = install(lambda q: FakeResp({"fail_count": 0, "articles_found": 5}))
        db.update_feed_after_poll("f1", True, 3, "")
        patch = self._patch_of(client)
        assert patch["last_new_article_at"] == patch["last_success_at"]
        assert patch["articles_found"] == 8

    def test_failure_updates_last_polled_and_fail_count(self, install):
        client = install(lambda q: FakeResp({"fail_count": 1}))
        db.update_feed_after_poll("f1", False, 0, "timeout")
        patch = self._patch_of(client)
        assert patch["fail_count"] == 2
        assert "last_polled_at" in patch
        assert "last_success_at" not in patch

    def test_error_disable_sets_reason(self, install):
        client = install(lambda q: FakeResp({"fail_count": 4}))
        db.update_feed_after_poll("f1", False, 0, "boom")
        patch = self._patch_of(client)
        assert patch["is_active"] is False
        assert patch["disabled_reason"] == "errors"

    def test_reactivate_dormant_feed(self, install):
        client = install(lambda q: FakeResp({"fail_count": 0, "articles_found": 0}))
        db.update_feed_after_poll("f1", True, 2, "", reactivate=True)
        patch = self._patch_of(client)
        assert patch["is_active"] is True
        assert patch["disabled_reason"] is None

    def test_missing_optional_column_retries_without_it(self, install):
        seen = []

        def handler(q):
            if q.op("update"):
                patch = q.op("update")[0][1][0]
                seen.append(dict(patch))
                if "last_new_article_at" in patch:
                    raise Exception("PGRST204 Could not find the 'last_new_article_at' column")
            return FakeResp({"fail_count": 0, "articles_found": 0})
        install(handler)
        db.update_feed_after_poll("f1", True, 1, "")
        assert len(seen) == 2
        assert "last_new_article_at" not in seen[1]
        assert "last_polled_at" in seen[1]

    def test_mark_dormant_sets_reason(self, install):
        client = install(lambda q: FakeResp([]))
        db.mark_feed_dormant("f1", "stale")
        patch = self._patch_of(client)
        assert patch == {"is_active": False, "disabled_reason": "dormant"}


class TestCircuitBreakerDormancy:

    def test_last_new_article_uses_dedicated_column_not_last_success(self):
        feed = {"last_success_at": _iso(minutes=1), "last_new_article_at": _iso(days=45)}
        last = get_last_new_article_date(feed)
        assert (datetime.now(timezone.utc) - last).days == 45
        dormant, _ = check_dormancy(feed, last)
        assert dormant

    def test_last_success_alone_is_ignored(self):
        assert get_last_new_article_date({"last_success_at": _iso(minutes=1)}) is None

    def test_created_at_fallback_when_never_produced(self):
        feed = {"last_new_article_at": None, "created_at": _iso(days=40)}
        dormant, _ = check_dormancy(feed, get_last_new_article_date(feed))
        assert dormant

    def test_naive_created_at_ok(self):
        naive = (datetime.now(timezone.utc) - timedelta(days=40)).replace(tzinfo=None).isoformat()
        dormant, _ = check_dormancy({"created_at": naive}, None)
        assert dormant

    def test_tracking_flag(self):
        assert has_new_article_tracking({"last_new_article_at": None})
        assert not has_new_article_tracking({})


class TestDormantRecheck:

    def _handler(self, active, dormant):
        def handler(q):
            if any(a == ("is_active", False) for _, a, _ in q.op("eq")):
                return FakeResp(dormant)
            return FakeResp(active)
        return handler

    def test_dormant_feed_due_after_interval_is_included_and_tagged(self, install):
        dormant = [
            {"id": "old", "is_active": False, "disabled_reason": "dormant",
             "last_polled_at": _iso(days=8)},
            {"id": "recent", "is_active": False, "disabled_reason": "dormant",
             "last_polled_at": _iso(days=2)},
        ]
        client = install(self._handler([], dormant))
        due = db.get_due_feeds()
        assert [f["id"] for f in due] == ["old"]
        assert due[0]["_dormant_recheck"] is True
        # the query must filter on disabled_reason='dormant' (not error-disabled)
        dq = [q for q in client.queries
              if any(a == ("is_active", False) for _, a, _ in q.op("eq"))][0]
        assert ("disabled_reason", "dormant") in [a for _, a, _ in dq.op("eq")]

    def test_dormant_feed_never_polled_is_included(self, install):
        install(self._handler([], [{"id": "x", "is_active": False,
                                    "disabled_reason": "dormant", "last_polled_at": None}]))
        assert [f["id"] for f in db.get_due_feeds()] == ["x"]

    def test_dormant_query_failure_does_not_break_active_polling(self, install):
        def handler(q):
            if any(a == ("is_active", False) for _, a, _ in q.op("eq")):
                raise Exception("boom")
            return FakeResp([{"id": "a", "last_polled_at": None}])
        install(handler)
        assert [f["id"] for f in db.get_due_feeds()] == ["a"]

    def test_should_skip_distinguishes_states(self):
        dormant = {"is_active": False, "disabled_reason": "dormant"}
        errored = {"is_active": False, "disabled_reason": "errors"}
        assert should_skip_feed(dormant)[0]                       # not tagged: skipped
        assert not should_skip_feed({**dormant, "_dormant_recheck": True})[0]
        assert should_skip_feed({**errored, "_dormant_recheck": True})[0]

    def test_dormant_recheck_due_helper(self):
        assert dormant_recheck_due({"last_polled_at": _iso(days=8)})
        assert not dormant_recheck_due({"last_polled_at": _iso(days=1)})
        assert dormant_recheck_due({"last_polled_at": None})


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 4: pipeline_runs tier/cadence
# ─────────────────────────────────────────────────────────────────────────────

class TestRunLogging:

    def test_sends_both_columns_when_present(self, install):
        client = install(lambda q: FakeResp([{"id": "run1"}]))
        assert db.log_run_start("daily", False) == "run1"
        row = client.queries[0].op("insert")[0][1][0]
        assert row["tier"] == "daily" and row["cadence"] == "daily"

    @pytest.mark.parametrize("missing", ["tier", "cadence"])
    def test_start_retries_without_missing_column(self, install, missing):
        def handler(q):
            row = q.op("insert")[0][1][0]
            if missing in row:
                raise Exception(f"PGRST204 Could not find the '{missing}' column of 'pipeline_runs'")
            return FakeResp([{"id": "run2"}])
        client = install(handler)
        assert db.log_run_start("daily", False) == "run2"
        assert missing not in client.queries[-1].op("insert")[0][1][0]

    def test_start_survives_both_columns_missing(self, install):
        def handler(q):
            row = q.op("insert")[0][1][0]
            for c in ("tier", "cadence"):
                if c in row:
                    raise Exception(f"PGRST204 Could not find the '{c}' column")
            return FakeResp([{"id": "run3"}])
        install(handler)
        assert db.log_run_start(None, True) == "run3"

    def test_finish_retries_without_tier(self, install):
        def handler(q):
            row = q.op("update")[0][1][0]
            if "tier" in row:
                raise Exception("PGRST204 Could not find the 'tier' column")
            return FakeResp([])
        client = install(handler)
        db.log_run_finish("run1", {"cadence": "daily", "new_articles": 3})
        last = client.queries[-1].op("update")[0][1][0]
        assert "tier" not in last and last["new_articles"] == 3

    def test_unrelated_error_is_not_retried_forever(self, install):
        client = install(lambda q: (_ for _ in ()).throw(Exception("network down")))
        assert db.log_run_start("daily", False) is None
        assert len(client.queries) == 1


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 14: UPSERT COUNTING
# ─────────────────────────────────────────────────────────────────────────────

class TestUpsertCounting:

    def test_fallback_counts_conflict_skips_as_duplicates(self, install):
        calls = {"n": 0}

        def handler(q):
            calls["n"] += 1
            rows = q.op("upsert")[0][1][0]
            if isinstance(rows, list):
                raise Exception("batch failed")            # force row-by-row
            return FakeResp([rows] if rows["url_hash"] != 2 else [])   # hash 2 = conflict skip
        install(handler)
        rows = [{"url_hash": 1, "url": "a"}, {"url_hash": 2, "url": "b"}, {"url_hash": 3, "url": "c"}]
        assert db.upsert_articles(rows) == (2, 1)

    def test_fallback_failed_rows_are_not_duplicates(self, install):
        def handler(q):
            rows = q.op("upsert")[0][1][0]
            if isinstance(rows, list) or rows["url_hash"] == 2:
                raise Exception("bad row")
            return FakeResp([rows])
        install(handler)
        rows = [{"url_hash": 1, "url": "a"}, {"url_hash": 2, "url": "b"}]
        inserted, dups = db.upsert_articles(rows)
        assert (inserted, dups) == (1, 0)
        assert len(rows) - inserted - dups == 1             # one failure, derivable

    def test_timeout_does_not_retry_row_by_row(self, install):
        """An overloaded DB must not be hit once per row (2026-10-05 smoke test)."""
        calls = {"n": 0}

        def handler(q):
            calls["n"] += 1
            raise Exception("The read operation timed out")
        install(handler)
        rows = [{"url_hash": i, "url": str(i)} for i in range(20)]
        assert db.upsert_articles(rows) == (0, 0)
        assert calls["n"] <= 2                              # the batch + its one reconnect retry

    def test_batch_path(self, install):
        install(lambda q: FakeResp([{"url_hash": 1}]))
        assert db.upsert_articles([{"url_hash": 1}, {"url_hash": 2}]) == (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 8: ONE .in_() QUERY FOR BOTH HASHES
# ─────────────────────────────────────────────────────────────────────────────

class TestUrlHashExists:

    def test_single_in_query_for_multiple_hashes(self, install):
        client = install(lambda q: FakeResp([], count=1))
        assert db.url_hash_exists([11, -22]) is True
        assert len(client.queries) == 1
        (_, args, _), = client.queries[0].op("in_")
        assert args[0] == "url_hash" and sorted(args[1]) == [-22, 11]

    def test_int_argument_still_supported(self, install):
        install(lambda q: FakeResp([], count=0))
        assert db.url_hash_exists(5) is False

    def test_fails_open(self, install):
        install(lambda q: (_ for _ in ()).throw(Exception("x")))
        assert db.url_hash_exists([1, 2]) is False



# ─────────────────────────────────────────────────────────────────────────────
# prune_articles_if_needed — loops the SQL function until the target is met
# ─────────────────────────────────────────────────────────────────────────────

class TestPruneArticles:

    def _client(self, monkeypatch, results):
        from pipeline import db as pdb
        calls = []

        class RPC:
            def __init__(self, params):
                self.params = params

            def execute(self):
                calls.append(self.params)
                r = results.pop(0)
                if isinstance(r, Exception):
                    raise r
                return type("R", (), {"data": r})()

        client = type("C", (), {"rpc": lambda self, name, params: RPC(params)})()
        monkeypatch.setattr(pdb, "get_client", lambda: client)
        monkeypatch.setattr(pdb, "ARTICLE_HARD_LIMIT", 2_000_000)
        monkeypatch.setattr(pdb, "ARTICLE_PRUNE_TARGET", 1_900_000)
        return pdb, calls

    def test_loops_until_short_chunk(self, monkeypatch):
        pdb, calls = self._client(monkeypatch, [50_000, 50_000, 1234])
        assert pdb.prune_articles_if_needed() == 101_234
        assert calls[0]["p_hard_limit"] == pdb.ARTICLE_HARD_LIMIT
        assert all(c["p_hard_limit"] == pdb.ARTICLE_PRUNE_TARGET for c in calls[1:])

    def test_disabled_by_default_never_calls_db(self, monkeypatch):
        from pipeline import db as pdb, config as pcfg
        assert pcfg.ARTICLE_HARD_LIMIT == 0          # safe default: no deletion
        monkeypatch.setattr(pdb, "get_client", lambda: (_ for _ in ()).throw(AssertionError("called DB")))
        monkeypatch.setattr(pdb, "ARTICLE_HARD_LIMIT", 0)
        assert pdb.prune_articles_if_needed() == 0

    def test_nothing_to_do(self, monkeypatch):
        pdb, calls = self._client(monkeypatch, [0])
        assert pdb.prune_articles_if_needed() == 0 and len(calls) == 1

    def test_failure_reports_partial_progress_without_raising(self, monkeypatch):
        pdb, _ = self._client(monkeypatch, [50_000, RuntimeError("timeout")])
        assert pdb.prune_articles_if_needed() == 50_000



# ─────────────────────────────────────────────────────────────────────────────
# Per-thread clients + transient-error retry (production incident 2026-10-03:
# ~30 threads sharing one client got "Server disconnected" on bursts)
# ─────────────────────────────────────────────────────────────────────────────

class TestClientPerThread:

    def test_each_thread_gets_its_own_client(self, monkeypatch):
        import threading
        import supabase
        made = []
        monkeypatch.setattr(db, "SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setattr(db, "SUPABASE_KEY", "k")
        monkeypatch.setattr(supabase, "create_client", lambda *a, **k: made.append(object()) or made[-1])
        db._local.client = None
        seen = []
        threads = [threading.Thread(target=lambda: seen.append(db.get_client())) for _ in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert len({id(c) for c in seen}) == 3
        assert db.get_client() is db.get_client()          # stable within a thread

    def test_transient_error_reconnects_and_retries_once(self, monkeypatch):
        import httpx
        monkeypatch.setattr(db.time, "sleep", lambda s: None)
        calls = []

        def op():
            calls.append(1)
            if len(calls) == 1:
                raise httpx.RemoteProtocolError("Server disconnected")
            return "ok"
        assert db._retry(op) == "ok" and len(calls) == 2

    def test_non_transient_error_is_not_retried(self):
        calls = []

        def op():
            calls.append(1)
            raise ValueError("22001 value too long")
        with pytest.raises(ValueError):
            db._retry(op)
        assert len(calls) == 1



class TestExistingHashes:

    def test_chunks_and_parses_padded_char_hashes(self, monkeypatch):
        """Production url_hash is char(32): values come back space-padded strings."""
        calls = []

        class Q:
            def select(self, *a):
                return self

            def in_(self, col, vals):
                self.vals = list(vals)
                calls.append(len(self.vals))
                return self

            def execute(self):
                return type("R", (), {"data": [{"url_hash": f"{v:<32}"} for v in self.vals if v % 2 == 0]})()

        monkeypatch.setattr(db, "get_client", lambda: type("C", (), {"table": lambda self, n: Q()})())
        found = db.existing_hashes(range(-10, 300))
        assert calls == [150, 150, 10]
        assert found == {v for v in range(-10, 300) if v % 2 == 0}

    def test_lookup_failure_fails_open(self, monkeypatch):
        monkeypatch.setattr(db, "get_client", lambda: (_ for _ in ()).throw(ValueError("down")))
        assert db.existing_hashes([1, 2]) == set()
