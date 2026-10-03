"""
tests/test_ingestion_poller.py
══════════════════════════════
Regression tests for pipeline/poller.py (items 1, 2, 7, 8, 14) and main.py
exit handling (item 5).

Network tests use a throw-away HTTP server bound to 127.0.0.1 (no external
traffic); Supabase is replaced by monkeypatched db functions.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pipeline import poller, db
from pipeline.poller import _download_feed, _fetch_rss_blocking, FeedFetchError
from pipeline.crawler import CrawledArticle


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TEST SERVER
# ─────────────────────────────────────────────────────────────────────────────

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>T</title><link>http://example.com/</link>
<item><title>One</title><link>/story/1</link></item>
<item><title>Two</title><link>/story/2</link></item>
</channel></rss>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/ok":
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.send_header("Content-Length", str(len(RSS)))
            self.end_headers()
            self.wfile.write(RSS)
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/feeds/real.xml")
            self.end_headers()
        elif self.path == "/feeds/real.xml":
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.end_headers()
            self.wfile.write(RSS)
        elif self.path == "/hang":               # accepts, then says nothing
            time.sleep(3)
        elif self.path == "/drip":               # headers, then 1 byte / 100 ms
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.end_headers()
            try:
                for _ in range(60):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass
        elif self.path == "/big":
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.end_headers()
            try:
                self.wfile.write(b"x" * (2 * 1024 * 1024))
            except OSError:
                pass
        elif self.path == "/404":
            self.send_error(404)
        else:
            self.send_error(500)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 1: BOUNDED DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadFeed:

    def test_ok(self, server):
        body, headers, final = _download_feed(server + "/ok", timeout=2, deadline=5)
        assert body == RSS and final.endswith("/ok")
        assert headers["content-type"].startswith("application/rss")

    def test_hung_server_times_out_quickly(self, server):
        t0 = time.monotonic()
        with pytest.raises(FeedFetchError, match="timed out"):
            _download_feed(server + "/hang", timeout=0.3, deadline=5)
        assert time.monotonic() - t0 < 2

    def test_slow_drip_cut_off_by_wallclock_deadline(self, server):
        """Each recv succeeds within the socket timeout, so only the overall
        deadline can stop this."""
        t0 = time.monotonic()
        with pytest.raises(FeedFetchError, match="deadline"):
            _download_feed(server + "/drip", timeout=2, deadline=0.5)
        assert time.monotonic() - t0 < 3        # would be ~6 s without the deadline

    def test_max_size_enforced(self, server):
        with pytest.raises(FeedFetchError, match="larger"):
            _download_feed(server + "/big", timeout=2, deadline=5, max_bytes=100_000)

    def test_http_error(self, server):
        with pytest.raises(FeedFetchError, match="HTTP 404"):
            _download_feed(server + "/404", timeout=2, deadline=5)

    def test_rejects_non_http_scheme(self):
        with pytest.raises(FeedFetchError, match="scheme"):
            _download_feed("file:///etc/passwd")

    def test_connection_refused(self):
        with pytest.raises(FeedFetchError):
            _download_feed("http://127.0.0.1:1/x", timeout=1, deadline=2)


class TestFetchRssBlocking:

    def test_parses_and_resolves_relative_links(self, server):
        entries, meta, err = _fetch_rss_blocking(server + "/ok")
        assert err is None and meta["title"] == "T"
        assert [e["link"] for e in entries] == [server + "/story/1", server + "/story/2"]

    def test_relative_links_use_final_url_after_redirect(self, server):
        entries, _, err = _fetch_rss_blocking(server + "/redirect")
        assert err is None and entries[0]["link"] == server + "/story/1"

    def test_failure_is_returned_not_raised(self, server):
        entries, _, err = _fetch_rss_blocking(server + "/404")
        assert entries == [] and "HTTP 404" in err

    def test_timeout_is_an_error_message(self, server, monkeypatch):
        monkeypatch.setattr(
            poller, "_download_feed",
            lambda url: _download_feed(url, timeout=0.2, deadline=1),
        )
        entries, _, err = _fetch_rss_blocking(server + "/hang")
        assert entries == [] and "fetch failed" in err

    def test_garbage_body_is_bozo_error(self, monkeypatch):
        monkeypatch.setattr(poller, "_download_feed",
                            lambda url: (b"<<<not xml", {}, url))
        entries, _, err = _fetch_rss_blocking("http://x/feed")
        assert entries == [] and err


# ─────────────────────────────────────────────────────────────────────────────
# poll_one_feed HARNESS
# ─────────────────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.feed_updates = []
        self.dormant = []
        self.upserts = []


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()
    monkeypatch.setattr(db, "existing_hashes", lambda hashes: set())
    monkeypatch.setattr(
        db, "update_feed_after_poll",
        lambda *a, **k: r.feed_updates.append((a, k)),
    )
    monkeypatch.setattr(db, "mark_feed_dormant", lambda fid, reason: r.dormant.append((fid, reason)))

    def upsert(rows):
        r.upserts.append(rows)
        return len(rows), 0
    monkeypatch.setattr(db, "upsert_articles", upsert)
    return r


def _poll(feed):
    async def go():
        loop = asyncio.get_running_loop()
        return await poller.poll_one_feed(
            feed, asyncio.Semaphore(2), asyncio.Semaphore(2), set(), loop
        )
    return asyncio.run(go())


def _feed(**kw):
    base = {"id": "f1", "feed_url": "http://f/x", "update_cadence": "daily",
            "is_active": True, "fail_count": 0,
            "last_new_article_at": None, "created_at": "2000-01-01T00:00:00+00:00"}
    base.update(kw)
    return base


def _entries(n=2):
    return [{"title": f"Title {i}", "link": f"https://s.example/a{i}"} for i in range(n)]


def _fake_crawl(entry, feed, norm, h, sh):
    return CrawledArticle(feed_id=feed["id"], url=norm, url_hash=h,
                          title=entry["title"], title_simhash=sh)


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 1: TIMEOUT RECORDED AS FAILURE; EXECUTOR SIZING
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeoutRecordedAsFailure:

    def test_hung_fetch_counts_as_failure(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "FEED_FETCH_DEADLINE_SECONDS", 0.05)
        monkeypatch.setattr(poller, "FEED_FETCH_TIMEOUT_SECONDS", 0.05)
        monkeypatch.setattr(poller, "_FETCH_GRACE_SECONDS", 0.05)
        monkeypatch.setattr(poller, "_fetch_rss_blocking",
                            lambda url: (time.sleep(0.6), ([], {}, None))[1])
        res = _poll(_feed())
        assert res["errors"] == 1 and res["new"] == 0
        (args, _), = rec.feed_updates
        assert args[:3] == ("f1", False, 0)            # success=False -> fail_count++
        assert "timed out" in args[3]

    def test_download_error_counts_as_failure(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking",
                            lambda url: ([], {}, "fetch failed: timed out after 20s"))
        res = _poll(_feed())
        assert res["errors"] == 1
        assert rec.feed_updates[0][0][1] is False


class TestExecutorSizing:

    def test_default_executor_sized_from_config(self, rec, monkeypatch):
        captured = {}
        monkeypatch.setattr(poller, "EXECUTOR_MAX_WORKERS", 17)
        monkeypatch.setattr(db, "log_run_start", lambda *a: "run")
        monkeypatch.setattr(poller, "log_run_start", lambda *a: "run")
        monkeypatch.setattr(poller, "log_run_finish", lambda *a: None)
        monkeypatch.setattr(db, "get_due_feeds", lambda c=None: [])

        async def go():
            loop = asyncio.get_running_loop()
            orig = loop.set_default_executor

            def spy(ex):
                captured["ex"] = ex
                orig(ex)
            loop.set_default_executor = spy
            await poller.run_pipeline("daily")
        asyncio.run(go())
        ex = captured["ex"]
        assert isinstance(ex, ThreadPoolExecutor) and ex._max_workers == 17

    def test_config_default_covers_concurrency(self):
        from pipeline import config
        assert config.EXECUTOR_MAX_WORKERS >= config.MAX_CONCURRENT_FEEDS + config.MAX_CONCURRENT_ARTICLES


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 2: DORMANCY
# ─────────────────────────────────────────────────────────────────────────────

class TestDormancyInPoller:

    def test_feed_that_just_inserted_is_never_marked_dormant(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (_entries(), {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        # pre-poll snapshot looks 60 days stale
        res = _poll(_feed(last_new_article_at="2000-01-01T00:00:00+00:00"))
        assert res["new"] == 2
        assert rec.dormant == []

    def test_empty_poll_of_stale_feed_is_marked_dormant(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: ([], {}, None))
        res = _poll(_feed(last_new_article_at="2000-01-01T00:00:00+00:00"))
        assert res["new"] == 0 and res["errors"] == 0
        assert [d[0] for d in rec.dormant] == ["f1"]

    def test_failed_inserts_never_mark_a_feed_dormant(self, rec, monkeypatch):
        """Incident 2026-10-03: every insert failed, so "0 new" made healthy feeds dormant."""
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (_entries(), {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        monkeypatch.setattr(db, "upsert_articles", lambda rows: (0, 0))   # all rows failed
        res = _poll(_feed(last_new_article_at="2000-01-01T00:00:00+00:00"))
        assert res["new"] == 0 and res["errors"] == 2
        assert rec.dormant == []

    def test_never_productive_old_feed_dormant_via_created_at(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: ([], {}, None))
        _poll(_feed())                                  # created_at = year 2000
        assert [d[0] for d in rec.dormant] == ["f1"]

    def test_no_dormancy_when_column_not_migrated(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: ([], {}, None))
        feed = _feed()
        del feed["last_new_article_at"]
        _poll(feed)
        assert rec.dormant == []

    def test_recent_feed_not_dormant(self, rec, monkeypatch):
        from datetime import datetime, timezone
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: ([], {}, None))
        _poll(_feed(last_new_article_at=datetime.now(timezone.utc).isoformat()))
        assert rec.dormant == []

    def test_dormant_recheck_with_article_reactivates(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (_entries(1), {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        feed = _feed(is_active=False, disabled_reason="dormant", _dormant_recheck=True)
        res = _poll(feed)
        assert res["skipped"] is False and res["new"] == 1
        (args, _), = rec.feed_updates
        assert args == ("f1", True, 1, "", True)        # reactivate=True

    def test_dormant_recheck_empty_stays_dormant_without_rechecking_dormancy(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: ([], {}, None))
        feed = _feed(is_active=False, disabled_reason="dormant", _dormant_recheck=True,
                     last_new_article_at="2000-01-01T00:00:00+00:00")
        _poll(feed)
        assert rec.feed_updates[0][0] == ("f1", True, 0, "", False)
        assert rec.dormant == []

    def test_error_disabled_feed_is_skipped_even_if_tagged(self, rec):
        res = _poll(_feed(is_active=False, disabled_reason="errors", _dormant_recheck=True))
        assert res["skipped"] is True and rec.feed_updates == []


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 7: ENTRY ERRORS ARE ERRORS, NOT DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryErrors:

    def test_exception_logged_at_warning_and_counted_as_error(self, rec, monkeypatch, caplog):
        def crawl(entry, feed, norm, h, sh):
            if entry["title"] == "Title 1":
                raise RuntimeError("kaboom")
            return _fake_crawl(entry, feed, norm, h, sh)
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (_entries(3), {}, None))
        monkeypatch.setattr(poller, "crawl_article", crawl)
        with caplog.at_level("DEBUG"):
            res = _poll(_feed())
        assert res["new"] == 2
        assert res["errors"] == 1
        assert res["exact_dups"] == 0                   # NOT counted as a duplicate
        warn = [r for r in caplog.records if r.levelname == "WARNING" and "kaboom" in r.getMessage()]
        assert warn

    def test_invalid_links_are_neither_dups_nor_errors(self, rec, monkeypatch):
        entries = _entries(1) + [{"title": "x", "link": "mailto:a@b"}, {"title": "y"}]
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (entries, {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        res = _poll(_feed())
        assert res["new"] == 1 and res["errors"] == 0 and res["exact_dups"] == 0

    def test_uppercase_scheme_accepted(self, rec, monkeypatch):
        entries = [{"title": "Up", "link": "HTTP://Example.COM/Story"}]
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (entries, {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        assert _poll(_feed())["new"] == 1

    def test_failed_db_rows_counted_as_errors(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (_entries(3), {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        monkeypatch.setattr(db, "upsert_articles", lambda rows: (1, 1))   # 1 row lost
        res = _poll(_feed())
        assert res["new"] == 1 and res["exact_dups"] == 1 and res["errors"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 8: EITHER HASH COUNTS AS SEEN
# ─────────────────────────────────────────────────────────────────────────────

class TestLegacyHashTransition:

    def test_one_batched_db_lookup_per_feed_with_both_hashes(self, rec, monkeypatch):
        """All entries' new + legacy hashes in ONE lookup (no per-entry queries)."""
        from pipeline.dedup import url_hash, legacy_url_hash
        calls = []
        monkeypatch.setattr(db, "existing_hashes", lambda hashes: calls.append(set(hashes)) or set())
        urls = ["http://s.example/a?sid=7", "https://s.example/b"]
        monkeypatch.setattr(poller, "_fetch_rss_blocking",
                            lambda url: ([{"title": f"T{i}", "link": u} for i, u in enumerate(urls)], {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        assert _poll(_feed())["new"] == 2
        assert len(calls) == 1
        assert calls[0] == {f(u) for u in urls for f in (url_hash, legacy_url_hash)}

    def test_article_stored_under_legacy_hash_is_not_reingested(self, rec, monkeypatch):
        from pipeline.dedup import legacy_url_hash
        legacy = legacy_url_hash("http://s.example/a")
        monkeypatch.setattr(db, "existing_hashes", lambda hashes: {legacy} & set(hashes))
        monkeypatch.setattr(poller, "_fetch_rss_blocking",
                            lambda url: ([{"title": "T", "link": "http://s.example/a"}], {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        res = _poll(_feed())
        assert res["new"] == 0 and res["exact_dups"] == 1

    def test_duplicate_entries_within_one_feed_insert_once(self, rec, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (
            [{"title": "T", "link": "https://s.example/a"}, {"title": "T again", "link": "https://s.example/a?utm_source=x"}],
            {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)
        res = _poll(_feed())
        assert res["new"] == 1 and res["exact_dups"] == 1

    def test_only_new_hash_is_stored_and_http_url_preserved_for_fetching(self, rec, monkeypatch):
        from pipeline.dedup import url_hash
        seen = {}

        def crawl(entry, feed, norm, h, sh):
            seen.update(norm=norm, h=h)
            return _fake_crawl(entry, feed, norm, h, sh)
        monkeypatch.setattr(poller, "_fetch_rss_blocking",
                            lambda url: ([{"title": "T", "link": "http://www.s.example/a/?utm_source=x"}], {}, None))
        monkeypatch.setattr(poller, "crawl_article", crawl)
        _poll(_feed())
        assert seen["norm"] == "http://s.example/a"          # still http: crawl stays fetchable
        assert seen["h"] == url_hash("https://s.example/a")  # hash is scheme-independent
        assert rec.upserts[0][0]["url_hash"] == seen["h"]


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 14: TITLE / SIMHASH CONSISTENCY, NEAR-DUP
# ─────────────────────────────────────────────────────────────────────────────

class TestNearDup:

    def test_near_dup_flagged_using_stored_title_hash(self, rec, monkeypatch):
        entries = [{"title": "Budget 2024 Finance Minister announces tax relief", "link": "https://a.example/1"},
                   {"title": "Budget 2024 Finance Minister announces tax relief", "link": "https://b.example/2"}]
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: (entries, {}, None))
        monkeypatch.setattr(poller, "crawl_article", _fake_crawl)

        # process sequentially so one is seen before the other
        async def go():
            loop = asyncio.get_running_loop()
            seen_sh: set = set()
            out = []
            for e in entries:
                out.append(await poller._process_one_entry(
                    e, _feed(), set(), seen_sh, asyncio.Semaphore(1), loop))
            return out
        a, b = asyncio.run(go())
        assert a.is_duplicate is False and b.is_duplicate is True


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 5: main.py EXITS 1 WHEN FEEDS CAN'T BE LOADED
# ─────────────────────────────────────────────────────────────────────────────

class TestDbOutage:

    def test_run_pipeline_raises_and_finalises_run_row(self, monkeypatch):
        finished = []
        monkeypatch.setattr(poller, "log_run_start", lambda *a: "run1")
        monkeypatch.setattr(poller, "log_run_finish", lambda rid, s: finished.append((rid, dict(s))))

        def boom(cadence=None):
            raise db.FeedLoadError("supabase down")
        monkeypatch.setattr(db, "get_due_feeds", boom)
        with pytest.raises(db.FeedLoadError):
            asyncio.run(poller.run_pipeline("daily"))
        assert finished and finished[0][0] == "run1" and finished[0][1]["errors"] == 1

    def test_main_exits_1(self, monkeypatch):
        import main

        async def crash(cadence=None, dry_run=False):
            raise db.FeedLoadError("supabase down")
        monkeypatch.setattr(main, "run_pipeline", crash)
        monkeypatch.setattr("sys.argv", ["main.py", "--cadence", "daily"])
        with pytest.raises(SystemExit) as ei:
            main.main()
        assert ei.value.code == 1
