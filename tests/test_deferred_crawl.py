"""
tests/test_deferred_crawl.py
════════════════════════════
Deferred crawling (2026-10-06): ingestion only DISCOVERS articles from feed
metadata; processing runners crawl them later (pipeline.crawler.crawl_record,
enrichment/handoff_runner.py) and bodies go to Storage (enrichment/body_store.py).
No network: fetches are monkeypatched.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from pipeline import crawler, db, poller
from pipeline.crawler import CrawledArticle, crawl_record, discover_article
from pipeline.handoff import Handoff, read

FEED = {"id": "f1", "feed_url": "http://f/x", "domain": "s.example", "language_code": "hi",
        "publisher_name": "S", "has_paywall": False}

PAGE = """<html><head>
<meta property="og:title" content="Page Title | Site">
<meta property="og:description" content="A page description that is long enough.">
<meta property="og:image" content="https://s.example/img.jpg">
<meta property="article:published_time" content="2026-10-05T08:00:00+05:30">
<script type="application/ld+json">{"author": {"name": "R. Reporter"}}</script>
</head><body><article><p>%s</p></article></body></html>""" % (" ".join(["Body sentence number %d." % i for i in range(80)]))


@pytest.fixture(autouse=True)
def _fresh_domain_tracker():
    crawler.reset_domain_failures()
    yield
    crawler.reset_domain_failures()


class TestDiscover:

    def test_no_http_and_feed_metadata(self, monkeypatch):
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks",
                            lambda *a, **k: pytest.fail("discovery must not fetch"))
        entry = {"title": "Headline &amp; more", "summary": "A description longer than twenty chars",
                 "published": "Mon, 05 Oct 2026 06:00:00 GMT"}
        a = discover_article(entry, FEED, "https://s.example/a", 123, 99)
        assert a.title == "Headline & more" and a.description == "A description longer than twenty chars"
        assert a.published_at == datetime(2026, 10, 5, 6, tzinfo=timezone.utc)
        assert not a.is_crawled and a.full_text == "" and a.language_code == "hi"

    def test_feed_full_text_means_already_crawled(self):
        body = "<p>" + " ".join(["word"] * 300) + "</p>"
        entry = {"title": "T", "content": [{"value": body}]}
        a = discover_article(entry, FEED, "https://s.example/a", 1, 0)
        assert a.is_crawled and a.crawl_strategy == "rss_content" and len(a.full_text) > 500

    def test_future_date_capped_and_paywall_flag(self):
        future = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        a = discover_article({"title": "T", "published": future}, dict(FEED, has_paywall=True),
                             "https://s.example/a", 1, 0)
        assert a.published_at <= a.crawled_at and a.paywalled
        assert "paywalled" not in a.to_db_row()                 # not a DB column


class TestCrawlRecord:

    def test_success_fills_crawl_columns_keeps_feed_title(self, monkeypatch):
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", lambda url, paywalled=False: (PAGE, "default"))
        out = crawl_record({"url": "https://s.example/a", "title": "Feed title", "published_at": None})
        assert out["is_crawled"] and out["crawl_strategy"] == "default"
        assert "Body sentence number 5." in out["full_text"]
        assert "title" not in out                                # the feed title stays canonical
        assert out["description"].startswith("A page description")
        assert out["top_image_url"] == "https://s.example/img.jpg"
        assert out["published_at"] == "2026-10-05T02:30:00+00:00"   # filled because missing, in UTC
        assert out["og_tags"]["og:title"] == "Page Title | Site"

    def test_existing_publish_date_not_overwritten_and_title_filled_when_missing(self, monkeypatch):
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", lambda url, paywalled=False: (PAGE, "googlebot"))
        out = crawl_record({"url": "https://s.example/a", "title": "", "published_at": "2026-10-01T00:00:00+00:00"})
        assert "published_at" not in out
        assert out["title"] == "Page Title | Site" and isinstance(out["title_simhash"], int)

    def test_failure_and_tripped_domain(self, monkeypatch):
        calls = []

        def fail(url, paywalled=False):
            calls.append(url)
            return None, "failed"
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", fail)
        for i in range(20):
            assert crawl_record({"url": f"https://dead.example/{i}"}) == {"is_crawled": False,
                                                                        "crawl_strategy": "failed"}
        assert len(calls) < 20                                   # domain fail-fast kicked in

    def test_never_raises(self, monkeypatch):
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", lambda *a, **k: 1 / 0)
        assert crawl_record({"url": "https://s.example/x"})["crawl_strategy"] == "failed"


class TestIncrementalHandoff:

    def test_each_add_is_on_disk_immediately(self, tmp_path):
        p = tmp_path / "h.jsonl.gz"
        h = Handoff(path=p)
        arts = [CrawledArticle(feed_id="f", url=f"https://x/{i}", url_hash=i, paywalled=(i == 1)) for i in range(3)]
        h.add_inserted([{"id": 10, "url_hash": "0"}], arts)
        assert [r["id"] for r in read([p])] == [10]              # survives a kill right here
        h.add_inserted([{"id": 11, "url_hash": "1"}, {"id": 12, "url_hash": "2"}], arts)
        recs = read([p])
        assert [r["id"] for r in recs] == [10, 11, 12] and recs[1]["paywalled"] is True
        assert h.write() == 3                                    # finish = log only

    def test_new_run_truncates_old_file(self, tmp_path):
        p = tmp_path / "h.jsonl.gz"
        with gzip.open(p, "wt") as f:
            f.write(json.dumps({"id": 1}) + "\n")
        Handoff(path=p)
        assert read([p]) == []


def _poll(feed, **kw):
    async def go():
        loop = asyncio.get_running_loop()
        return await poller.poll_one_feed(feed, asyncio.Semaphore(2), asyncio.Semaphore(2), set(), loop, **kw)
    return asyncio.run(go())


def _feed(**kw):
    base = {"id": "f1", "feed_url": "http://f/x", "update_cadence": "daily", "is_active": True,
            "fail_count": 0, "last_new_article_at": None, "created_at": "2000-01-01T00:00:00+00:00"}
    base.update(kw)
    return base


class TestPollerDeferred:

    @pytest.fixture(autouse=True)
    def _db(self, monkeypatch):
        monkeypatch.setattr(db, "existing_hashes", lambda hashes: set())
        monkeypatch.setattr(db, "upsert_articles_returning",
                            lambda rows: ([dict(r, id=i + 1) for i, r in enumerate(rows)], 0))

    def test_ingest_does_not_crawl_when_deferred(self, monkeypatch, tmp_path):
        monkeypatch.setattr(poller, "CRAWL_AT_INGEST", False)
        monkeypatch.setattr(poller, "crawl_article", lambda *a: pytest.fail("ingest crawled"))
        monkeypatch.setattr(poller, "_fetch_rss_blocking",
                            lambda url: ([{"title": f"T{i}", "link": f"https://s.example/a{i}"} for i in range(3)],
                                         {}, None))
        h = Handoff(path=tmp_path / "h.jsonl.gz")
        res = _poll(_feed(), polls=db.FeedPollBatch(), handoff=h)
        assert res["new"] == 3 and len(read([tmp_path / "h.jsonl.gz"])) == 3

    def test_deadline_leaves_feed_due(self, monkeypatch):
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: pytest.fail("polled after deadline"))
        polls = db.FeedPollBatch()
        res = _poll(_feed(), polls=polls, deadline=time.perf_counter() - 1)
        assert res.get("deferred") and polls.items == []         # nothing recorded → still due

    def test_poll_state_flushed_as_it_goes(self, monkeypatch):
        flushed = []
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: ([], {}, "HTTP 500"))
        polls = db.FeedPollBatch(batch_size=2)
        monkeypatch.setattr(polls, "flush", lambda: flushed.append(len(polls.items)) or polls.items.clear() or 0)
        for i in range(5):
            _poll(_feed(id=f"f{i}"), polls=polls)
        assert flushed == [2, 2] and len(polls.items) == 1


class TestBodyStore:

    def _client(self, uploads, buckets=("article-bodies",), fail=False):
        class Bucket:
            def upload(self, path, data, opts):
                if fail:
                    raise RuntimeError("storage down")
                uploads.append((path, data, opts))

        class Storage:
            def list_buckets(self):
                return [{"name": b} for b in buckets]

            def create_bucket(self, name, options=None):
                uploads.append(("CREATE", name, options))

            def from_(self, name):
                return Bucket()

        class C:
            storage = Storage()
        return lambda: C()

    def test_uploads_gzip_jsonl_and_skips_empty_bodies(self):
        from enrichment.body_store import BodyStore
        uploads = []
        bs = BodyStore("handoff-0-123", client_factory=self._client(uploads))
        bs.add({"id": 1, "url": "u", "full_text": "Body one"})
        bs.add({"id": 2, "url": "v", "full_text": "  "})
        assert bs.flush() == 1 and bs.uploaded == 1
        path, data, opts = uploads[0]
        assert path.endswith(".jsonl.gz") and "handoff-0-123" in path and opts["content-type"] == "application/gzip"
        rows = [json.loads(line) for line in gzip.decompress(data).decode().splitlines()]
        assert rows == [{"id": 1, "url": "u", "domain": None, "language_code": None, "title": None,
                         "published_at": None, "crawled_at": None, "crawl_strategy": None,
                         "full_text": "Body one"}]

    def test_creates_missing_bucket_and_survives_upload_failure(self):
        from enrichment.body_store import BodyStore
        uploads = []
        bs = BodyStore("t", client_factory=self._client(uploads, buckets=(), fail=True))
        bs.add({"id": 1, "full_text": "x"})
        assert bs.flush() == 0 and bs.failed == 1
        assert uploads[0][0] == "CREATE" and uploads[0][2] == {"public": False}


class TestSweeper:

    def test_claims_saves_releases_and_dead_letters(self, monkeypatch):
        from enrichment import handoff_runner
        calls = {}
        claimed = [{"id": i, "url": f"https://s.example/{i}", "title": "t", "enrich_attempts": 3 if i == 2 else 1}
                   for i in range(4)]
        monkeypatch.setattr(handoff_runner.db, "claim_batch",
                            lambda n, min_age_hours=0: calls.setdefault("claim", (n, min_age_hours)) and claimed)
        monkeypatch.setattr(handoff_runner.db, "save_enrichment_batch", lambda items: len(items))
        monkeypatch.setattr(handoff_runner.db, "log_run_start", lambda row: None)
        monkeypatch.setattr(handoff_runner.db, "log_run_finish", lambda rid, row: None)
        monkeypatch.setattr(handoff_runner.db, "release_claims",
                            lambda ids: calls.setdefault("released", sorted(ids)) and len(ids))
        monkeypatch.setattr(handoff_runner.db, "mark_enrichment_failed",
                            lambda aid, err: calls.setdefault("dead", []).append(aid))
        monkeypatch.setattr(handoff_runner, "crawl_record",
                            lambda rec: {"is_crawled": True, "crawl_strategy": "default", "full_text": "body"})
        monkeypatch.setattr(handoff_runner, "_hash_image", lambda url: None)
        monkeypatch.setattr(handoff_runner, "BodyStore",
                            lambda tag, enabled=True: type("B", (), {"add": lambda s, r: None, "flush": lambda s: 0,
                                                                     "uploaded": 0})())

        def enrich(article):
            if article["id"] in (2, 3):
                raise RuntimeError("poison")
            assert article["full_text"] == "body"                # crawled before enrichment
            return {"category": "x"}, []
        monkeypatch.setattr(handoff_runner, "enrich_one", enrich)
        s = handoff_runner.run_sweeper(500, 6)
        assert calls["claim"] == (500, 6)
        assert sorted(s["saved_ids"]) == [0, 1]
        assert calls["dead"] == [2]                              # final attempt → dead letter
        assert "released" not in calls                           # 3 keeps its lease (retried later)
