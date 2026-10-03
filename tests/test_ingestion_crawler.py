"""
tests/test_ingestion_crawler.py
═══════════════════════════════
Regression tests for pipeline/crawler.py (items 6, 7, 11, 14, S2).
No network: urllib is monkeypatched.
"""

import json
import threading
import urllib.error
from datetime import datetime, timezone

import pytest

from pipeline import crawler
from pipeline.crawler import (
    CrawledArticle, _extract_jsonld, _extract_publish_date, _pick_image,
    _jsonld_image, _fetch_with_fallbacks, DomainFailureTracker, crawl_article,
    _is_permanent_status, reset_domain_failures,
)


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 6: IMAGE PRIORITY
# ─────────────────────────────────────────────────────────────────────────────

class TestPickImage:

    def test_og_image_wins_when_jsonld_has_no_image(self):
        """The old precedence bug discarded og:image in exactly this case."""
        assert _pick_image({"og:image": "https://c/og.jpg"}, {}, "https://c/rss.jpg", "", "https://s/a") \
            == "https://c/og.jpg"

    def test_og_image_wins_over_jsonld_dict(self):
        assert _pick_image({"og:image": "og"}, {"image": {"url": "jl"}}, "rss", "", "") == "og"

    def test_jsonld_dict(self):
        assert _pick_image({}, {"image": {"url": "https://c/jl.jpg"}}, "rss", "", "") == "https://c/jl.jpg"

    def test_jsonld_string(self):
        assert _pick_image({}, {"image": "https://c/jl.jpg"}, "rss", "", "") == "https://c/jl.jpg"

    def test_jsonld_list_of_strings_and_dicts(self):
        assert _jsonld_image({"image": ["https://c/1.jpg", "https://c/2.jpg"]}) == "https://c/1.jpg"
        assert _jsonld_image({"image": [{"url": "https://c/1.jpg"}]}) == "https://c/1.jpg"
        assert _jsonld_image({"image": [{"width": 5}, "https://c/2.jpg"]}) == "https://c/2.jpg"

    def test_jsonld_weird_shapes_do_not_raise(self):
        assert _jsonld_image({"image": 123}) == ""
        assert _jsonld_image({"image": [None, 4]}) == ""
        assert _jsonld_image({"image": {"url": None}}) == ""

    def test_rss_image_next(self):
        assert _pick_image({}, {}, "https://c/rss.jpg", "", "") == "https://c/rss.jpg"

    def test_first_img_last_resort_resolves_relative(self):
        html = '<img src="/img/logo.png"><img src="/img/photo.jpg">'
        assert _pick_image({}, {}, "", html, "https://site.com/news/a") == "https://site.com/img/photo.jpg"

    def test_nothing_found(self):
        assert _pick_image({}, {}, "", "<p>x</p>", "https://s/a") == ""


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 7: DATE / JSON-LD HARDENING
# ─────────────────────────────────────────────────────────────────────────────

class TestPublishDate:

    def test_naive_jsonld_date_becomes_utc(self):
        d = _extract_publish_date({}, {}, {"datePublished": "2024-02-01T10:00:00"})
        assert d.tzinfo is not None and d.utcoffset().total_seconds() == 0
        # the crash this prevented:
        assert d < datetime.now(timezone.utc)

    def test_date_only_is_utc(self):
        d = _extract_publish_date({}, {"article:published_time": "2024-02-01"}, {})
        assert d == datetime(2024, 2, 1, tzinfo=timezone.utc)

    def test_offset_preserved(self):
        d = _extract_publish_date({}, {}, {"datePublished": "2024-02-01T10:00:00+05:30"})
        assert d.utcoffset().total_seconds() == 5.5 * 3600

    @pytest.mark.parametrize("bad", [{"@value": "2024-01-01"}, 20240101, ["2024-01-01"], None, "", "not a date"])
    def test_non_string_or_garbage_date_does_not_raise(self, bad):
        assert _extract_publish_date({}, {}, {"datePublished": bad}) is None

    def test_garbage_jsonld_falls_through_to_og(self):
        d = _extract_publish_date({}, {"article:published_time": "2024-03-04T00:00:00Z"},
                                  {"datePublished": {"x": 1}})
        assert d == datetime(2024, 3, 4, tzinfo=timezone.utc)

    def test_rfc822_naive_minus_0000_is_utc(self):
        d = _extract_publish_date({"published": "Mon, 01 Jan 2024 10:00:00 -0000"}, {}, {})
        assert d == datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

    def test_rss_parsed_tuple_still_wins(self):
        import time
        t = time.strptime("2024-05-05 05:05:05", "%Y-%m-%d %H:%M:%S")
        d = _extract_publish_date({"published_parsed": t}, {}, {"datePublished": "2020-01-01"})
        assert d.year == 2024


class TestExtractJsonLd:

    @staticmethod
    def _page(payload) -> str:
        return f'<script type="application/ld+json">{json.dumps(payload)}</script>'

    def test_list_with_non_dict_first_element(self):
        page = self._page(["junk", {"@type": "NewsArticle", "headline": "H"}])
        assert _extract_jsonld(page)["headline"] == "H"

    def test_list_of_only_strings(self):
        assert _extract_jsonld(self._page(["a", "b"])) == {}

    def test_scalar_payload(self):
        assert _extract_jsonld(self._page("hello")) == {}
        assert _extract_jsonld(self._page(42)) == {}

    def test_graph_wrapper(self):
        page = self._page({"@graph": [{"@type": "WebSite"}, {"@type": "Article", "headline": "G"}]})
        assert _extract_jsonld(page)["headline"] == "G"

    def test_second_block_used_if_first_invalid(self):
        page = '<script type="application/ld+json">{bad</script>' + \
               self._page({"@type": "BlogPosting", "headline": "B"})
        assert _extract_jsonld(page)["headline"] == "B"


# ─────────────────────────────────────────────────────────────────────────────
# S2: NUL BYTES
# ─────────────────────────────────────────────────────────────────────────────

class TestNulStripping:

    def test_all_text_fields_and_nested_og_tags_cleaned(self):
        a = CrawledArticle(
            feed_id="f\x00", url="https://x/a\x00", url_hash=1,
            title="ti\x00tle", description="de\x00sc", full_text="bo\x00dy",
            author="au\x00thor", top_image_url="https://x/i\x00.jpg",
            og_tags={"og:title": "t\x00", "_videos": [{"url": "v\x00"}], "n": 3},
            publisher_name="pu\x00b", domain="d\x00", feed_url="fu\x00",
            iab_tier1="i\x001", iab_tier2="i\x002", language="Eng\x00lish",
        )
        row = a.to_db_row()

        def walk(v):
            if isinstance(v, str):
                assert "\x00" not in v
            elif isinstance(v, dict):
                for k, x in v.items():
                    walk(k); walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
        walk(row)
        assert row["title"] == "title"
        assert row["og_tags"]["_videos"][0]["url"] == "v"
        assert row["og_tags"]["n"] == 3          # non-strings untouched
        assert row["title_simhash"] is None


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 11: RETRY POLICY, DEADLINE, DOMAIN FAIL-FAST
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    reset_domain_failures()
    monkeypatch.setattr(crawler, "RETRY_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(crawler, "MAX_FETCH_RETRIES", 3)
    yield
    reset_domain_failures()


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "x", {}, None)


class _Resp:
    def __init__(self, body):
        self._b = body.encode()
        self.headers = {"Content-Type": "text/html"}

    def read(self, n=-1):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestRetryPolicy:

    def test_status_classification(self):
        for code in (400, 401, 403, 404, 410, 451):
            assert _is_permanent_status(code)
        for code in (408, 429, 500, 502, 503, None):
            assert not _is_permanent_status(code)

    @pytest.mark.parametrize("code", [403, 404, 410, 451])
    def test_permanent_4xx_not_retried(self, monkeypatch, code):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.full_url)
            raise _http_error(req.full_url, code)
        monkeypatch.setattr(crawler.urllib.request, "urlopen", fake_urlopen)
        html, strategy = _fetch_with_fallbacks("https://example.com/a", paywalled=True)
        assert html is None and strategy == "failed"
        # googlebot strategy: exactly ONE attempt instead of MAX_FETCH_RETRIES
        assert calls.count("https://example.com/a") == 1

    def test_404_skips_second_ua_strategy(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(req.headers.get("User-agent"))
            raise _http_error(req.full_url, 404)
        monkeypatch.setattr(crawler.urllib.request, "urlopen", fake_urlopen)
        _fetch_with_fallbacks("https://example.com/a", paywalled=False)
        direct = [c for c in calls if c in (crawler.USER_AGENT, crawler.GOOGLEBOT_UA)]
        # default UA once (404 -> gone); googlebot never tried on the same URL
        assert crawler.GOOGLEBOT_UA not in calls[:1]
        assert calls[0] == crawler.USER_AGENT and calls.count(crawler.GOOGLEBOT_UA) == 0

    def test_403_still_falls_through_to_googlebot(self, monkeypatch):
        seen = []

        def fake_urlopen(req, timeout=None):
            ua = req.headers.get("User-agent")
            seen.append(ua)
            if ua == crawler.GOOGLEBOT_UA:
                return _Resp("<html>" + "x" * 600 + "</html>")
            raise _http_error(req.full_url, 403)
        monkeypatch.setattr(crawler.urllib.request, "urlopen", fake_urlopen)
        html, strategy = _fetch_with_fallbacks("https://example.com/a")
        assert strategy == "googlebot" and html
        assert seen.count(crawler.USER_AGENT) == 1       # not retried 3x

    @pytest.mark.parametrize("code", [408, 429, 503])
    def test_transient_errors_are_retried(self, monkeypatch, code):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                raise _http_error(req.full_url, code)
            return _Resp("<html>" + "y" * 600 + "</html>")
        monkeypatch.setattr(crawler.urllib.request, "urlopen", fake_urlopen)
        html, strategy = _fetch_with_fallbacks("https://example.com/a")
        assert strategy == "default" and len(calls) == 3


class TestArticleDeadline:

    def test_deadline_stops_further_attempts(self, monkeypatch):
        calls = []
        clock = {"t": 1000.0}

        def fake_urlopen(req, timeout=None):
            calls.append(timeout)
            clock["t"] += 10                      # each attempt "takes" 10 s
            raise urllib.error.URLError("slow")
        monkeypatch.setattr(crawler.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(crawler.time, "monotonic", lambda: clock["t"])
        monkeypatch.setattr(crawler.time, "sleep", lambda s: None)
        html, strategy = _fetch_with_fallbacks("https://example.com/a", deadline_seconds=25)
        assert (html, strategy) == (None, "failed")
        # 25 s budget / 10 s per attempt -> far fewer than 2 UAs x 3 retries + amp + wayback
        assert len(calls) <= 4

    def test_request_timeout_is_clamped_to_remaining_budget(self, monkeypatch):
        seen = []

        def fake_urlopen(req, timeout=None):
            seen.append(timeout)
            raise urllib.error.URLError("x")
        monkeypatch.setattr(crawler.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(crawler.time, "sleep", lambda s: None)
        _fetch_with_fallbacks("https://example.com/a", deadline_seconds=3)
        assert seen and max(seen) <= 3.0


class TestDomainFailFast:

    def test_trips_after_threshold_and_success_resets(self):
        t = DomainFailureTracker(3)
        t.record("a.com", False); t.record("a.com", False)
        assert not t.is_tripped("a.com")
        t.record("a.com", True)                  # streak reset
        t.record("a.com", False); t.record("a.com", False)
        assert not t.is_tripped("a.com")
        t.record("a.com", False)
        assert t.is_tripped("a.com")
        assert not t.is_tripped("b.com")         # per domain

    def test_thread_safe_counting(self):
        t = DomainFailureTracker(1000)
        def worker():
            for _ in range(250):
                t.record("x.com", False)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        [th.start() for th in threads]; [th.join() for th in threads]
        assert t.is_tripped("x.com")             # exactly 1000 failures recorded, none lost

    def test_reset(self):
        t = DomainFailureTracker(1)
        t.record("a.com", False)
        t.reset()
        assert not t.is_tripped("a.com")

    def test_crawl_article_skips_http_for_tripped_domain_and_keeps_rss(self, monkeypatch):
        attempts = []
        monkeypatch.setattr(
            crawler, "_fetch_with_fallbacks",
            lambda url, paywalled=False, deadline_seconds=None: attempts.append(url) or (None, "failed"),
        )
        monkeypatch.setattr(crawler, "_domain_failures", DomainFailureTracker(2))
        entry = {"title": "Hello world", "summary": "A reasonably long RSS description text."}
        feed = {"id": "f", "language_code": "en", "language_name": "English"}
        for i in range(4):
            art = crawl_article(entry, feed, f"https://dead.example/a{i}", i, 7)
            assert art.is_crawled is False and art.crawl_strategy == "failed"
            assert art.title == "Hello world" and "RSS description" in art.description
        # first two attempted, the remaining two short-circuited
        assert len(attempts) == 2

    def test_success_does_not_trip(self, monkeypatch):
        html = "<html><head><meta property='og:title' content='OG'/></head><body>" + "p" * 600 + "</body></html>"
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", lambda *a, **k: (html, "default"))
        monkeypatch.setattr(crawler, "_best_fulltext", lambda h, u: "full text")
        tracker = DomainFailureTracker(2)
        monkeypatch.setattr(crawler, "_domain_failures", tracker)
        for i in range(5):
            crawl_article({"title": "T"}, {"id": "f"}, f"https://ok.example/a{i}", i, 1)
        assert not tracker.is_tripped("ok.example")


# ─────────────────────────────────────────────────────────────────────────────
# ITEMS 13 + 14: language and title/simhash consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestCrawlArticleMetadata:

    HTML = ("<html><head><meta property='og:title' content='OG Title | Site'/>"
            "<meta property='og:image' content='https://c/og.jpg'/></head><body>"
            + "p" * 600 + "</body></html>")

    def _crawl(self, monkeypatch, entry, feed=None):
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", lambda *a, **k: (self.HTML, "default"))
        monkeypatch.setattr(crawler, "_best_fulltext", lambda h, u: "body text")
        return crawl_article(entry, feed or {"id": "f", "language_name": "Hindi"},
                             "https://s.example/a", 1, 4242)

    def test_title_stays_rss_title_and_simhash_matches(self, monkeypatch):
        art = self._crawl(monkeypatch, {"title": "Budget &amp; tax relief"})
        assert art.title == "Budget & tax relief"          # not og:title
        assert art.title_simhash == 4242                    # unchanged, same source
        assert art.og_tags["og:title"] == "OG Title | Site"  # still preserved

    def test_empty_rss_title_falls_back_and_recomputes_simhash(self, monkeypatch):
        from pipeline.dedup import simhash
        art = self._crawl(monkeypatch, {"title": ""})
        assert art.title == "OG Title | Site"
        assert art.title_simhash == simhash("OG Title | Site")

    def test_og_image_survives_without_jsonld(self, monkeypatch):
        art = self._crawl(monkeypatch, {"title": "T"})
        assert art.top_image_url == "https://c/og.jpg"

    def test_language_comes_from_feed(self, monkeypatch):
        art = self._crawl(monkeypatch, {"title": "T"})
        assert art.language == "Hindi"
