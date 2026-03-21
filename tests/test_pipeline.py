"""
tests/test_pipeline.py
══════════════════════
Unit tests — fully offline, no Supabase connection needed.

HOW TO RUN:
  pip install pytest
  pytest tests/ -v

WHAT ARE UNIT TESTS?
  Small, isolated tests that verify each function works correctly.
  They run in milliseconds and catch bugs before they reach production.
  These tests don't connect to Supabase or make any network requests.
"""

import pytest
from datetime import datetime, timezone, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.dedup import normalise_url, url_hash, simhash, hamming_distance, is_near_duplicate


class TestNormaliseUrl:

    def test_strips_utm_source(self):
        url = "https://ndtv.com/story?utm_source=twitter"
        assert normalise_url(url) == "https://ndtv.com/story"

    def test_strips_multiple_tracking_params(self):
        url = "https://ndtv.com/story?utm_source=tw&utm_campaign=hp&fbclid=abc"
        assert normalise_url(url) == "https://ndtv.com/story"

    def test_keeps_content_params(self):
        url = "https://example.com/article?id=123&section=sports"
        result = normalise_url(url)
        assert "id=123" in result
        assert "section=sports" in result

    def test_strips_www_prefix(self):
        assert normalise_url("https://www.thehindu.com/news/") == \
               "https://thehindu.com/news"

    def test_strips_trailing_slash(self):
        assert normalise_url("https://example.com/article/") == \
               "https://example.com/article"

    def test_preserves_root_slash(self):
        # Root URL should keep its slash
        result = normalise_url("https://example.com/")
        assert result == "https://example.com"

    def test_lowercases_domain(self):
        assert normalise_url("https://NDTV.COM/story") == \
               "https://ndtv.com/story"

    def test_strips_fragment(self):
        assert normalise_url("https://example.com/art#comments") == \
               "https://example.com/art"

    def test_adds_https_scheme(self):
        assert normalise_url("example.com/story").startswith("https://")

    def test_empty_string(self):
        assert normalise_url("") == ""

    def test_same_article_different_tracking(self):
        """The key test: same article, different tracking params → same normalised URL"""
        urls = [
            "https://www.thehindu.com/news/art.cms?utm_source=twitter&utm_medium=social",
            "https://thehindu.com/news/art.cms?fbclid=IwAR0abc",
            "https://thehindu.com/news/art.cms/",
            "https://www.thehindu.com/news/art.cms",
        ]
        normalised = {normalise_url(u) for u in urls}
        assert len(normalised) == 1, f"Expected 1 unique URL, got: {normalised}"


class TestUrlHash:

    def test_returns_integer(self):
        assert isinstance(url_hash("https://example.com"), int)

    def test_deterministic(self):
        h1 = url_hash("https://ndtv.com/story")
        h2 = url_hash("https://ndtv.com/story")
        assert h1 == h2

    def test_different_urls_different_hashes(self):
        assert url_hash("https://ndtv.com/story-a") != \
               url_hash("https://ndtv.com/story-b")

    def test_normalises_before_hashing(self):
        """Same article with tracking params should produce same hash"""
        h1 = url_hash("https://www.thehindu.com/news/?utm_source=fb")
        h2 = url_hash("https://thehindu.com/news")
        assert h1 == h2

    def test_fits_postgresql_bigint(self):
        """Must fit in PostgreSQL bigint range: -2^63 to 2^63-1"""
        h = url_hash("https://example.com/some/long/path/article-123456789")
        assert -(2**63) <= h < 2**63


class TestSimHash:

    def test_returns_integer(self):
        assert isinstance(simhash("Budget 2024 analysis"), int)

    def test_deterministic(self):
        s1 = simhash("RBI raises interest rates by 25 basis points")
        s2 = simhash("RBI raises interest rates by 25 basis points")
        assert s1 == s2

    def test_empty_string(self):
        assert simhash("") == 0

    def test_different_topics_far_apart(self):
        h1 = simhash("IPL 2024 cricket match Chennai Mumbai")
        h2 = simhash("Stock market Sensex Nifty falls 500 points budget")
        dist = hamming_distance(h1, h2)
        assert dist > 3, f"Unrelated topics should have distance > 3, got {dist}"


class TestHammingDistance:

    def test_identical_hashes(self):
        assert hamming_distance(12345, 12345) == 0

    def test_completely_different(self):
        # All 64 bits different
        assert hamming_distance(0, (1 << 64) - 1) == 64

    def test_one_bit_different(self):
        assert hamming_distance(0b1000, 0b1001) == 1


class TestIsNearDuplicate:

    def test_identical_titles(self):
        t = "Budget 2024: Finance Minister announces tax relief"
        assert is_near_duplicate(simhash(t), simhash(t))

    def test_clearly_different_topics(self):
        t1 = simhash("IPL cricket Mumbai Indians beat Chennai Super Kings")
        t2 = simhash("Sensex falls stock market global selloff recession")
        assert not is_near_duplicate(t1, t2)


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER TESTS
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.circuit_breaker import check_dormancy, should_skip_feed


class TestShouldSkipFeed:

    def test_active_feed_not_skipped(self):
        feed = {"is_active": True}
        skip, _ = should_skip_feed(feed)
        assert not skip

    def test_inactive_feed_skipped(self):
        feed = {"is_active": False}
        skip, reason = should_skip_feed(feed)
        assert skip
        assert "inactive" in reason.lower()


class TestCheckDormancy:

    def test_feed_with_recent_articles_not_dormant(self):
        feed = {}
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        dormant, _ = check_dormancy(feed, recent)
        assert not dormant

    def test_feed_stale_30_days_is_dormant(self):
        feed = {}
        stale = datetime.now(timezone.utc) - timedelta(days=31)
        dormant, reason = check_dormancy(feed, stale)
        assert dormant
        assert "31" in reason or "30" in reason

    def test_feed_at_29_days_not_dormant(self):
        feed = {}
        recent_enough = datetime.now(timezone.utc) - timedelta(days=29)
        dormant, _ = check_dormancy(feed, recent_enough)
        assert not dormant

    def test_feed_never_produced_articles_new_feed_not_dormant(self):
        """A brand-new feed has never produced articles — don't mark dormant yet"""
        feed = {"created_at": datetime.now(timezone.utc).isoformat()}
        dormant, _ = check_dormancy(feed, None)
        assert not dormant

    def test_feed_never_produced_articles_old_feed_is_dormant(self):
        """If a feed was added 31 days ago and never had articles → dormant"""
        old_date = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        feed = {"created_at": old_date}
        dormant, reason = check_dormancy(feed, None)
        assert dormant


# ─────────────────────────────────────────────────────────────────────────────
# CRAWLER TESTS (no network)
# ─────────────────────────────────────────────────────────────────────────────

from pipeline.crawler import (
    is_paywalled, _extract_og_tags, _extract_rss_description,
    _extract_rss_image, _extract_domain,
)


class TestIsPaywalled:

    def test_known_paywalled_domain(self):
        assert is_paywalled("https://thehindu.com/news/article.html")

    def test_www_paywalled_domain(self):
        assert is_paywalled("https://www.ft.com/content/abc-123")

    def test_subdomain_of_paywalled(self):
        assert is_paywalled("https://epaper.thehindu.com/article")

    def test_free_domain(self):
        assert not is_paywalled("https://ndtv.com/india/story")

    def test_db_paywall_flag_overrides(self):
        """has_paywall=True from feeds table should override domain check"""
        assert is_paywalled("https://ndtv.com/story", db_has_paywall=True)


class TestExtractOgTags:

    SAMPLE_HTML = """
    <html><head>
    <meta property="og:title" content="Budget 2024: FM announces tax relief"/>
    <meta property="og:description" content="Finance Minister announced income tax relief"/>
    <meta property="og:image" content="https://cdn.example.com/img.jpg"/>
    <meta property="og:url" content="https://example.com/budget-2024"/>
    <meta property="article:published_time" content="2024-02-01T10:00:00+05:30"/>
    <meta property="article:author" content="John Doe"/>
    <link rel="canonical" href="https://example.com/budget-2024"/>
    </head></html>
    """

    def test_extracts_og_title(self):
        tags = _extract_og_tags(self.SAMPLE_HTML)
        assert tags["og:title"] == "Budget 2024: FM announces tax relief"

    def test_extracts_og_image(self):
        tags = _extract_og_tags(self.SAMPLE_HTML)
        assert tags["og:image"] == "https://cdn.example.com/img.jpg"

    def test_extracts_canonical(self):
        tags = _extract_og_tags(self.SAMPLE_HTML)
        assert tags["canonical"] == "https://example.com/budget-2024"

    def test_extracts_published_time(self):
        tags = _extract_og_tags(self.SAMPLE_HTML)
        assert "article:published_time" in tags

    def test_empty_html(self):
        tags = _extract_og_tags("")
        assert tags == {}

    def test_twitter_image_fallback(self):
        html = '<meta name="twitter:image" content="https://cdn.example.com/tw.jpg"/>'
        tags = _extract_og_tags(html)
        assert tags.get("og:image") == "https://cdn.example.com/tw.jpg"


class TestRssHelpers:

    def test_description_strips_html_tags(self):
        entry = {"summary": "<p>Hello <b>world</b></p>"}
        assert _extract_rss_description(entry) == "Hello world"

    def test_description_from_description_key(self):
        entry = {"description": "This is a news article"}
        assert _extract_rss_description(entry) == "This is a news article"

    def test_image_from_media_content(self):
        entry = {"media_content": [{"url": "https://example.com/img.jpg"}]}
        assert _extract_rss_image(entry) == "https://example.com/img.jpg"

    def test_image_from_enclosure(self):
        entry = {"enclosures": [{"href": "https://img.com/a.jpg", "type": "image/jpeg"}]}
        assert _extract_rss_image(entry) == "https://img.com/a.jpg"

    def test_image_empty_when_none(self):
        assert _extract_rss_image({}) == ""


class TestExtractDomain:

    def test_strips_www(self):
        assert _extract_domain("https://www.ndtv.com/story") == "ndtv.com"

    def test_no_www(self):
        assert _extract_domain("https://thehindu.com/news") == "thehindu.com"
