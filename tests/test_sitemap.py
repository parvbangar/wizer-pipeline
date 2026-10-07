"""
tests/test_sitemap.py — pipeline/sitemap.py (news sitemaps as a source).

Fixtures mirror formats seen on Indian publishers on 2026-10-06: CDATA titles
(Jagran), naive local timestamps, sitemap indexes (Dainik Bhaskar: 28
editions), misspelt language codes (Tamil as "tn"), lastmod-only sitemaps.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from pipeline import db, poller
from pipeline.crawler import discover_article
from pipeline.sitemap import fetch_news_sitemap, normalize_language, parse_date, parse_sitemap, since_for_feed

NOW = datetime(2026, 10, 6, 6, 0, tzinfo=timezone.utc)


def urlset(*items):
    body = "".join(items)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
            'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9" '
            'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">' + body + "</urlset>")


def news_url(loc, date, title="T", lang="hi", image=None, keywords=None):
    img = f"<image:image><image:loc>{image}</image:loc></image:image>" if image else ""
    kw = f"<news:keywords>{keywords}</news:keywords>" if keywords else ""
    return (f"<url><loc>{loc}</loc><news:news><news:publication><news:name>P</news:name>"
            f"<news:language>{lang}</news:language></news:publication>"
            f"<news:publication_date>{date}</news:publication_date>"
            f"<news:title><![CDATA[{title}]]></news:title>{kw}</news:news>{img}</url>")


def index(*children):
    body = "".join(f"<sitemap><loc>{loc}</loc><lastmod>{lm}</lastmod></sitemap>" for loc, lm in children)
    return f'<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}</sitemapindex>'


def downloader(pages: dict):
    calls = []

    def dl(url):
        calls.append(url)
        if url not in pages or pages[url] is None:
            raise RuntimeError(f"HTTP 404 {url}")
        return pages[url].encode("utf-8"), {}, url
    dl.calls = calls
    return dl


class TestParsing:

    def test_dates(self):
        assert parse_date("2026-10-06T10:30:00+05:30") == datetime(2026, 10, 6, 5, 0, tzinfo=timezone.utc)
        assert parse_date("2026-10-06T10:30:00Z") == datetime(2026, 10, 6, 10, 30, tzinfo=timezone.utc)
        assert parse_date("2026-10-06T10:30:00") == datetime(2026, 10, 6, 5, 0, tzinfo=timezone.utc)  # naive = IST
        assert parse_date("Tue, 06 Oct 2026 10:30:00 +0530") == datetime(2026, 10, 6, 5, 0, tzinfo=timezone.utc)
        assert parse_date("garbage") is None and parse_date(None) is None

    def test_urlset_with_cdata_image_keywords(self):
        kind, items = parse_sitemap(urlset(news_url("https://j.example/a", "2026-10-06T05:00:00+00:00",
                                                    title="विपक्षी दलों &amp; CEC", image="https://j.example/i.jpg",
                                                    keywords="CEC, Election")))
        assert kind == "urlset" and len(items) == 1
        it = items[0]
        assert it["title"] == "विपक्षी दलों & CEC" and it["language"] == "hi" and it["news"]
        assert it["image"] == "https://j.example/i.jpg" and it["keywords"] == "CEC, Election"

    def test_title_comes_from_news_block_not_image_title(self):
        # Mathrubhumi: <image:title> (a file name) precedes <news:title> in each <url>.
        xml = urlset(
            "<url><loc>https://m.example/a</loc>"
            "<image:image><image:loc>https://img.example/x.jpg</image:loc><image:title>Mohanlal.jpg</image:title></image:image>"
            "<news:news><news:publication><news:name>M</news:name><news:language>ml</news:language></news:publication>"
            "<news:publication_date>2026-10-06T08:17:09Z</news:publication_date>"
            "<news:title>Jailer 2 trailer</news:title></news:news></url>")
        _, items = parse_sitemap(xml)
        assert items[0]["title"] == "Jailer 2 trailer"
        assert items[0]["language"] == "ml"
        assert items[0]["image"] == "https://img.example/x.jpg"

    def test_index(self):
        kind, items = parse_sitemap(index(("https://b.example/s1.xml", "2026-10-06T05:00:00Z")))
        assert kind == "index" and items[0]["loc"] == "https://b.example/s1.xml"

    @pytest.mark.parametrize("code,want", [("hi", "hi"), ("en-IN", "en"), ("tn", "ta"), ("Hindi", "hi"),
                                           ("od", "or"), ("", ""), ("zh-cn", "zh"), ("??", "")])
    def test_language_normalisation(self, code, want):
        assert normalize_language(code) == want


class TestFetch:

    def test_window_and_feedparser_shape(self):
        page = urlset(
            news_url("https://p.example/new", "2026-10-06T05:00:00+00:00", title="New", lang="tn",
                     image="https://p.example/i.jpg"),
            news_url("https://p.example/old", "2026-10-03T05:00:00+00:00"),            # beyond 48 h
            news_url("https://p.example/future", "2026-10-09T05:00:00+00:00"),         # scheduled
            news_url("https://p.example/new", "2026-10-06T05:00:00+00:00"),            # duplicate loc
        )
        entries, meta, err = fetch_news_sitemap("https://p.example/sm.xml",
                                                downloader({"https://p.example/sm.xml": page}), now=NOW)
        assert err is None and [e["link"] for e in entries] == ["https://p.example/new"]
        e = entries[0]
        assert e["title"] == "New" and e["language"] == "ta"
        assert e["media_thumbnail"] == [{"url": "https://p.example/i.jpg"}]
        assert datetime(*e["published_parsed"][:6], tzinfo=timezone.utc) == datetime(2026, 10, 6, 5, tzinfo=timezone.utc)

    def test_incremental_since(self):
        page = urlset(news_url("https://p.example/a", "2026-10-06T05:00:00Z"),
                      news_url("https://p.example/b", "2026-10-06T01:00:00Z"))
        entries, _, _ = fetch_news_sitemap("u", downloader({"u": page}), since=NOW - timedelta(hours=2), now=NOW)
        assert [e["link"] for e in entries] == ["https://p.example/a"]

    def test_index_follows_recent_children_and_tolerates_broken_ones(self):
        pages = {
            "https://b.example/index.xml": index(("https://b.example/mp.xml", "2026-10-06T05:00:00Z"),
                                                 ("https://b.example/up.xml", "2026-10-06T04:00:00Z"),
                                                 ("https://b.example/dead.xml", "2026-10-06T04:00:00Z"),
                                                 ("https://b.example/stale.xml", "2026-09-01T00:00:00Z")),
            "https://b.example/mp.xml": urlset(news_url("https://b.example/mp/1", "2026-10-06T05:00:00Z")),
            "https://b.example/up.xml": urlset(news_url("https://b.example/up/1", "2026-10-06T04:00:00Z")),
            "https://b.example/dead.xml": None,
        }
        dl = downloader(pages)
        entries, meta, err = fetch_news_sitemap("https://b.example/index.xml", dl, now=NOW)
        assert err is None and sorted(e["link"] for e in entries) == ["https://b.example/mp/1", "https://b.example/up/1"]
        assert meta["children"] == 3 and meta["child_errors"] == 1
        assert "https://b.example/stale.xml" not in dl.calls       # older than the window: not fetched

    def test_all_children_failing_is_an_error(self):
        pages = {"i": index(("https://x/a.xml", "2026-10-06T05:00:00Z")), "https://x/a.xml": None}
        entries, _, err = fetch_news_sitemap("i", downloader(pages), now=NOW)
        assert entries == [] and err

    def test_undated_entries_only_kept_from_undated_sitemaps(self):
        undated = '<urlset><url><loc>https://u.example/1</loc></url><url><loc>https://u.example/2</loc></url></urlset>'
        entries, _, _ = fetch_news_sitemap("u", downloader({"u": undated}), now=NOW)
        assert len(entries) == 2
        mixed = urlset(news_url("https://m.example/1", "2026-10-06T05:00:00Z"),
                       news_url("https://m.example/2", "2026-10-06T05:00:00Z"),
                       "<url><loc>https://m.example/undated</loc></url>")
        entries, _, _ = fetch_news_sitemap("m", downloader({"m": mixed}), now=NOW)
        assert "https://m.example/undated" not in [e["link"] for e in entries]

    def test_download_failure(self):
        entries, _, err = fetch_news_sitemap("nope", downloader({}), now=NOW)
        assert entries == [] and err.startswith("fetch failed")

    def test_since_for_feed(self):
        assert since_for_feed({}) is None
        assert since_for_feed({"last_success_at": "2026-10-06T05:00:00+00:00"}) == \
            datetime(2026, 10, 6, 2, tzinfo=timezone.utc)


class TestPollerIntegration:

    def test_sitemap_feed_goes_through_the_same_pipeline(self, monkeypatch, tmp_path):
        page = urlset(news_url("https://p.example/a", (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                               title="Sitemap headline", lang="hi"),
                      news_url("https://p.example/b", (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                               title="Marathi-declared headline", lang="mr"))
        monkeypatch.setattr(poller, "_download_feed", lambda url, **kw: (page.encode(), {}, url))
        monkeypatch.setattr(poller, "_fetch_rss_blocking", lambda url: pytest.fail("RSS path used for a sitemap"))
        monkeypatch.setattr(poller, "CRAWL_AT_INGEST", False)
        monkeypatch.setattr(db, "existing_hashes", lambda hashes: set())
        sent = []
        monkeypatch.setattr(db, "upsert_articles_returning",
                            lambda rows: (sent.extend(rows) or [dict(r, id=1) for r in rows], 0))
        feed = {"id": "s1", "feed_url": "https://p.example/sm.xml", "feed_type": "news_sitemap",
                "update_cadence": "breaking_news", "is_active": True, "fail_count": 0,
                "language_code": "en", "last_new_article_at": None, "created_at": "2000-01-01T00:00:00+00:00"}
        polls = db.FeedPollBatch()

        async def go():
            loop = asyncio.get_running_loop()
            return await poller.poll_one_feed(feed, asyncio.Semaphore(2), asyncio.Semaphore(2), set(), loop,
                                              polls=polls)
        res = asyncio.run(go())
        assert res["new"] == 1 and sent[0]["title"] == "Sitemap headline"
        assert sent[0]["language_code"] == "hi"                 # entry language beats the feed default
        assert len(sent) == 1                                   # the "mr" entry is outside the en/hi scope
        assert polls.items[0]["success"] and polls.items[0]["new_articles"] == 1

    def test_discover_uses_entry_language(self):
        a = discover_article({"title": "T", "language": "te"}, {"id": "f", "language_code": "en"},
                             "https://x/a", 1, 0)
        assert a.language_code == "te"
