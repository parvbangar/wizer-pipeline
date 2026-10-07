"""
tests/test_official.py — official Indian sources (pipeline/official.py) and the
crawler features they need (PDF text, PIB dates, naive-IST feed dates).

Fixtures in tests/fixtures/official/ are real responses captured 2026-10-06
(NSE / BSE feeds trimmed to their first 250 / 400 items).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import feedparser
import pytest

from pipeline import crawler, official
from pipeline.crawler import crawl_record, discover_article

FIX = Path(__file__).parent / "fixtures" / "official"


def fixture(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8", errors="replace")


class TestPIB:

    def test_listing_parses_ministry_and_canonical_link(self):
        entries = official.parse_pib_listing(fixture("pib_allrel_en.html"), "3", "1")
        assert len(entries) == 4
        e = entries[0]
        assert e["link"] == "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2319428"   # PRID only: dedup across offices
        assert e["title"].startswith("Prime Minister shares Sanskrit Subhashitam")
        assert [t["term"] for t in e["tags"]] == ["Prime Minister's Office", "PIB"]
        assert e["language"] == "en"

    def test_language_from_listing_id(self):
        html = ("<h3 class='font104'>मंत्रालय</h3><ul class='num'><li><a title='T' "
                "href='/PressReleaseDetail.aspx?PRID=99' target=\"_blank\">शीर्षक</a></li></ul></div>")
        e = official.parse_pib_listing(html, "5", "16")[0]
        assert e["language"] == "te" and e["title"] == "शीर्षक" and e["tags"][0]["term"] == "मंत्रालय"

    def test_previous_day_swept_after_midnight(self):
        class Client:
            def __init__(self):
                self.posts = []

            def get(self, url):
                return type("R", (), {"status_code": 200, "text": fixture("pib_allrel_en.html")})()

            def post(self, url, data):
                self.posts.append(data)
                html = ("<input type='hidden' name='x' value='y'/><h3 class='font104'>M</h3>"
                        "<ul><li><a title='Late' href='/PressReleaseDetail.aspx?PRID=1' >Late</a></li></ul></div>")
                return type("R", (), {"status_code": 200, "text": html})()
        c = Client()
        entries, meta, err = official.fetch_pib_daily("https://www.pib.gov.in/allRel.aspx?reg=3&lang=1", c,
                                                      now=datetime(2026, 10, 5, 19, 0, tzinfo=timezone.utc))  # 00:30 IST
        assert err is None and meta["yesterday"] == 1 and len(entries) == 5
        assert c.posts[0]["ctl00$ContentPlaceHolder1$ddlday"] == "5"            # 5 Oct, yesterday in IST
        c2 = Client()
        official.fetch_pib_daily("https://www.pib.gov.in/allRel.aspx?reg=3&lang=1", c2,
                                 now=datetime(2026, 10, 6, 6, 0, tzinfo=timezone.utc))
        assert c2.posts == []                                                   # daytime: today only


class TestExchanges:

    def test_exchange_date_is_ist(self):
        assert official.exchange_date("06-Oct-2026 12:39:10") == datetime(2026, 10, 6, 7, 9, 10, tzinfo=timezone.utc)
        assert official.exchange_date("garbage") is None

    def test_nse_drops_nav_rows_and_keeps_filings(self):
        xml = fixture("nse_online_announcements.xml")
        entries = official.parse_nse_announcements(xml)
        assert 0 < len(entries) < xml.count("<item>")
        e = entries[0]
        assert e["title"].startswith("Ujjivan Small Finance Bank Limited: ")
        assert e["link"].endswith(".pdf") and e["published"] == "2026-10-06T07:09:10+00:00"
        assert [t["term"] for t in e["tags"]][:2] == ["NSE", "corporate filing"] and e["tags"][2]["term"] == "UJJIVANBANK"
        assert not any("NAV" in x.get("summary", "") and "as on" in x.get("summary", "").lower() for x in entries)

    def test_bse_drops_fund_navs_but_not_companies_named_nav(self):
        entries = official.parse_bse_announcements(fixture("bse_announcements.xml"))
        titles = [e["title"] for e in entries]
        assert entries and not any(":" in t and t.split(":", 1)[1].strip().upper().startswith(("NAV ", "ISIF NAV"))
                                   for t in titles)
        assert all(e["tags"][0]["term"] == "BSE" for e in entries)
        assert official._NAV.search("iSIF NAV as on 05.10.2026") and official._NAV.search("NAV Upload 05-10-2026")
        assert not official._NAV.search("Navneet Education compliance certificate")
        assert not official._NAV.search("Navi Mumbai land allotment")


class TestSEBI:

    def test_listing_rows(self):
        entries = official.parse_sebi_listing(fixture("sebi_listing.html"))
        assert len(entries) == 25
        e = entries[1]
        assert e["title"] == "Pentacle Consultants (I) Limited - DRHP"              # embedded PDF link stripped
        assert e["link"].endswith("_105036.html")
        assert [t["term"] for t in e["tags"]] == ["SEBI", "Public Issues"]
        assert e["published"] == "2026-10-05T18:30:00+00:00"                      # 6 Oct IST


class TestDispatch:

    def test_unknown_type(self):
        assert official.fetch_official("nope", "u", client=object())[2].startswith("unknown")

    def test_http_error_is_reported_not_raised(self):
        class C:
            def get(self, url):
                return type("R", (), {"status_code": 503, "text": ""})()
        entries, _, err = official.fetch_official("nse_announcements", "u", client=C())
        assert entries == [] and "503" in err


class TestCrawlerSupport:

    def test_pdf_url_crawled_as_pdf(self, monkeypatch):
        crawler.reset_domain_failures()
        monkeypatch.setattr(crawler, "_fetch_pdf_text", lambda url, timeout=30: "Board meeting outcome text")
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", lambda *a, **k: pytest.fail("HTML fetch for a PDF"))
        out = crawl_record({"url": "https://nsearchives.nseindia.com/corporate/X_06102026.pdf", "title": "t"})
        assert out == {"full_text": "Board meeting outcome text", "is_crawled": True, "crawl_strategy": "pdf"}

    def test_thin_page_follows_embedded_pdf_viewer(self, monkeypatch):
        crawler.reset_domain_failures()
        page = ("<html><body><p>Home » Orders</p><iframe src='../../../web/?file=https://www.sebi.gov.in/"
                "sebi_data/attachdocs/sep-2026/ORDER_1.pdf' width='100%'></iframe>"
                "<a href='https://www.sebi.gov.in/sebi_data/commondocs/other.pdf'>x</a></body></html>")
        fetched = []
        monkeypatch.setattr(crawler, "_fetch_with_fallbacks", lambda url, paywalled=False: (page, "default"))
        monkeypatch.setattr(crawler, "_fetch_pdf_text",
                            lambda url, timeout=30: fetched.append(url) or "ADJUDICATION ORDER " * 50)
        out = crawl_record({"url": "https://www.sebi.gov.in/enforcement/orders/sep-2026/x_1.html", "title": "t"})
        assert fetched == ["https://www.sebi.gov.in/sebi_data/attachdocs/sep-2026/ORDER_1.pdf"]
        assert out["crawl_strategy"] == "default+pdf" and out["full_text"].startswith("ADJUDICATION ORDER")

    def test_pib_posted_on_date(self):
        html = "<span id='PrDateTime'>Posted On: 06 OCT 2026 9:53AM by PIB Delhi</span>"
        assert crawler._page_specific_date(html, "https://www.pib.gov.in/PressReleasePage.aspx?PRID=1") == \
            datetime(2026, 10, 6, 4, 23, tzinfo=timezone.utc)
        assert crawler._page_specific_date(html, "https://example.com/x") is None

    def test_naive_rbi_date_read_as_ist_on_indian_feeds_only(self):
        e = feedparser.parse(fixture("rbi_pressreleases.xml").encode()).entries[0]
        assert e["published"] == "Tue, 06 Oct 2026 10:30:00"
        indian = discover_article(e, {"id": "r", "country_code": "IND"}, e.link, 1, 0)
        foreign = discover_article(e, {"id": "r", "country_code": "USA", "language_code": "en"}, e.link, 1, 0)
        assert indian.published_at == datetime(2026, 10, 6, 5, 0, tzinfo=timezone.utc)
        # A foreign feed's naive date stays as given (UTC) — only capped at crawl time if in the future.
        assert foreign.published_at == min(datetime(2026, 10, 6, 10, 30, tzinfo=timezone.utc), foreign.crawled_at)

    @pytest.mark.parametrize("raw,naive", [("Tue, 06 Oct 2026 10:30:00", True), ("Tue, 06 Oct 2026 10:30:00 +0530", False),
                                           ("Tue, 06 Oct 2026 10:30:00 GMT", False), ("2026-10-06T10:30:00Z", False),
                                           ("Tue, 06 Oct 2026 10:30:00 IST", False), ("", False)])
    def test_naive_detection(self, raw, naive):
        assert crawler._naive_feed_date({"published": raw}) is naive


class TestRegistry:

    def test_sources_are_complete_and_unique(self):
        from tools.register_official_sources import sources
        rows = sources()
        urls = [r["feed_url"] for r in rows]
        assert len(urls) == len(set(urls))
        types = {r["feed_type"] for r in rows}
        assert {"pib_daily", "nse_announcements", "bse_announcements", "sebi_listing", "rss"} <= types
        assert all(r["update_cadence"] == "official" and r["country_code"] == "IND" for r in rows)
        assert {r["language_code"] for r in rows if r["feed_type"] == "pib_daily"} == {"en", "hi"}   # en/hi scope (2026-10-07)
        assert all(r["language_code"] in ("en", "hi") for r in rows)
