"""tests/test_coverage.py — tools/coverage_audit.py (completeness measurement), pure parts."""

from __future__ import annotations

from datetime import date

import pytest

from tools.coverage_audit import alerts, gkg_urls, host_key, lincoln_petersen, miss_row

DAY = date(2026, 10, 6)


@pytest.mark.parametrize("raw,key", [("https://www.Jagran.com/news/x", "jagran.com"), ("m.bhaskar.com", "bhaskar.com"),
                                     ("www.thehindu.com", "thehindu.com"), ("https://amp.ndtv.com:443/a", "ndtv.com"),
                                     ("hindi.news18.com", "hindi.news18.com")])
def test_host_key(raw, key):
    assert host_key(raw) == key


def test_lincoln_petersen_chapman():
    # we stored 900, GDELT saw 100, 90 of them we also have → ~1000 articles exist
    assert lincoln_petersen(900, 100, 90) == pytest.approx(1000, rel=0.01)
    assert lincoln_petersen(0, 10, 0) is None and lincoln_petersen(10, 0, 0) is None


def test_miss_row_sitemap_and_gdelt():
    ref = {f"https://p.in/{i}" for i in range(10)}
    stored = {f"https://p.in/{i}" for i in range(8)}
    r = miss_row(DAY, "p.in", "sitemap", ref, stored)
    assert (r["ref_count"], r["ours"], r["missing"], r["miss_rate"]) == (10, 8, 2, 0.2)
    assert r["sample_missing"] == ["https://p.in/8", "https://p.in/9"]
    g = miss_row(DAY, "p.in", "gdelt", ref, stored, ours_total=400)
    assert g["ours"] == 400 and g["est_universe"] == pytest.approx((401 * 11) / 9 - 1, rel=1e-3)


def test_gkg_urls_filters_indian_hosts():
    lines = ["\t".join(["id", "20261006", "1", "src", url, "rest"]) for url in
             ("https://www.jagran.com/a", "https://www.nytimes.com/b", "https://somepaper.co.in/c",
              "https://www.unknown-indian.in/d", "not-a-url")]
    got = list(gkg_urls(lines, {"jagran.com"}))
    assert got == [("jagran.com", "https://www.jagran.com/a"), ("somepaper.co.in", "https://somepaper.co.in/c"),
                   ("unknown-indian.in", "https://www.unknown-indian.in/d")]


def test_alerts():
    rows = [
        {"reference": "sitemap", "domain": "big.in", "ref_count": 500, "missing": 30, "miss_rate": 0.06,
         "sample_missing": ["u1", "u2"], "ours": 470},
        {"reference": "sitemap", "domain": "small.in", "ref_count": 10, "missing": 5, "miss_rate": 0.5,
         "sample_missing": [], "ours": 5},                                   # too small to alert on
        {"reference": "ingest", "domain": "dead.in", "ref_count": 3, "ours": 3},
        {"reference": "ingest", "domain": "fine.in", "ref_count": 90, "ours": 90},
    ]
    out = alerts(rows, {"dead.in": 200, "fine.in": 100}, {"big.in", "small.in"})
    assert len(out) == 2 and out[0].startswith("big.in: missed 30 of 500") and out[1].startswith("dead.in")
