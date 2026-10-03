"""
tests/test_ingestion_dedup.py
════════════════════════════
Regression tests for the Layer 1 dedup fixes (items 8 and 9):
  * hamming_distance on signed int64 values
  * normalise_url: http/https collapse, uppercase scheme, strip-list policy
  * legacy normalisation / hash kept for the transition period
"""

import pytest

from pipeline.dedup import (
    normalise_url, _legacy_normalise_url, url_hash, legacy_url_hash,
    hamming_distance, simhash, is_near_duplicate,
)


# ── hamming_distance (item 9) ────────────────────────────────────────────────

class TestHammingSigned:

    def test_negative_vs_zero(self):
        # -1 is all 64 bits set in int64
        assert hamming_distance(-1, 0) == 64

    def test_sign_bit_only(self):
        assert hamming_distance(-(1 << 63), 0) == 1

    def test_both_negative(self):
        assert hamming_distance(-1, -2) == 1          # ...1111 vs ...1110

    def test_signed_equals_unsigned_view(self):
        # The same 64-bit pattern as signed and unsigned must be distance 0.
        assert hamming_distance(-1, (1 << 64) - 1) == 0

    def test_int64_extremes(self):
        assert hamming_distance(-(1 << 63), (1 << 63) - 1) == 64

    def test_symmetric(self):
        a, b = -123456789, 987654321
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_near_duplicate_with_negative_hashes(self):
        h = simhash("Budget 2024: Finance Minister announces tax relief")
        assert is_near_duplicate(h, h)


# ── normalise_url (item 8) ───────────────────────────────────────────────────

class TestNormaliseUrlFixes:

    def test_http_and_https_same_hash(self):
        assert url_hash("http://example.com/a") == url_hash("https://example.com/a")
        assert normalise_url("http://www.example.com/a/") == "https://example.com/a"

    def test_uppercase_scheme(self):
        assert normalise_url("HTTP://X.COM/a") == "https://x.com/a"
        assert normalise_url("HTTPS://X.COM/a") == "https://x.com/a"

    def test_default_ports_dropped(self):
        assert normalise_url("http://x.com:80/a") == "https://x.com/a"
        assert normalise_url("https://x.com:443/a") == "https://x.com/a"

    def test_non_default_port_kept(self):
        assert normalise_url("https://x.com:8443/a") == "https://x.com:8443/a"

    def test_protocol_relative(self):
        assert normalise_url("//x.com/a") == "https://x.com/a"

    def test_keep_scheme_mode_preserves_http(self):
        # Used for the URL we actually fetch/store (http-only publishers).
        assert normalise_url("http://x.com/a?utm_source=t", collapse_scheme=False) == "http://x.com/a"
        assert normalise_url("HTTP://X.com/a", collapse_scheme=False) == "http://x.com/a"

    def test_idempotent(self):
        u = "http://www.X.com:80//a//b/?b=2&a=1&utm_medium=z#frag"
        once = normalise_url(u)
        assert normalise_url(once) == once

    @pytest.mark.parametrize("param", ["sid", "source", "origin", "share"])
    def test_ambiguous_params_kept(self, param):
        """These double as article ids on some sites, so distinct values must
        stay distinct (they used to be stripped and collapse articles)."""
        a = normalise_url(f"https://news.example.com/story?{param}=1")
        b = normalise_url(f"https://news.example.com/story?{param}=2")
        assert a != b
        assert url_hash(a) != url_hash(b)

    @pytest.mark.parametrize("param", ["utm_source", "utm_whatever", "fbclid",
                                       "gclid", "ref", "session_id", "hsa_cre"])
    def test_tracking_params_still_stripped(self, param):
        assert normalise_url(f"https://x.com/a?{param}=abc") == "https://x.com/a"


# ── legacy compatibility ─────────────────────────────────────────────────────

class TestLegacyNormalisation:

    def test_legacy_matches_old_behaviour(self):
        assert _legacy_normalise_url("https://www.ndtv.com/s/?utm_source=a&sid=9&source=x") \
            == "https://ndtv.com/s"
        # http stays http in the legacy form
        assert _legacy_normalise_url("http://ndtv.com/s") == "http://ndtv.com/s"

    def test_legacy_hash_differs_for_http_urls(self):
        u = "http://ndtv.com/s"
        assert legacy_url_hash(u) != url_hash(u)
        # ...but is exactly what the old url_hash produced for an https twin
        assert legacy_url_hash("https://ndtv.com/s") == url_hash("https://ndtv.com/s")

    def test_clean_https_url_hash_unchanged(self):
        """The common case (clean https URL) must keep its stored hash, so
        most already-stored articles need no legacy lookup at all."""
        u = "https://www.thehindu.com/news/art123.ece?utm_source=tw"
        assert legacy_url_hash(u) == url_hash(u)

    def test_legacy_hash_in_signed_range(self):
        assert -(2**63) <= legacy_url_hash("http://x.com/a?sid=1") < 2**63
