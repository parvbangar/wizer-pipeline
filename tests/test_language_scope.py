"""tests/test_language_scope.py — ingestion keeps English and Hindi articles only (2026-10-07)."""

from __future__ import annotations

import pytest

from pipeline.language_scope import article_language, in_scope


@pytest.mark.parametrize("title, desc, declared, want", [
    ("RBI keeps repo rate unchanged", "", "en", "en"),
    ("आरबीआई ने रेपो रेट में कोई बदलाव नहीं किया", "", "en", "hi"),        # Hindi item in an "en" feed
    ("मुंबईत पावसाचा जोर वाढला, अनेक भागांत पाणी साचले आहे", "", "hi", "mr"),   # Marathi in a "hi" sitemap
    ("சென்னையில் கனமழை", "", "en", "ta"),
    ("ఆంధ్రప్రదేశ్ లో భారీ వర్షాలు", "", "te", "te"),
    ("পশ্চিমবঙ্গে ভারী বৃষ্টি", "", "bn", "bn"),
    ("پاکستان میں بارش", "", "ur", "ur"),
    ("Российский рубль упал", "", "en", "other"),
    ("Le gouvernement annonce", "", "fr", "fr"),                          # Latin: trust the declaration
    ("Modi ne kaha ki desh aage badhega", "", "hi", "hi"),                 # romanised Hindi from a Hindi source
    ("", "", "hi", "hi"),                                                 # nothing to read: declared
    # 2026-10-07 purge leftovers:
    ("विना पदवी वैद्यकीय व्यवसाय करणाऱ्यावर गुन्हा", "", "mr", "mr"),        # no marker words: declaration decides
    ("विना पदवी वैद्यकीय व्यवसाय करणाऱ्यावर गुन्हा", "", "hi", "hi"),
    ("सरकार ने कहा है कि योजना जारी रहेगी", "", "mr", "hi"),                  # Hindi evidence beats a "mr" label
    ("Trump-Blames-Ukraine-Democrats : అంతా వీళ్ల వ‌ల్లే,,! Andhra Prabha Top News", "", "en", "te"),
    ("Brazil-Election : గేమ్ చేంజ్ ..!,,Andhra Prabha Top Story", "", "en", "te"),
    ("Bollywood star says 'धन्यवाद' to fans after the premiere in Mumbai", "", "en", "en"),
    ("Sensex today: बाजार में तेजी, निफ्टी 25000 के पार", "", "en", "hi"),
    ("", "", None, "en"),
])
def test_article_language(title, desc, declared, want):
    assert article_language(title, desc, declared) == want


def test_scope_is_english_and_hindi():
    assert in_scope("en") and in_scope("hi")
    for lang in ("mr", "ta", "te", "bn", "ur", "gu", "kn", "ml", "or", "pa", "as", "fr", "other"):
        assert not in_scope(lang)
