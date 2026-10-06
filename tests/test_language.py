"""tests/test_language.py — enrichment/steps/language.py (script-first detection).

Accuracy on 3,030 publisher-labelled headlines: tools/language_eval/evaluate.py.
These cases pin each rule; the strings are real 2026-10-06 headlines unless noted.
"""

from __future__ import annotations

import pytest

from enrichment.steps.language import detect_language, script_counts


@pytest.mark.parametrize("text,lang", [
    ("ଦୁଇ ଜଣଙ୍କ ମୃତ୍ୟୁ, ଜଣେ ଆହତ: ଟ୍ରକ ଧକ୍କାରେ ଅଟୋ ଚୂରମାର", "or"),        # lingua said Welsh
    ("ಬೆಂಗಳೂರಿನಲ್ಲಿ ಭಾರಿ ಮಳೆ: ಸಂಚಾರ ಅಸ್ತವ್ಯಸ್ತ", "kn"),                  # lingua: nothing
    ("ഓർമിക്കാൻ", "ml"),                                                  # short, script-unique
    ("অসমৰ বানপানীত দুজনৰ মৃত্যু", "as"),                                # ৰ → Assamese, not Bengali
    ("কলকাতায় বৃষ্টির পূর্বাভাস", "bn"),
    ("వర్షాలతో రైతుల ఆందోళన", "te"), ("சென்னையில் கனமழை எச்சரிக்கை", "ta"),
    ("ગુજરાતમાં વરસાદની આગાહી", "gu"), ("ਪੰਜਾਬ ਵਿੱਚ ਹੜ੍ਹਾਂ ਦਾ ਖ਼ਤਰਾ", "pa"),
    ("حکومت نے نئی پالیسی کا اعلان کیا", "ur"),
    ("मुंबईत नऊ महिन्यांत ७२,८०४ घरांची विक्री", "mr"),                   # genitive -ांची
    ("दक्षिण मुंबईत ठाकरे बंधूंचा १ किमीचा लाँगमार्च", "mr"),
    ("संसद में विपक्ष ने सरकार पर चर्चा की मांग की है", "hi"),            # चर्चा is Hindi, not Marathi -चा
    ("सरकार ने कहा कि नई नीति अगले महीने से लागू होगी", "hi"),
    ("RBI keeps repo rate unchanged, says inflation easing", "en"),
    ("sarkar ne kaha ki yeh faisla bilkul sahi hai aur logon ke liye accha hai", "hi-latn"),  # constructed
])
def test_languages(text, lang):
    assert detect_language("", "", text) == lang


def test_body_beats_headline_and_too_short_is_none():
    assert detect_language("யாழ்ப்பாணத்தில் இன்று கனமழை பெய்தது " * 3, "", "Rain") == "ta"
    assert detect_language("", "", "ok") is None


def test_script_counts():
    c = script_counts("Pune Video: पुण्यातील")
    assert c["latn"] == 9 and c["deva"] > 5
