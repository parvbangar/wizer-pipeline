"""
tests/test_enrichment_steps.py
══════════════════════════════
Offline unit tests for the Layer 2 enrichment steps. Every model (spaCy,
transformers, lingua, YAKE) is replaced by a small fake, so the tests pin
down OUR logic — thresholds, salience, mapping, filtering, edge cases — and
run in CI without the 2 GB ML stack.
"""

from __future__ import annotations

import sys
import types

import pytest

from enrichment.steps import (
    classifier, keywords, language, ner, sentiment, summarizer, text_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# text_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestTextStats:

    def test_counts_words_and_reading_time(self):
        assert text_stats.compute_text_stats("word " * 410, "") == {"word_count": 410, "reading_time_mins": 2.0}

    def test_falls_back_to_description(self):
        assert text_stats.compute_text_stats("", "three words here")["word_count"] == 3

    def test_empty(self):
        assert text_stats.compute_text_stats(None, None) == {"word_count": 0, "reading_time_mins": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# summarizer
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizer:

    def test_prefers_rich_description(self):
        desc = "A journalist-written summary that is comfortably longer than one hundred and fifty " \
               "characters, which is the threshold for trusting the RSS description outright."
        assert summarizer.summarize_article("Body text.", desc) == desc

    def test_first_three_substantial_sentences(self):
        body = ("The first sentence is long enough to count here. Short one. "
                "The second real sentence also clears the bar easily. "
                "A third substantial sentence follows right after it. "
                "The fourth sentence must not be included at all.")
        out = summarizer.summarize_article(body, "")
        assert out.count(".") == 3 and "fourth" not in out and "Short one" not in out

    def test_hindi_danda_split(self):
        body = "यह पहला वाक्य काफी लंबा है ताकि गिना जाए। यह दूसरा वाक्य भी काफी लंबा है ताकि गिना जाए।"
        assert summarizer.summarize_article(body, "").count("।") == 2

    def test_html_entities_unescaped_and_capped(self):
        out = summarizer.summarize_article("", "Tata &amp; Sons " * 100, max_chars=50)
        assert "&amp;" not in out and len(out) == 50

    def test_nothing_available(self):
        assert summarizer.summarize_article("", "") is None


# ─────────────────────────────────────────────────────────────────────────────
# ner
# ─────────────────────────────────────────────────────────────────────────────

class _Ent:
    def __init__(self, text, label):
        self.text, self.label_ = text, label


class _Doc:
    def __init__(self, ents):
        self.ents = ents


class TestNer:

    def test_salience_components(self):
        assert ner._compute_salience("Modi", "Modi visits Pune", "Modi said today", 1) == 0.9
        assert ner._compute_salience("Pune", "Modi visits Pune", "x" * 300 + " Pune", 5) == 0.8
        assert ner._compute_salience("RBI", "Budget", "y" * 300, 1) == 0.1

    def test_extraction_maps_filters_and_ranks(self):
        doc = _Doc([_Ent("Modi", "PERSON"), _Ent("Modi", "PERSON"), _Ent("BJP", "NORP"),
                    _Ent("₹5 crore", "MONEY"), _Ent("iphone", "GPE"), _Ent("X", "ORG")])
        out = ner._extract_entities_spacy("Modi speaks", "Modi speaks\n\nbody", "Modi speaks today", lambda t: doc)
        texts = [e["entity_text"] for e in out]
        assert texts[0] == "Modi"
        assert "₹5 crore" not in texts            # MONEY is not stored
        assert "iphone" not in texts              # lowercase "GPE" = misclassified common noun
        assert "X" not in texts                   # single character
        assert {"entity_text": "BJP", "entity_type": "ORG", "salience": 0.1} in out   # NORP → ORG

    def test_english_vs_multilingual_dispatch(self, monkeypatch):
        used = []
        monkeypatch.setattr(ner, "_get_nlp_en", lambda: used.append("en") or (lambda t: _Doc([])))
        monkeypatch.setattr(ner, "_get_nlp_xx", lambda: used.append("xx") or (lambda t: _Doc([])))
        ner.extract_entities("t", "body", "", "en-in")
        ner.extract_entities("t", "body", "", "hi")
        ner.extract_entities("t", "body", "", None)
        assert used == ["en", "xx", "en"]

    def test_missing_model_returns_empty(self, monkeypatch):
        def missing():
            raise OSError("not installed")
        monkeypatch.setattr(ner, "_get_nlp_en", missing)
        assert ner.extract_entities("t", "body", "", "en") == []


# ─────────────────────────────────────────────────────────────────────────────
# sentiment
# ─────────────────────────────────────────────────────────────────────────────

def _sentiment_pipe(pos, neu, neg):
    return lambda text, **kw: [[{"label": "positive", "score": pos},
                                {"label": "neutral", "score": neu},
                                {"label": "negative", "score": neg}]]


class TestSentiment:

    @pytest.mark.parametrize("scores,label", [
        ((0.70, 0.20, 0.10), "positive"),
        ((0.10, 0.20, 0.70), "negative"),
        ((0.36, 0.30, 0.34), "neutral"),     # neither side confident enough
    ])
    def test_threshold_labelling(self, monkeypatch, scores, label):
        monkeypatch.setattr(sentiment, "_get_pipeline", lambda: _sentiment_pipe(*scores))
        out = sentiment.analyse_sentiment("Title", "Desc", "en")
        assert out["sentiment"] == label
        assert out["sentiment_score"] == round(scores[0] - scores[2], 4)
        assert sum(out["sentiment_stats"].values()) == pytest.approx(100, abs=0.1)

    def test_empty_text_and_failure(self, monkeypatch):
        """Regression: an empty article used to be scored as the text "."."""
        def must_not_load():
            raise AssertionError("model loaded for an empty article")
        monkeypatch.setattr(sentiment, "_get_pipeline", must_not_load)
        assert sentiment.analyse_sentiment("", "", "en")["sentiment"] is None
        assert sentiment.analyse_sentiment(None, "  ", "en")["sentiment"] is None

        def broken():
            raise RuntimeError("download failed")
        monkeypatch.setattr(sentiment, "_get_pipeline", broken)
        assert sentiment.analyse_sentiment("t", "d", "en") == {"sentiment": None, "sentiment_score": None}


# ─────────────────────────────────────────────────────────────────────────────
# classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifier:

    def test_every_label_maps_to_a_category(self):
        assert set(classifier._CANDIDATE_LABELS) == set(classifier._LABEL_TO_CATEGORY)
        assert set(classifier._TAG_CANDIDATE_LABELS) == set(classifier._TAG_LABEL_TO_TAG)

    @pytest.mark.parametrize("scores,expected", [
        ({"politics": 0.9, "crime": 0.4}, "politics"),
        ({"economy": 0.8, "politics": 0.3}, "business"),            # synonym label → category
        ({"politics": 0.29}, "general"),                            # below threshold
        ({"sports": 0.96, "cricket": 0.54}, "cricket"),             # cricket is a sub-type of sports
        ({"sports": 0.96, "cricket": 0.30}, "sports"),
        ({"international news": 0.9, "cricket": 0.8}, "world"),     # hierarchy only refines "sports"
        ({}, "general"),
    ])
    def test_pick_category(self, scores, expected):
        assert classifier.pick_category(scores) == expected

    def test_uses_news_template_and_independent_scoring(self, monkeypatch):
        seen = {}

        def pipe(text, **kw):
            seen.update(kw)
            return {"labels": ["politics"], "scores": [0.9]}
        monkeypatch.setattr(classifier, "_get_pipeline", lambda: pipe)
        classifier.classify_article("Headline", "", "")
        assert seen["multi_label"] is True
        assert seen["hypothesis_template"] == "This news article is about {}."

    def test_confident_label_wins(self, monkeypatch):
        label = classifier._CANDIDATE_LABELS[0]
        monkeypatch.setattr(classifier, "_get_pipeline",
                            lambda: lambda text, **kw: {"labels": [label], "scores": [0.9]})
        assert classifier.classify_article("IPL final", "", "") == "cricket"

    def test_low_confidence_is_general(self, monkeypatch):
        monkeypatch.setattr(classifier, "_get_pipeline", lambda: lambda text, **kw: {
            "labels": [classifier._CANDIDATE_LABELS[1]],
            "scores": [classifier.CLASSIFY_CONFIDENCE_THRESHOLD - 0.01]})
        assert classifier.classify_article("Something", "", "") == "general"

    def test_tags_threshold_and_cap(self, monkeypatch):
        labels = classifier._TAG_CANDIDATE_LABELS[:7]
        monkeypatch.setattr(classifier, "_get_pipeline", lambda: lambda text, **kw: {
            "labels": labels, "scores": [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.1]})
        tags = classifier.classify_tags("t", "d", "")
        assert len(tags) == 5 and tags[0] == "government"

    def test_empty_article_never_reaches_model(self, monkeypatch):
        """Regression: "." (from an f-string of empty parts) used to be classified."""
        def must_not_load():
            raise AssertionError("model loaded for an empty article")
        monkeypatch.setattr(classifier, "_get_pipeline", must_not_load)
        assert classifier.classify_article("", "", "") == "general"
        assert classifier.classify_tags(None, None, None) == []

    def test_classification_text(self):
        desc = "The Reserve Bank raised the repo rate by 25 basis points on Friday."
        assert classifier._classification_text("RBI hikes rate", desc, "x " * 500) == f"RBI hikes rate. {desc}"
        assert classifier._classification_text("Flood alert", "", "Heavy rain hit Assam.") ==             "Flood alert. Heavy rain hit Assam."                     # body lead when no description
        assert len(classifier._classification_text("T", "word " * 400, "")) <= classifier.CLASSIFY_MAX_CHARS

    def test_failure_degrades_gracefully(self, monkeypatch):
        def broken():
            raise RuntimeError("no model")
        monkeypatch.setattr(classifier, "_get_pipeline", broken)
        assert classifier.classify_article("t", "d", "") == "general"
        assert classifier.classify_tags("t", "d", "") == []


# ─────────────────────────────────────────────────────────────────────────────
# keywords (fake YAKE module)
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywords:

    @pytest.fixture
    def fake_yake(self, monkeypatch):
        seen = {}

        class KeywordExtractor:
            def __init__(self, **kw):
                seen.update(kw)

            def extract_keywords(self, text):
                return [("Read More", 0.01), ("Union Budget", 0.02), ("subscribe to our newsletter", 0.03),
                        ("ab", 0.04), ("tax slabs", 0.05)]

        monkeypatch.setitem(sys.modules, "yake", types.SimpleNamespace(KeywordExtractor=KeywordExtractor))
        return seen

    def test_boilerplate_and_tiny_keywords_filtered(self, fake_yake):
        out = keywords.extract_keywords("Union Budget 2026", "Finance minister presented the budget " * 5, "", "en")
        assert out == ["Union Budget", "tax slabs"]

    def test_language_specific_stopwords(self, fake_yake):
        keywords.extract_keywords("बजट", "वित्त मंत्री ने बजट पेश किया " * 5, "", "hi")
        assert fake_yake["lan"] == "hi"
        keywords.extract_keywords("Title", "Some body text that is long enough " * 3, "", "ta")
        assert fake_yake["lan"] == "en"           # no YAKE stoplist for Tamil

    def test_too_short(self, fake_yake):
        assert keywords.extract_keywords("", "tiny", "", "en") == []


# ─────────────────────────────────────────────────────────────────────────────
# language (fake lingua detector)
# ─────────────────────────────────────────────────────────────────────────────

class TestLanguage:

    def test_detects_from_richest_text(self, monkeypatch):
        seen = []

        class Det:
            def detect_language_of(self, text):
                seen.append(text)
                return types.SimpleNamespace(iso_code_639_1=types.SimpleNamespace(name="HI"))

        monkeypatch.setattr(language, "_get_detector", lambda: Det())
        # Latin text goes to the statistical detector, body first.
        assert language.detect_language("The full article body is right here.", "desc text", "title") == "hi"
        assert seen[0].startswith("The full article body")
        # Devanagari is decided by script + words, without the statistical detector.
        seen.clear()
        assert language.detect_language("पूरा लेख यहाँ है, काफी लंबा पाठ", "desc", "title") == "hi"
        assert seen == []

    def test_short_or_undetectable(self, monkeypatch):
        class Det:
            def detect_language_of(self, text):
                return None
        monkeypatch.setattr(language, "_get_detector", lambda: Det())
        assert language.detect_language("", "", "hi") is None
        assert language.detect_language("long enough text for the detector", "", "") is None


# ─────────────────────────────────────────────────────────────────────────────
# images — signed 64-bit storage of the pHash
# ─────────────────────────────────────────────────────────────────────────────

class TestImages:

    def test_non_http_url_skipped(self):
        from enrichment.steps import images
        assert images.download_and_hash_image("") is None
        assert images.download_and_hash_image("data:image/png;base64,xx") is None

    @pytest.mark.parametrize("hex_hash,expected", [
        ("0000000000000001", 1),
        ("7fffffffffffffff", 2 ** 63 - 1),
        ("8000000000000000", -(2 ** 63)),
        ("ffffffffffffffff", -1),
    ])
    def test_unsigned_hash_reinterpreted_as_signed(self, monkeypatch, hex_hash, expected):
        """The bit pattern must survive the trip into PostgreSQL bigint."""
        from enrichment.steps import images

        class Resp:
            headers = {"Content-Type": "image/jpeg"}

            def read(self, n):
                return b"img"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = types.SimpleNamespace(open=lambda b: types.SimpleNamespace(convert=lambda m: "rgb"))
        monkeypatch.setitem(sys.modules, "PIL", fake_pil)
        monkeypatch.setitem(sys.modules, "imagehash",
                            types.SimpleNamespace(phash=lambda img, hash_size: hex_hash))
        monkeypatch.setattr(images.urllib.request, "urlopen", lambda req, timeout: Resp())
        assert images.download_and_hash_image("https://x.com/a.jpg") == expected
