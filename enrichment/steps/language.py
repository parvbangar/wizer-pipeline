"""
enrichment/steps/language.py
════════════════════════════
Detects the actual language of an article (headline / description / body).

WHY NOT THE FEED'S language_code: feeds mislabel (an 'en' feed publishes a
Hindi article; wire services republish in several languages; many feeds have
no language at all). language_detected routes every later step.

HOW — SCRIPT FIRST, THEN WORDS, THEN A STATISTICAL MODEL:
  1. Count letters per Unicode script. Seven of India's major languages own
     their script outright: Tamil, Telugu, Kannada, Malayalam, Gujarati,
     Gurmukhi (Punjabi), Odia. A dominant script decides those directly.
  2. Shared scripts are disambiguated by letters and function words:
       Bengali script → Assamese if it uses ৰ / ৱ (Assamese-only letters),
                        else Bengali
       Devanagari     → Marathi vs Hindi by function words ("आहे", "आणि",
                        "च्या"… vs "है", "और", "में"…) and the Marathi ळ;
                        Nepali by its own markers ("छ", "गरेको"…)
       Arabic script  → Urdu (the Indian context)
  3. Latin script (and anything else) → lingua, as before. Romanised Hindi
     ("sarkar ne kaha ki…") is recognised by its function words and returned
     as "hi-latn", so it is not mistaken for English.

MEASURED (tools/language_eval: 3,000 publisher-labelled sitemap headlines,
13 Indian languages, 2026-10-06):
  lingua alone       71.1 %  — Malayalam, Kannada, Odia, Assamese 0 %
                               (Odia came out as Welsh, Assamese as Bengali)
  this detector      see docs/LANGUAGE.md (re-run tools/language_eval/evaluate.py)
"""

from __future__ import annotations

import logging
import re
import unicodedata

from enrichment.config import LANG_DETECT_MIN_CHARS

log = logging.getLogger(__name__)

# Unicode blocks → script name.
_SCRIPT_RANGES = (
    (0x0900, 0x097F, "deva"), (0x0980, 0x09FF, "beng"), (0x0A00, 0x0A7F, "guru"),
    (0x0A80, 0x0AFF, "gujr"), (0x0B00, 0x0B7F, "orya"), (0x0B80, 0x0BFF, "taml"),
    (0x0C00, 0x0C7F, "telu"), (0x0C80, 0x0CFF, "knda"), (0x0D00, 0x0D7F, "mlym"),
    (0x0600, 0x06FF, "arab"), (0x0750, 0x077F, "arab"), (0xFB50, 0xFDFF, "arab"), (0xFE70, 0xFEFF, "arab"),
    (0xABC0, 0xABFF, "mtei"), (0x1C50, 0x1C7F, "olck"),
)
# Scripts that identify one language on their own.
_SCRIPT_LANG = {"guru": "pa", "gujr": "gu", "orya": "or", "taml": "ta", "telu": "te",
                "knda": "kn", "mlym": "ml", "arab": "ur", "mtei": "mni", "olck": "sat"}
_DOMINANT = 0.5          # share of letters a script needs to decide

_ASSAMESE_LETTERS = set("ৰৱ")
_MARATHI_WORDS = {"आहे", "आहेत", "आणि", "होते", "झाले", "झाली", "केले", "केली", "नाही", "त्यांच्या",
                  "त्यांनी", "यांच्या", "मध्ये", "साठी", "म्हणून", "पण", "आता", "हे", "ही", "या", "व",
                  "तर", "असे", "करणार", "होणार", "दिली", "घेतला", "केला", "आली", "आला", "लागले",
                  "काय", "कोण", "करेल", "घ्या", "तुम्ही", "आम्ही", "त्यांना", "ची", "चा", "चे", "चं",
                  "जाणून", "कसे", "कसा", "कधी", "इथे", "तिथे", "आज", "उद्या", "मात्र", "म्हणजे", "होईल"}
# Inflections that are Marathi and not Hindi ("-ला"/"-ने" are not: मामला, जाने).
_MARATHI_SUFFIXES = ("च्या", "ांना", "ांनी", "तील", "थील", "ून", "णार", "णाऱ्या", "मध्ये",
                     "लेल्या", "ल्याने", "लाही", "ताना", "तेय", "तोय", "लेले", "ल्यास", "णारे")
# Marathi genitive -चा/-ची/-चे after a vowel sign or anusvara (घरांची, बंधूंचा,
# सुविधांचा). Hindi चर्चा has a virama there, so it does not match.
_MARATHI_GENITIVE = re.compile("[ा-ौं]च[ाीे]$")
_HINDI_WORDS = {"है", "हैं", "और", "में", "की", "के", "का", "को", "से", "पर", "ने", "भी", "था", "थी", "थे",
                "गया", "गई", "किया", "करने", "होगा", "रहा", "रही", "रहे", "लिए", "यह", "वह", "कि", "जो",
                "तो", "नहीं", "हुआ", "हुई", "बाद", "साथ", "अब", "एक", "कहा", "दिया"}
_NEPALI_WORDS = {"छ", "छन्", "गरेको", "भएको", "गर्न", "हुने", "पनि", "लागि", "थियो", "गरे", "भने"}
_ROMANISED_HINDI = {"hai", "hain", "ke", "ki", "ka", "ko", "mein", "aur", "nahi", "nahin", "kya", "bhi",
                    "tha", "thi", "kar", "karne", "gaya", "gayi", "liye", "wala", "wali", "raha", "rahi",
                    "hua", "hui", "kaha", "par", "se", "bhai", "sarkar", "ab", "yeh", "woh"}
_WORD = re.compile(r"[\wऀ-෿]+", re.UNICODE)

_detector = None


# lingua decides only what the script rules leave open (Latin and other
# scripts). Restricting it to languages an Indian news corpus actually carries
# removes absurd guesses on short Latin headlines (Tagalog, Nynorsk, Swahili,
# Welsh — all seen on 2026-10-06 headlines).
_LINGUA_LANGUAGES = ("ENGLISH", "HINDI", "MARATHI", "BENGALI", "TAMIL", "TELUGU", "GUJARATI", "PUNJABI",
                     "URDU", "FRENCH", "GERMAN", "SPANISH", "PORTUGUESE", "ITALIAN", "DUTCH", "RUSSIAN",
                     "UKRAINIAN", "CHINESE", "JAPANESE", "KOREAN", "ARABIC", "PERSIAN", "TURKISH",
                     "INDONESIAN", "MALAY", "THAI", "VIETNAMESE", "HEBREW", "GREEK", "POLISH")


def _get_detector():
    """lingua over _LINGUA_LANGUAGES; built once, ~300 ms."""
    global _detector
    if _detector is None:
        from lingua import Language, LanguageDetectorBuilder
        langs = [getattr(Language, n) for n in _LINGUA_LANGUAGES if hasattr(Language, n)]
        _detector = LanguageDetectorBuilder.from_languages(*langs).build()
        log.info("Lingua language detector loaded (%d languages)", len(langs))
    return _detector


def script_counts(text: str) -> dict[str, int]:
    """Letters per script ('latn' for Latin letters, 'other' for the rest)."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha() and unicodedata.category(ch) not in ("Mn", "Mc"):
            continue
        cp = ord(ch)
        script = None
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                script = name
                break
        if script is None:
            script = "latn" if cp < 0x0250 else "other"
        counts[script] = counts.get(script, 0) + 1
    return counts


def _devanagari_language(text: str) -> str:
    words = _WORD.findall(text)
    mr = sum(1 for w in words if w in _MARATHI_WORDS) + 2 * text.count("ळ")
    mr += sum(1 for w in words if len(w) > 3 and w not in _HINDI_WORDS
              and (w.endswith(_MARATHI_SUFFIXES) or _MARATHI_GENITIVE.search(w)))
    hi = sum(1 for w in words if w in _HINDI_WORDS)
    ne = sum(1 for w in words if w in _NEPALI_WORDS)
    if ne > max(mr, hi):
        return "ne"
    return "mr" if mr > hi else "hi"


def _romanised_hindi(text: str) -> bool:
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", text)]
    if len(words) < 4:
        return False
    hits = sum(1 for w in words if w in _ROMANISED_HINDI)
    return hits >= 3 and hits / len(words) >= 0.2


def detect_language(full_text: str, description: str, title: str) -> str | None:
    """
    ISO 639-1 code of the article's language ('hi', 'ta', 'en', …), 'hi-latn'
    for romanised Hindi, or None when the text is too short / undecidable.
    Uses the richest text available (body, then description, then headline).
    """
    text = ((full_text or "").strip() or (description or "").strip() or (title or "").strip())
    sample = text[:3000]
    counts = script_counts(sample)
    letters = sum(counts.values())
    # A script-unique language is certain from a handful of letters; the
    # statistical fallback needs LANG_DETECT_MIN_CHARS of text.
    if letters < 4:
        return None
    script, n = max(counts.items(), key=lambda kv: kv[1])
    if n / letters >= _DOMINANT:
        if script in _SCRIPT_LANG:
            return _SCRIPT_LANG[script]
        if script == "beng":
            return "as" if any(ch in _ASSAMESE_LETTERS for ch in sample) else "bn"
        if script == "deva":
            return _devanagari_language(sample)
        if script == "latn" and _romanised_hindi(sample):
            return "hi-latn"
    if len(text) < LANG_DETECT_MIN_CHARS:
        return None
    try:
        lang = _get_detector().detect_language_of(sample)
    except Exception as e:                       # lingua missing / failure: script alone was not enough
        log.debug("lingua failed: %s", e)
        return None
    return lang.iso_code_639_1.name.lower() if lang is not None else None
