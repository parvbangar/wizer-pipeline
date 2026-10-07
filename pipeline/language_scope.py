"""
pipeline/language_scope.py
══════════════════════════
Which articles ingestion keeps: English and Hindi only (decided 2026-10-07;
WIZER_LANGUAGES overrides). Applied per article, not per feed, because one
publisher's feed or sitemap can mix languages.

Decision on headline + description, no model (ingest has no ML stack):
  - mostly Devanagari  → Hindi vs Marathi by function words and inflections
                         (enrichment.steps.language); only Hindi is kept;
  - mostly another script with its own language (Tamil, Bengali, Urdu …) → that language;
  - Latin script       → the language the feed / sitemap declares (romanised
                         Hindi from a Hindi source stays Hindi);
  - too little text    → the declared language.
"""

from __future__ import annotations

import os

from enrichment.steps.language import _SCRIPT_LANG, devanagari_scores, script_counts

LANGUAGES = frozenset(
    c.strip().lower() for c in os.getenv("WIZER_LANGUAGES", "en,hi").split(",") if c.strip()
)

# A non-Latin script decides once it holds this share of the letters. Not a
# majority: Andhra Prabha headlines lead with an English URL slug
# ("Trump-Blames-Ukraine-Democrats : <Telugu>"), and Hindi headlines mix in
# English words. One quoted Hindi word in an English headline stays below it.
_SCRIPT_SHARE = 0.25
# Any other Indian (or non-Latin) script decides from this many letters even as a
# minority: English outlets quote Hindi, but practically never write Telugu or Tamil.
_OTHER_SCRIPT_LETTERS = 8


def _devanagari(text: str, declared: str) -> str:
    """Hindi vs Marathi vs Nepali; with no evidence either way, the declaration decides."""
    mr, hi, ne = devanagari_scores(text)
    if ne > max(mr, hi):
        return "ne"
    if mr != hi:
        return "mr" if mr > hi else "hi"
    return declared if declared in ("mr", "ne") else "hi"


def article_language(title: str, description: str, declared: str | None) -> str:
    """Best ISO 639-1 code for a discovered article (see module docstring)."""
    declared = (declared or "").strip().lower()
    text = f"{title or ''} {description or ''}"[:2000]
    counts = script_counts(text)
    letters = sum(counts.values())
    non_latin = {s: n for s, n in counts.items() if s != "latn"}
    if letters >= 4 and non_latin:
        script, n = max(non_latin.items(), key=lambda kv: kv[1])
        if n / letters >= _SCRIPT_SHARE or (script != "deva" and n >= _OTHER_SCRIPT_LETTERS):
            if script == "deva":
                return _devanagari(text, declared)
            if script in _SCRIPT_LANG:
                return _SCRIPT_LANG[script]
            if script == "beng":
                return "bn"
            return script                # another script entirely (Cyrillic, Arabic …): not in scope
    return declared or "en"


def in_scope(language: str) -> bool:
    return language in LANGUAGES
