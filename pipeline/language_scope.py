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

from enrichment.steps.language import _DOMINANT, _SCRIPT_LANG, _devanagari_language, script_counts

LANGUAGES = frozenset(
    c.strip().lower() for c in os.getenv("WIZER_LANGUAGES", "en,hi").split(",") if c.strip()
)


def article_language(title: str, description: str, declared: str | None) -> str:
    """Best ISO 639-1 code for a discovered article (see module docstring)."""
    declared = (declared or "").strip().lower()
    text = f"{title or ''} {description or ''}"[:2000]
    counts = script_counts(text)
    letters = sum(counts.values())
    if letters >= 4:
        script, n = max(counts.items(), key=lambda kv: kv[1])
        if n / letters >= _DOMINANT:
            if script == "deva":
                return _devanagari_language(text)
            if script in _SCRIPT_LANG:
                return _SCRIPT_LANG[script]
            if script == "beng":
                return "bn"
            if script != "latn":
                return script            # another script entirely (Cyrillic, Arabic …): not in scope
    return declared or "en"


def in_scope(language: str) -> bool:
    return language in LANGUAGES
