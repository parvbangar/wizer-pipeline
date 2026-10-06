#!/usr/bin/env python3
"""
tools/language_eval/evaluate.py
═══════════════════════════════
Language-detection accuracy on publisher-labelled headlines.

sitemap_titles.json: headlines from Indian news sitemaps (2026-10-06), each
with the language its publisher declares in <news:language> — at most 300
per language, at most 40 per sitemap. Headlines are the hard case (short).

Publisher labels are not perfect (e.g. a Malayalam outlet's English edition
listed as 'ml'); the report therefore also gives accuracy on the subset whose
script agrees with the label ("script-consistent"), which removes that noise
for the script-unique languages.

  python tools/language_eval/evaluate.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from enrichment.steps.language import _SCRIPT_LANG, detect_language, script_counts   # noqa: E402

HERE = Path(__file__).parent
SCRIPT_OF = {v: k for k, v in _SCRIPT_LANG.items()} | {"hi": "deva", "mr": "deva", "bn": "beng", "as": "beng",
                                                       "en": "latn", "fr": "latn", "ru": "other"}


def main() -> int:
    for stream in (sys.stdout,):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    data = [d for d in json.loads((HERE / "sitemap_titles.json").read_text(encoding="utf-8")) if d["lang"] != "ne"]
    ok, tot, cons_ok, cons_tot = (collections.Counter() for _ in range(4))
    confusion = collections.defaultdict(collections.Counter)
    for d in data:
        pred = detect_language("", "", d["title"])
        pred_base = (pred or "").split("-")[0]
        tot[d["lang"]] += 1
        ok[d["lang"]] += pred_base == d["lang"]
        confusion[d["lang"]][pred] += 1
        counts = script_counts(d["title"])
        dominant = max(counts, key=counts.get) if counts else None
        if dominant == SCRIPT_OF.get(d["lang"]):
            cons_tot[d["lang"]] += 1
            cons_ok[d["lang"]] += pred_base == d["lang"]
    print(f"{'lang':5s} {'acc':>7s} {'n':>5s}   {'script-consistent':>18s}   top predictions")
    for lang in sorted(tot, key=lambda l: -tot[l]):
        c = f"{cons_ok[lang] / cons_tot[lang]:.1%} of {cons_tot[lang]}" if cons_tot[lang] else "-"
        print(f"{lang:5s} {ok[lang] / tot[lang]:7.1%} {tot[lang]:5d}   {c:>18s}   {confusion[lang].most_common(3)}")
    n_ok, n = sum(ok.values()), sum(tot.values())
    c_ok, c_n = sum(cons_ok.values()), sum(cons_tot.values())
    print(f"\noverall {n_ok / n:.1%} of {n}   script-consistent {c_ok / c_n:.1%} of {c_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
