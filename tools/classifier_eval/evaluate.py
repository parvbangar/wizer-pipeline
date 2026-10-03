#!/usr/bin/env python3
"""
tools/classifier_eval/evaluate.py
═════════════════════════════════
Measure the zero-shot category classifier against hand-labelled headlines.

category_gold.json holds 160 live Indian headlines (English + Hindi) with the
SET of acceptable categories for each — many stories legitimately fit two
(a G20 trade statement is business AND world). A prediction is correct when
it is in the set.

Compares label designs / scoring modes and prints accuracy, the share of
articles that fell back to "general", and the most common confusions.

USAGE:
  python tools/classifier_eval/evaluate.py                  # all designs
  python tools/classifier_eval/evaluate.py --design production
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from enrichment.steps import classifier as C  # noqa: E402

GOLD = Path(__file__).with_name("category_gold.json")

# Design used before 2026-10 (long "about X, Y or Z" phrases, softmax across
# labels). Kept here verbatim so the regression stays measurable.
LEGACY_LABELS = {
    "about cricket, the IPL, or a cricket match or tournament": "cricket",
    "about Indian politics, elections, parliament, or government policy": "politics",
    "about business, the economy, stock markets, or corporate finance": "business",
    "about Bollywood, movies, music, entertainment, or Indian celebrities": "entertainment",
    "about technology, software, artificial intelligence, or tech startups": "technology",
    "about sports other than cricket, such as football, hockey, kabaddi, or athletics": "sports",
    "about health, medicine, hospitals, disease, or public healthcare": "health",
    "about education, schools, colleges, university exams, or student affairs": "education",
    "about crime, police, courts, arrests, or legal investigations": "crime",
    "about the environment, climate change, floods, droughts, or pollution": "environment",
    "about international relations, foreign affairs, or events outside India": "world",
    "about cryptocurrency, bitcoin, blockchain, or digital currency": "crypto",
}


def predict(clf, text: str, labels: dict[str, str], template: str, multi: bool, threshold) -> str:
    r = clf(text, candidate_labels=list(labels), hypothesis_template=template, multi_label=multi,
            batch_size=C.CLASSIFY_BATCH_SIZE)
    if threshold is None:                               # production decision rule
        return C.pick_category(dict(zip(r["labels"], r["scores"])))
    best, score = r["labels"][0], r["scores"][0]
    return labels[best] if score >= threshold else "general"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default="all")
    args = ap.parse_args()

    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=C.DEBERTA_MODEL, device=-1)
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    texts = [C._classification_text(g["title"], g["description"], "") for g in gold]

    designs = {
        "legacy (long labels, softmax, t=0.20)":
            (LEGACY_LABELS, "This example is {}.", False, 0.20),
        "production":
            (C._LABEL_TO_CATEGORY, C.HYPOTHESIS_TEMPLATE, C.CLASSIFY_MULTI_LABEL, None),
    }
    if args.design != "all":
        designs = {k: v for k, v in designs.items() if k.startswith(args.design)}

    for name, (labels, template, multi, thr) in designs.items():
        preds = [predict(clf, t, labels, template, multi, thr) for t in texts]
        ok = [p in g["gold"] for p, g in zip(preds, gold)]
        confusions = Counter((g["gold"][0], p) for p, g, k in zip(preds, gold, ok) if not k)
        print(f"── {name}")
        print(f"   accuracy {sum(ok) / len(ok):.1%}   'general' fallback {preds.count('general') / len(preds):.1%}")
        print(f"   predicted: {dict(Counter(preds).most_common())}")
        print(f"   top confusions (gold → predicted): {confusions.most_common(6)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
