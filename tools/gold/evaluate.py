#!/usr/bin/env python3
"""
tools/gold/evaluate.py
══════════════════════
The 13-language category gold set: build it from two independent labelling
passes, measure their agreement, and score classifiers against it.

  python tools/gold/evaluate.py agreement        # pass A vs pass B (+ writes the adjudication batch)
  python tools/gold/evaluate.py build            # A==B, else adjudicated → category_gold_v2.json
  python tools/gold/evaluate.py score pred_current.json [pred_other.json …]

Protocol (docs/ACCURACY.md): 1,400 headlines/descriptions from production,
150 en + 150 hi + 100 per other Indian language, ≤6 per publisher. Two
labelling passes with the rubric in CATEGORY_RUBRIC.md, in different orders
(A: shuffled across languages, B: grouped by language); disagreements are
adjudicated by a third, independent pass that sees both labels.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
LABELS = HERE / "labels"
CATS = ["politics", "business", "cricket", "sports", "entertainment", "technology", "health",
        "education", "crime", "environment", "world", "crypto", "general"]


def _load(prefix: str) -> dict[int, dict]:
    out = {}
    for f in sorted(LABELS.glob(f"{prefix}*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            out[int(r["gid"])] = r
    return out


def _items() -> dict[int, dict]:
    return {int(r["gid"]): r for r in json.loads((HERE / "category_sample.json").read_text(encoding="utf-8"))}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_agreement() -> int:
    items, A, B = _items(), _load("A"), _load("B")
    both = sorted(set(A) & set(B))
    print(f"labelled: A {len(A)}, B {len(B)}, both {len(both)} of {len(items)}")
    by_lang = collections.defaultdict(list)
    for g in both:
        by_lang[items[g]["lang"]].append((A[g]["category"], B[g]["category"]))
    print(f"{'lang':5s} {'n':>4s} {'agree':>7s} {'kappa':>6s}")
    for lang in sorted(by_lang, key=lambda l: -len(by_lang[l])):
        pairs = by_lang[lang]
        print(f"{lang:5s} {len(pairs):4d} {sum(a == b for a, b in pairs) / len(pairs):7.1%} "
              f"{cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]):6.3f}")
    allp = [p for v in by_lang.values() for p in v]
    print(f"all   {len(allp):4d} {sum(a == b for a, b in allp) / len(allp):7.1%} "
          f"{cohen_kappa([a for a, _ in allp], [b for _, b in allp]):6.3f}")
    disagree = [g for g in both if A[g]["category"] != B[g]["category"]]
    adj = [{**{k: items[g][k] for k in ("gid", "lang", "title", "description", "domain")},
            "label_1": A[g]["category"], "label_2": B[g]["category"]} for g in disagree]
    (HERE / "batches" / "ADJ.json").write_text(json.dumps(adj, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(adj)} disagreements → batches/ADJ.json for adjudication")
    return 0


def cmd_build() -> int:
    items, A, B, J = _items(), _load("A"), _load("B"), _load("ADJ")
    gold, missing = [], []
    for g, it in sorted(items.items()):
        if g in A and g in B and A[g]["category"] == B[g]["category"]:
            label, how = A[g]["category"], "agreed"
        elif g in J:
            label, how = J[g]["category"], "adjudicated"
        else:
            missing.append(g)
            continue
        cant = bool(A.get(g, {}).get("cant_tell")) and bool(B.get(g, {}).get("cant_tell"))
        gold.append({**{k: it[k] for k in ("gid", "lang", "title", "description", "domain", "url")},
                     "category": label, "how": how, "cant_tell": cant})
    (HERE / "category_gold_v2.json").write_text(json.dumps(gold, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"gold: {len(gold)} items ({sum(g['how'] == 'adjudicated' for g in gold)} adjudicated), "
          f"{len(missing)} without a decision")
    print("categories:", dict(collections.Counter(g["category"] for g in gold).most_common()))
    return 0


def cmd_score(pred_files: list[str]) -> int:
    gold = json.loads((HERE / "category_gold_v2.json").read_text(encoding="utf-8"))
    gold = [g for g in gold if not g["cant_tell"]]
    for pf in pred_files:
        pred = json.loads((HERE / pf).read_text(encoding="utf-8"))
        by_lang = collections.defaultdict(lambda: [0, 0])
        confusion = collections.Counter()
        for g in gold:
            p = pred.get(str(g["gid"]))
            if p is None:
                continue
            by_lang[g["lang"]][0] += p == g["category"]
            by_lang[g["lang"]][1] += 1
            if p != g["category"]:
                confusion[(g["category"], p)] += 1
        ok, n = sum(v[0] for v in by_lang.values()), sum(v[1] for v in by_lang.values())
        macro = sum(v[0] / v[1] for v in by_lang.values()) / len(by_lang)
        print(f"\n== {pf}: accuracy {ok / n:.1%} of {n}   (macro over languages {macro:.1%})")
        print("   " + "  ".join(f"{l} {v[0] / v[1]:.0%}" for l, v in sorted(by_lang.items(), key=lambda x: -x[1][1])))
        print("   most common errors (gold → predicted):",
              ", ".join(f"{a}→{b} {c}" for (a, b), c in confusion.most_common(8)))
    return 0


def main() -> int:
    for s in (sys.stdout,):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]
    if cmd == "agreement":
        return cmd_agreement()
    if cmd == "build":
        return cmd_build()
    if cmd == "score":
        return cmd_score(sys.argv[2:])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
