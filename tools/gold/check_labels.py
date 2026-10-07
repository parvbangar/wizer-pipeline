"""Validate labeller output files in tools/gold/labels/ (count, gids, category names)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CATS = {"politics", "business", "cricket", "sports", "entertainment", "technology", "health",
        "education", "crime", "environment", "world", "crypto", "general"}

ok = True
for f in sorted((HERE / "labels").glob("*.json")):
    labels = json.loads(f.read_text(encoding="utf-8"))
    batch = HERE / "batches" / f.name
    want = {r["gid"] for r in json.loads(batch.read_text(encoding="utf-8"))} if batch.exists() else None
    got = {r["gid"] for r in labels}
    bad = {r["category"] for r in labels} - CATS
    problems = []
    if want is not None and got != want:
        problems.append(f"gids differ (missing {len(want - got)}, extra {len(got - want)})")
    if bad:
        problems.append(f"unknown categories {bad}")
    print(f"{f.name:10s} {len(labels):4d}  {'OK' if not problems else '; '.join(problems)}")
    ok &= not problems
sys.exit(0 if ok else 1)
