#!/usr/bin/env python3
"""
tools/gold/sample.py
════════════════════
Draw a language-stratified sample of production headlines for category labelling.

  python tools/gold/sample.py train_sample.json --en 700 --hi 700 --other 380 \
      --exclude category_sample.json --seed train1

Rules (the same as the gold set, docs/ACCURACY.md):
  - stratified by detected language (articles.language), quotas per language;
  - at most --per-domain items per publisher (default 6), so no outlet dominates;
  - deterministic order: md5(id || seed), not TABLESAMPLE (block sampling is biased
    towards whatever was ingested together);
  - --exclude: no URL and no title already in another sample (gold must stay unseen);
  - gids continue after the largest gid of the excluded files, so labels never collide.

Read-only connection: WIZER_READ_DSN (a libpq connection string to the prod pooler).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
LANGS = ["en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "or", "pa", "ur", "as"]

QUERY = """
select id::text, title, coalesce(description, ''), coalesce(domain, ''), url
  from articles
 where language = %(lang)s
   and title is not null and length(title) >= 15
 order by md5(id::text || %(seed)s)
 limit %(limit)s
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--en", type=int, default=700)
    ap.add_argument("--hi", type=int, default=700)
    ap.add_argument("--other", type=int, default=380)
    ap.add_argument("--per-domain", type=int, default=6)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", default="train1")
    args = ap.parse_args()

    import psycopg
    dsn = os.environ.get("WIZER_READ_DSN")
    if not dsn:
        print("WIZER_READ_DSN is not set", file=sys.stderr)
        return 2

    seen_urls, seen_titles, next_gid = set(), set(), 1
    for name in args.exclude:
        for it in json.loads((HERE / name).read_text(encoding="utf-8")):
            seen_urls.add(it["url"])
            seen_titles.add(it["title"].strip().lower())
            next_gid = max(next_gid, int(it["gid"]) + 1)

    out = []
    with psycopg.connect(dsn, options="-c default_transaction_read_only=on -c statement_timeout=120000") as conn:
        for lang in LANGS:
            quota = args.en if lang == "en" else args.hi if lang == "hi" else args.other
            rows = conn.execute(QUERY, {"lang": lang, "seed": args.seed, "limit": quota * 60}).fetchall()
            per_domain: dict[str, int] = {}
            picked = 0
            for aid, title, desc, domain, url in rows:
                key = title.strip().lower()
                if url in seen_urls or key in seen_titles or per_domain.get(domain, 0) >= args.per_domain:
                    continue
                per_domain[domain] = per_domain.get(domain, 0) + 1
                seen_urls.add(url)
                seen_titles.add(key)
                out.append({"id": aid, "title": title, "description": desc[:400], "domain": domain,
                            "url": url, "lang": lang, "gid": next_gid})
                next_gid += 1
                picked += 1
                if picked >= quota:
                    break
            print(f"{lang}: {picked}/{quota} from {len(per_domain)} publishers")

    (HERE / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(out)} items → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
