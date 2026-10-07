#!/usr/bin/env python3
"""
tools/purge_languages.py
════════════════════════
Delete stored articles outside the language scope (English + Hindi since 2026-10-07,
pipeline/language_scope.py), keeping story clusters consistent.

  python tools/purge_languages.py --dsn "$DSN" --dry-run
  python tools/purge_languages.py --dsn "$DSN" --backup D:/03_Data/wizer_backup_2026-10-07_languages

Which articles: the same rule as ingestion (language_scope.article_language on headline
+ description, with the stored feed/sitemap language as the declaration), except that a
body-based language_detected in another Indian script overrides it. Latin-script
detections (often wrong on short English headlines) never delete an article.

Steps:
  1. find candidates (rows whose stored language is not en/hi, whose detected language
     is something else, or whose headline uses an Indic / Arabic script) and decide each;
  2. back up the rows to delete and their article_entities (jsonl.gz) — before any write;
  3. ONE transaction, under the clustering advisory lock:
       - a cluster losing SOME members: centroid_sum scaled by kept/article_count, so the
         average (what average-link scores) is unchanged and sum/count stay consistent
         (member vectors are not stored, so exact subtraction is impossible);
         article_count, outlet_set, outlet_count, language_set recomputed from the kept
         members; a deleted canonical/representative article replaced by the earliest kept one;
       - a cluster losing ALL members: deleted;
       - the articles deleted (article_entities cascade).
  4. afterwards, evict the Actions cache `cluster-state-*` (done by the operator, see
     docs/MIGRATIONS.md), so the next clustering run loads the corrected state.

Run it only while no `story-clustering` job is running (single-writer invariant).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
import time
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.language_scope import LANGUAGES, article_language  # noqa: E402

# Detected languages that are certain from the script (enrichment/steps/language.py).
_SCRIPT_CERTAIN = {"mr", "bn", "as", "ta", "te", "kn", "ml", "gu", "or", "pa", "ur"}

CANDIDATES = r"""
select id, coalesce(title, ''), coalesce(description, ''),
       nullif(lower(trim(coalesce(language, ''))), ''), nullif(lower(trim(coalesce(language_code, ''))), ''),
       nullif(lower(trim(coalesce(language_detected, ''))), '')
  from articles
 where lower(trim(coalesce(language, ''))) not in ('en', 'hi')
    or (language_detected is not null and lower(trim(language_detected)) not in ('en', 'hi', 'hi-latn'))
    or title ~ '[\u0980-\u0DFF\u0600-\u06FF]'
    or (lower(trim(language)) = 'hi' and title ~ '[\u0900-\u097F]')
"""


def decide(title: str, desc: str, language: str | None, code: str | None, detected: str | None) -> str:
    lang = article_language(title, desc, language or code)
    if detected in _SCRIPT_CERTAIN:
        lang = detected
    return lang


def _json_default(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return str(v)


def backup(conn, ids: list[int], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for table, col, name in (("articles", "id", "articles.jsonl.gz"),
                             ("article_entities", "article_id", "article_entities.jsonl.gz")):
        n = 0
        with gzip.open(out / name, "wt", encoding="utf-8") as f:
            for i in range(0, len(ids), 2000):
                cur = conn.execute(f"select * from {table} where {col} = any(%s)", (ids[i:i + 2000],))
                cols = [d.name for d in cur.description]
                for row in cur:
                    f.write(json.dumps(dict(zip(cols, row)), ensure_ascii=False, default=_json_default) + "\n")
                    n += 1
        print(f"  backup {table}: {n} rows → {out / name}")


def purge(conn, ids: list[int]) -> dict:
    stats = {}
    with conn.transaction():
        conn.execute("select pg_advisory_xact_lock(hashtextextended('wizer:article_clusters', 0))")
        conn.execute("create temp table purge_ids (id bigint primary key) on commit drop")
        with conn.cursor().copy("copy purge_ids (id) from stdin") as cp:
            for i in ids:
                cp.write_row((i,))
        conn.execute("analyze purge_ids")
        conn.execute("""
            create temp table purge_clusters on commit drop as
            select c.id, c.article_count,
                   (select count(*) from articles a
                     where a.cluster_id = c.id and a.id not in (select id from purge_ids))::int as kept
              from article_clusters c
             where c.id in (select distinct a.cluster_id from articles a
                             join purge_ids p on p.id = a.id where a.cluster_id is not null)""")
        stats["clusters_touched"] = conn.execute("select count(*) from purge_clusters").fetchone()[0]
        stats["clusters_deleted"] = conn.execute(
            "delete from article_clusters c using purge_clusters p where c.id = p.id and p.kept = 0").rowcount
        stats["clusters_rescaled"] = conn.execute("""
            with kept as (
              select a.cluster_id as id,
                     jsonb_agg(distinct coalesce(nullif(btrim(lower(a.domain)), ''), '(unknown)')) as outlets,
                     jsonb_agg(distinct lower(trim(coalesce(a.language_detected, a.language)))) as langs,
                     (array_agg(a.id order by coalesce(a.published_at, a.created_at), a.id))[1] as first_id,
                     (array_agg(a.title order by coalesce(a.published_at, a.created_at), a.id))[1] as first_title
                from articles a
                join purge_clusters p on p.id = a.cluster_id and p.kept > 0
               where a.id not in (select id from purge_ids)
               group by a.cluster_id)
            update article_clusters c
               set centroid_sum = case when c.centroid_sum is null or p.article_count <= 0 then c.centroid_sum
                                       else c.centroid_sum * array_fill(
                                              (p.kept::real / greatest(p.article_count, p.kept)::real),
                                              array[vector_dims(c.centroid_sum)])::vector end,
                   article_count = p.kept,
                   outlet_set    = k.outlets,
                   outlet_count  = jsonb_array_length(k.outlets),
                   language_set  = k.langs,
                   canonical_article_id = case when c.canonical_article_id in (select id from purge_ids)
                                               then k.first_id else c.canonical_article_id end,
                   representative_article_id = case when c.representative_article_id in (select id from purge_ids)
                                                    then k.first_id else c.representative_article_id end,
                   headline = case when c.canonical_article_id in (select id from purge_ids)
                                   then k.first_title else c.headline end,
                   updated_at = now()
              from purge_clusters p, kept k
             where c.id = p.id and k.id = p.id""").rowcount
        stats["articles_deleted"] = conn.execute(
            "delete from articles a using purge_ids p where a.id = p.id").rowcount
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--backup", help="directory for the jsonl.gz backup (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.backup:
        ap.error("--backup is required for a real run")

    import psycopg
    conn = psycopg.connect(args.dsn, autocommit=True, options="-c statement_timeout=900000")
    t0 = time.perf_counter()
    rows = conn.execute(CANDIDATES).fetchall()
    drop, keep_langs, drop_langs = [], collections.Counter(), collections.Counter()
    for aid, title, desc, language, code, detected in rows:
        lang = decide(title, desc, language, code, detected)
        if lang in LANGUAGES:
            keep_langs[lang] += 1
        else:
            drop.append(aid)
            drop_langs[lang] += 1
    print(f"{len(rows)} candidates in {time.perf_counter() - t0:.0f} s: delete {len(drop)} "
          f"{dict(drop_langs.most_common())}; keep {dict(keep_langs)}")
    if args.dry_run or not drop:
        return 0
    backup(conn, drop, Path(args.backup))
    stats = purge(conn, drop)
    print("purged:", stats, f"in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
