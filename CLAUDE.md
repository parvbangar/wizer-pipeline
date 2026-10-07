# WIZER — News Intelligence Pipeline: guide for Claude

Read this before changing anything. Deeper material lives in `README.md` (operations),
`docs/CLUSTERING.md` (clustering algorithm and calibration) and `docs/MIGRATIONS.md` (schema
order and rollout).

## What it is

An Indian-news pipeline running on GitHub Actions with Supabase (Postgres + pgvector):

1. **Layer 1, ingestion** (`main.py`, `pipeline/`): polls RSS/Atom feeds on 6 cadences,
   crawls full text, deduplicates, and stores rows in `articles`.
2. **Layer 2, enrichment** (`enrich.py`, `enrichment/`): claims articles from a work queue,
   runs 9 NLP steps, and puts every article into a **story cluster**.

**Language scope: English and Hindi only** (user decision, 2026-10-07).
- Ingest keeps an article only if `pipeline/language_scope.py` says en or hi. The check runs per
  article: Devanagari is split into Hindi vs Marathi, other scripts are dropped, and Latin script
  needs an en/hi declaration. `WIZER_LANGUAGES` overrides the scope.
- Feeds in other languages were deactivated with `disabled_reason = 'language_scope_en_hi'`.
- Stored articles in other languages were removed with `tools/purge_languages.py`; see
  `docs/MIGRATIONS.md`.
- The registration tools (`tools/discover_sitemaps.py`, `tools/register_official_sources.py`)
  register en/hi sources only.

## Repository layout

```
main.py  push_feeds.py                 Layer 1 entry points
enrich.py  cluster.py                  Layer 2 entry points (cluster.py: backfill / maintain / report)
pipeline/   config poller crawler db dedup circuit_breaker
enrichment/ config db runner clustering cluster_maintenance
            steps/ text_stats language sentiment ner keywords classifier summarizer images embedding
docs/       *.sql migrations (order: tools/db_migrations.py), CLUSTERING.md, MIGRATIONS.md
tools/      db_migrations.py  pg_backend.py  e2e_local.py  cluster_eval/  classifier_eval/
tests/      unit tests (offline) + test_clustering_sql.py (needs WIZER_TEST_DSN)
requirements-ingest.txt (Layer 1 only) ⊂ requirements.txt (all) ; requirements-dev.txt (tests)
```

Things described in older notes that **do not exist**: an `Ingestion/` folder, `recrawl.py`,
`propensity.py`, `train_propensity.py`, `cleanup_clusters.py`, `debug_classifier.py`,
`Feed_Validator/`. Propensity scoring was removed in `374dfe0` and has not been rebuilt (out
of scope so far); the `propensity_score` column is unused.

## Architecture: runners do the work, the database is a sink (2026-10-05)

Production runs on Supabase **Micro** (burstable CPU and disk I/O, no budget to upgrade).
Using Postgres as the pipeline's workspace exhausted its credits, so:

- **Ingest** (`ingest-*.yml`) only DISCOVERS. Each article is stored from feed metadata
  (`crawler.discover_article`, no HTTP), WITHOUT `full_text`.
  - Why: inline crawling let a 58-min run reach only 456 of 1,365 due feeds.
  - The inserted rows go to a hand-off file (`pipeline/handoff.py`, `WIZER_HANDOFF_PATH`),
    appended per feed so a killed run keeps them, and uploaded as the artifact `handoff`.
  - Feed poll outcomes are written every 200 feeds (`db.FeedPollBatch` →
    `wizer_record_feed_polls`).
  - `INGEST_TIME_BUDGET_MINUTES` stops new feeds before the job timeout; feeds not
    reached stay due.
- **`process.yml`** runs after every ingest:
  - `cluster`: in-memory clustering (`cluster.py run --memory`, `enrichment/cluster_job.py`,
    `enrichment/memory_clustering.py`). Its work list is every unclustered article in the
    DB; its state is cached between runs (Actions cache, delta-synced).
  - `enrich`: one shard per ~1,500 hand-off articles, up to 16 (`enrich.py --handoff --shard i
    --shards n`, `enrichment/handoff_runner.py`). Each article is crawled
    (`crawler.crawl_record`, 32 in flight), then enriched. Results plus crawl columns are
    saved 100 at a time (`wizer_save_enrichment_batch`).
  - Bodies go to Supabase Storage, bucket `article-bodies` (`enrichment/body_store.py`),
    never to Postgres. That bucket is the permanent full-text dataset.
- **`enrichment.yml` is the sweeper.** Every 2 h, 4 runners claim articles ingested more than 6 h
  ago that are still unenriched, and crawl + enrich them the same way (`enrich.py --sweep`).
- **`cluster.yml`** is a 2-hourly fallback of the same in-memory job.

All bulk RPCs live in `docs/bulk_io_migration.sql`.

## Data flow

**Layer 1** (`pipeline/poller.py`): `get_due_feeds(cadence)`, paginated past PostgREST's
1000-row cap. A feed is due once `FEED_DUE_TOLERANCE` (0.9) of its interval has passed.
For each feed:

- **Circuit breaker.** Error-disabled feeds need a manual reset. Dormant feeds get a weekly
  re-check (`feeds.disabled_reason`).
- **Bounded fetch** with a timeout, a size cap and a deadline, on an explicitly sized thread
  pool.
- **Per entry:**
  - URL normalised (http/https collapsed) and hashed with mmh3 → in-memory set → DB
    `url_hash`; during the transition the legacy hash is checked too
  - title SimHash, Hamming distance ≤ 3: stored, flagged `is_duplicate`
  - `crawl_article` (4 fetch strategies, 5 extractors, per-article deadline, domain
    fail-fast)
- **Store** with `upsert_articles` (`ON CONFLICT DO NOTHING`).
- **Dormancy check** only when the poll inserted nothing.
- **At the end of the run,** `wizer_prune_articles()` caps the table size.

**Layer 2** (`enrichment/runner.py`):

- `db.claim_batch()` → `wizer_claim_enrichment_batch` (`FOR UPDATE SKIP LOCKED`, 150-min
  lease). Queue v2: oldest-first by `coalesce(crawled_at, created_at)`, no age gate, failed
  crawls included; after 3 attempts retried once a day up to 7 times. `enriched_at` is never
  set for a failed article.
- The batch is sorted **oldest → newest** (better online clustering) and embedded once
  (`steps/embedding.py`, multilingual-E5-base, no prefix).
- **Per article** (`enrich_one`):
  - text_stats
  - [< 50 words (briefs, failed crawls): everything below runs on headline + description, except keywords]
  - language
  - [en/hi only: sentiment, NER, keywords]
  - category (`_category`: linear head on the E5 vector, `steps/category_head.py`; mDeBERTa
    zero-shot only as a fallback), tags (mDeBERTa), summary, image pHash
- **Then** `save_entities` → `clustering.assign_cluster` (RPC `wizer_assign_cluster`,
  which also stamps `articles.cluster_id`) → `save_article_enrichment`, which sets
  `enriched_at` **last**.
- **Crash** in `enrich_one`: retried via lease expiry; on the final regular attempt
  `enrich_error` records why and the article becomes a daily-retried dead letter.
- **Time budget or signal:** unprocessed claims are released.
- **Every run** writes an `enrichment_runs` row.

**Clustering job** (`cluster.py run`, `cluster.yml`, every 30 min): clusters EVERY
article that has no cluster yet, independent of NLP enrichment. Production ingests
~50K articles a day and NLP managed ~2.6K, so clustering cannot wait for NLP. It pages
through `wizer_fetch_unclustered`, embeds each page, and assigns 50 articles per round
trip with `wizer_assign_cluster_batch`. `enrich.py` skips embedding for articles that
already have a cluster.

**Clustering** (details in `docs/CLUSTERING.md`):

- **Decision** inside `wizer_assign_cluster`, under an advisory lock: candidates are the
  nearest centroids among active, same-model, time-compatible clusters (gap 18 h,
  span 120 h); the score is **average-link** (`q·Σm/n`, exact from `centroid_sum`).
- **Thresholds:** join ≥ 0.85; anchor (seed) similarity ≥ 0.75; gray zone implemented but
  off by default.
- **Maintenance** (`cluster.py maintain`, every 3 h): merge twins at average-link ≥ 0.845,
  reconcile counts, prune orphans.

## Invariants — do not break

- **Cluster state has ONE writer at a time: the concurrency group `story-clustering`**
  (`process.yml` cluster job, `cluster.yml`, `cluster_maintenance.yml`).
  - The in-memory clusterer owns centroid sums, counts, outlets and time spans. It writes
    them with `wizer_apply_cluster_changes`.
  - Maintenance merges in SQL, inside the same group.
  - Never write those columns from anywhere else; the v1 bug was read-modify-write races.
    In particular the sweeper runs with `CLUSTERING_ENABLED=false`.
  - Enrichment may add `top_entities` / `entity_set`, and must not bump `updated_at`.
- **`memory_clustering.py` must match `wizer_assign_cluster`.**
  `tests/test_clustering_sql.py::TestMemoryParity` checks decisions, partitions and the
  bulk-written rows against the SQL function. Change both together.
- **The delta-sync bookmark is the DB clock** returned by `wizer_apply_cluster_changes`.
  Rows it writes carry `updated_at = that time`, so they are not re-downloaded.
- **`wizer_assign_cluster` must stay idempotent.** An already-clustered article returns
  `existing`. That is what makes retries and `--force` safe.
- **`enriched_at` is the last write** for an article.
- **Python RPC parameter names must equal the SQL signature.**
  `tests/test_clustering.py::test_params_match_sql_signature_exactly` enforces this.
- **Migrations are idempotent and replay on an empty database.** Add new SQL as a new file
  plus an entry in `tools/db_migrations.MIGRATION_ORDER`, never by editing history in a
  non-idempotent way. `CREATE OR REPLACE FUNCTION` cannot change a return type: drop first.
- **New RPCs:** `REVOKE … FROM PUBLIC, anon, authenticated`; `GRANT … TO service_role`.
- **PostgREST returns at most 1000 rows per request.** Paginate (`.range()`) or page the
  RPC.
- **Embedding vectors:** never send a zero or NaN vector (SQL rejects them; NaN sorts above
  every number in Postgres). The dimension is fixed at 768 by the schema.
- **Thresholds are calibrated**, not chosen. Before changing a model or threshold, re-run
  `tools/cluster_eval/` (clustering) or `tools/classifier_eval/` (category) and update the
  docs with the numbers.

## Production facts (inspected read-only on 2026-10-03)

- Supabase project in ap-southeast-1: Postgres 17.6, pgvector 0.8.0.
  Connect through the session pooler (`aws-1-ap-southeast-1.pooler.supabase.com`, user
  `postgres.<project-ref>`); the direct
  `db.` host is IPv6-only.
- The live schema differs from `docs/migration.sql`:
  - `feeds.id` and `articles.feed_id` are `bigint`
  - `articles.url_hash` is `char(32)` holding a space-padded integer
  - `pipeline_runs` has `tier` but no `cadence`
  - there are extra objects (`stories` table, `article_stats` / `feed_health` /
    `dormancy_watch` views)
- API (PostgREST) calls run with `statement_timeout = 30s` and `lock_timeout = 8s`, so every
  RPC must finish well within that.
- All scheduled workflows were `disabled_inactivity` from 2026-06-07 (60 days after the last
  commit). `keepalive.yml` now prevents this.
- Before it stopped: about 50K articles a day were ingested, about 2.6K a day enriched, and
  every enrichment run hit the 120-minute timeout. In production, 75 % of articles were
  categorised `general` (the classifier bug fixed here).

## Known gaps / not done

- **Propensity / virality scoring** is not implemented (the column exists).
- **Tag classification** (`classify_tags`) still uses the long-phrase labels. It has not been
  measured, because there is no labelled tag set. The same lexical-overlap risk the category
  labels had probably applies.
- **Accuracy numbers** for every step live in `docs/ACCURACY.md` (language ID: 95.5 % on 3,030
  publisher-labelled headlines, all 13 languages).
- **NER, sentiment and keywords** run only for en/hi (`ENRICH_SUPPORTED_LANGUAGES`), which is
  now the whole corpus.
- **Category head:** 76.7 % on the 13-language gold set vs 51.9 % for mDeBERTa
  (`docs/ACCURACY.md`). Retrain with `tools/gold/train_head.py` whenever the embedding model
  changes; the head refuses vectors from any other model.
- **Clustering granularity** is the event: developments of one long-running story often form
  separate clusters (54 % of same-running-story pairs are joined).
- **`newspaper3k`'s internal re-fetch** is not bounded by the crawler's per-article deadline.
- **`LEGACY_URL_HASH_CHECK`** can be removed about 90 days after the URL-normalisation change
  is deployed.
- **`full_text` is not in Postgres** for articles ingested with a hand-off. It travels in
  the `handoff` artifact (7-day retention). The sweeper enriches without it.
- **Retention** is handled by the archive (`archive.py`, `archiver/`, `archive.yml`), not
  `ARTICLE_HARD_LIMIT`, which stays 0. Postgres keeps a 30-day hot window. Older days are
  exported to Parquet in the Storage bucket `article-archive`, verified, then deleted through
  `wizer_prune_archived_day()`. That function refuses unverified, hot-window or changed days.
  **Never delete articles any other way.**

## Running things

```bash
python main.py --cadence breaking_news [--dry-run --verbose]
python enrich.py [--batch-size N] [--dry-run] [--force] [--time-budget MIN]
python cluster.py backfill --hours 72 | maintain [--dry-run] | report
pytest tests/                                     # + WIZER_TEST_DSN=… for SQL tests
python tools/e2e_local.py --limit 300 --runners 2 # real models + local Postgres
```
