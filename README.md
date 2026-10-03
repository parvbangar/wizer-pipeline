# Wizer Ingestion Pipeline

A two-layer pipeline for Indian news. **Layer 1** polls thousands of RSS/Atom feeds, crawls
the full article text, deduplicates it and stores it in Supabase. **Layer 2** enriches every
article: language, sentiment, entities, keywords, category, tags, summary and image hash.
It then groups articles from any outlet, in any language, into **story clusters**. All of it
runs on GitHub Actions.

```
Layer 1 — Ingestion (main.py)                Layer 2 — Enrichment (enrich.py)
──────────────────────────────               ─────────────────────────────────────────────
RSS/Atom feeds (6 cadences)                  work queue: claim newest unenriched articles
   ↓  bounded fetch (timeout + size cap)        ↓  (FOR UPDATE SKIP LOCKED — runners never overlap)
parse entries                                 batch-embed for clustering (multilingual-E5)
   ↓                                             ↓  per article, oldest → newest:
dedup: URL hash (memory + DB) · SimHash       text stats → language → sentiment → NER →
   ↓                                          keywords → category → tags → summary → image pHash
crawl HTML (4 fetch strategies,                  ↓
           5 text extractors)                 STORY CLUSTERING  (wizer_assign_cluster, one SQL txn)
   ↓                                             ↓
articles table ─────────────────────────────▶ enriched article + cluster_id
                                               cluster maintenance every 3 h: merge · reconcile · prune
```

**Database:** Supabase (PostgreSQL + pgvector) · **Runtime:** GitHub Actions · **Docs:**
[`docs/CLUSTERING.md`](docs/CLUSTERING.md) (algorithm and calibration),
[`docs/MIGRATIONS.md`](docs/MIGRATIONS.md) (schema order and rollout runbook).

---

## Quick start

```bash
pip install -r requirements.txt                  # everything (Layer 1 + ML stack)
python -m spacy download en_core_web_md && python -m spacy download xx_ent_wiki_sm
cp env.example .env                              # SUPABASE_URL, SUPABASE_SERVICE_KEY

python main.py --cadence breaking_news --dry-run --verbose    # Layer 1, no writes
python enrich.py --dry-run --batch-size 20 --verbose          # Layer 2, no writes
python cluster.py report                                      # clustering + queue health
```

Apply the SQL migrations first, in the order given in [`docs/MIGRATIONS.md`](docs/MIGRATIONS.md).

---

## Layer 1 — Ingestion

### Schedules

| Cadence | Workflow | Cron | Poll interval |
|---|---|---|---|
| `breaking_news` | `ingest-breaking-news.yml` | every hour | 60 min |
| `multiple_daily` | `ingest-multiple-daily.yml` | every 3 h | 3 h |
| `daily` + `unknown` | `ingest-daily.yml` | every 12 h | 12 h |
| `several_weekly` | `ingest-several-weekly.yml` | daily 01:00 | 24 h |
| `weekly` | `ingest-weekly.yml` | daily 02:00 | 24 h |
| `monthly` | `ingest-monthly.yml` | daily 03:00 | 24 h |

A feed is due once `FEED_DUE_TOLERANCE` (default 0.9) of its interval has passed. The cron
period equals the interval and `last_polled_at` is stamped when the feed *finishes*, so a
strict `>=` comparison used to skip every other run. Feeds without a cadence default to
`unknown` and are polled with `daily`.

The ingest workflows install only `requirements-ingest.txt`, not the ML stack.

### How a feed is processed

1. **Circuit breaker.** Skip feeds disabled for errors (`fail_count ≥ 5`, needs a manual
   reset). Feeds disabled as **dormant** (no new article in 30 days) get one re-check a week
   and reactivate on their own if they publish again (`feeds.disabled_reason`).
2. **Fetch.** Download the feed with a socket timeout, a 10 MB cap and a wall-clock deadline,
   so slow-drip servers can't hang a run. The thread pool is sized from the concurrency
   settings. A timeout counts as a failure.
3. **Deduplicate** each entry:
   - normalised URL → MurmurHash3, checked first against an in-memory set, then against the
     `UNIQUE` index on `articles.url_hash`
   - http and https versions of a URL hash identically
   - during the transition, the hash from the previous normalisation is also checked (see
     `LEGACY_URL_HASH_CHECK`)
   - title SimHash, Hamming distance ≤ 3: **near-duplicates are stored**, flagged
     `is_duplicate = true`
4. **Crawl.** HTTP strategies are tried in order: default UA → Googlebot → AMP → Wayback.
   Permanent 4xx responses are not retried. There is a per-article deadline, and a domain
   that keeps failing is skipped for the rest of the run.
5. **Extract** with RSS `content:encoded`, trafilatura, readability, newspaper3k and `<p>`
   fallback, keeping the best result of at least 200 characters. Metadata comes from Open
   Graph, Twitter cards and JSON-LD. Image priority: `og:image` → JSON-LD → RSS media →
   first `<img>`.
6. **Store.** Batch insert with `ON CONFLICT (url_hash) DO NOTHING`, falling back to row by
   row. NUL bytes are stripped.
7. **Prune (off by default).** If `ARTICLE_HARD_LIMIT` / `ARTICLE_PRUNE_TARGET` are set,
   the oldest articles above the limit are deleted by `wizer_prune_articles()`, in chunks
   along the primary key. Defaults are 0 (disabled): production's history (2.98M articles)
   is kept until an archive-then-prune step exists.

A run that can't reach the database exits 1. It is never reported as a successful empty run.

**Keepalive.** GitHub disables every scheduled workflow in a repository that has had no
commit for 60 days. That is what stopped this pipeline on 2026-06-07; nothing ran for four
months. `keepalive.yml` re-enables all workflows through the API every week, which resets
the timer.

---

## Layer 2 — Enrichment

### Triggers and concurrency

`enrichment.yml` runs about a minute after any ingestion workflow finishes (`workflow_run`), plus
hourly at :30 as a fallback. **Three runners** start each time and each **claims** its own
batch of up to 1,000 articles from the work queue:

- **No overlap.** `wizer_claim_enrichment_batch()` leases the newest eligible articles
  (crawled, unenriched, published in the last 48 h) with `FOR UPDATE SKIP LOCKED`.
  Concurrent runners, overlapping triggers and manual runs never process the same article.
- **No lost work.** A claim is a 150-minute lease. A runner that dies leaves leases that
  expire. A runner that stops early (100-minute time budget, cancellation, Ctrl+C) releases
  its unprocessed articles immediately.
- **No poison pills.** An article that crashes processing three times is parked as a dead
  letter (`enrichment_queue_health.dead_letter`) instead of crashing every run.
- **Observability.** Each run writes an `enrichment_runs` row: queue depth, claimed,
  processed, failed, released, cluster outcomes, duration and stop reason.

### Steps

| # | Step | Output | Notes |
|---|---|---|---|
| 1 | `text_stats` | `word_count`, `reading_time_mins` | articles under 50 words skip steps 2–9 |
| 2 | `language` | `language_detected` | lingua (deterministic); en/hi get the full NLP set below |
| 3 | `sentiment` | `sentiment`, `sentiment_score`, `sentiment_stats` | multilingual DistilBERT, title + description |
| 4 | `ner` | `article_entities`, `ai_region`, `ai_org` | spaCy `en_core_web_md` / `xx_ent_wiki_sm`, salience-scored |
| 5 | `keywords` | `keywords` | YAKE with language-specific stopwords, boilerplate filtered |
| 6 | `classifier` | `category` | mDeBERTa zero-shot, all languages (below) |
| 7 | `tags` | `ai_tag` | same model, multi-label topic tags |
| 8 | `summary` | `ai_summary` | extractive: description or first three sentences |
| 9 | `images` | `image_phash` | 64-bit perceptual hash, stored as signed `bigint` |
| 10 | **clustering** | `cluster_id`, `article_clusters` | **every** article, every language, short ones too |

Each step fails independently: if NER crashes, the article still gets everything else.
`enriched_at` is written **last**, after entities and the cluster assignment.

**Category classification** uses short single-concept labels ("crime", "law and courts",
"economy", …) with the template *"This news article is about {}."*. Each label is scored
independently, and an article goes to `general` below 0.30. Cricket wins over sports when
the cricket label itself scores ≥ 0.5. On 160 hand-labelled live headlines this scores
**68.1 % accuracy with 11 % `general`**. The previous long-phrase labels scored 33.8 %
with 55 % `general`, because their "…or events outside India" wording made "world" win
almost everything. Evaluate changes with `tools/classifier_eval/evaluate.py`.

### Story clustering

The full write-up is in [`docs/CLUSTERING.md`](docs/CLUSTERING.md). In short:

- **Model:** `intfloat/multilingual-e5-base`, 768 dimensions, covers all major Indian
  languages, cross-lingual ROC-AUC 0.97. It was chosen over LaBSE on 400 hand-labelled
  headline pairs.
- **Decision:** the exact **average-link** similarity between the article and every member
  of a nearby cluster must be ≥ 0.85, the article must be similar to the cluster's seed
  (≥ 0.75), and the cluster must be live in time (gap ≤ 18 h, span ≤ 120 h).
- **Atomic:** find, decide and write happen in one SQL transaction under an advisory lock.
  Concurrent runners can't create twin clusters or lose counts.
- **Maintenance every 3 h** (`cluster_maintenance.yml`): merge twins at average-link
  ≥ 0.845, reconcile counts, prune orphans.
- **Measured quality** on live headlines: 72 % of same-event pairs end up together, 3 % of
  hard different-story pairs do. The v1 implementation it replaces (LaBSE at 0.82) found
  9.5 % of same-event pairs.

```bash
python cluster.py run                      # cluster every new article (the scheduled job)
python cluster.py backfill --hours 72      # catch up on a longer window
python cluster.py maintain [--dry-run]     # merge / reconcile / prune
python cluster.py report --top 20          # health, top stories, queue, recent runs
```

```sql
SELECT * FROM top_stories_24h LIMIT 20;    -- biggest stories right now
SELECT * FROM cluster_health;              -- clustered %, seed %, merges, …
SELECT * FROM enrichment_queue_health;     -- pending, in flight, dead letters, lag
```

---

## Article archive

Postgres keeps a **30-day hot window**; older articles move to Parquet in the private
Supabase Storage bucket `article-archive` (`archive.yml`, daily):

```
articles/YYYY/MM/DD/part-NNNN.parquet   every column, ≤ 10K articles per part (~15 MB)
entities/YYYY/MM/DD/part-NNNN.parquet   their named entities
clusters/YYYY/MM/DD.parquet             the story clusters they belong to (no vectors)
```

Rows are deleted only after their Parquet copy has been re-downloaded and verified (SHA-256,
row counts, exact id set), and the SQL delete function refuses any day that is unverified,
inside the hot window, or changed since export. On production data, Parquet with zstd takes
**about 1.4 KB per article against about 4.4 KB in Postgres**: the 2.98M articles held on
2026-10-03 are about 4 GB as Parquet against 13.2 GB in the database.

```bash
python archive.py run --dry-run          # what would move
python archive.py run --no-prune         # export + verify, delete nothing
python archive.py status                 # per-day log
python archive.py fetch --start 2026-04-09 --end 2026-04-30 --out ./archive
python -c "import duckdb; print(duckdb.sql(\"SELECT domain, count(*) c FROM 'archive/articles/**/*.parquet' GROUP BY 1 ORDER BY c DESC LIMIT 10\"))"
```

Deleting rows frees space *inside* Postgres (new rows reuse it). To shrink the database
files themselves after the first big archive, run `VACUUM FULL articles` (locks the table) or
`pg_repack` during a pause in ingestion. See `docs/MIGRATIONS.md`.

---

## Configuration

Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` as GitHub Actions secrets, and locally in
`.env`. Every other setting has a default in `pipeline/config.py` or `enrichment/config.py`,
documented where it is defined. The settings most often changed:

| Variable | Default | |
|---|---|---|
| `MAX_CONCURRENT_FEEDS` / `MAX_CONCURRENT_ARTICLES` | 15 / 5 | Layer 1 concurrency (the workflows override these) |
| `FEED_DUE_TOLERANCE` | 0.9 | fraction of the poll interval after which a feed is due |
| `ENRICH_BATCH_SIZE` | 1000 | articles claimed per runner |
| `ENRICH_MAX_AGE_HOURS` | 48 | only enrich articles published this recently (0 = all) |
| `ENRICH_TIME_BUDGET_MINUTES` | 0 (CI: 100) | stop taking new work after this long, release the rest |
| `ENRICH_CLAIM_LEASE_MINUTES` | 150 | must exceed the enrich job timeout |
| `CLUSTERING_ENABLED` | true | kill switch (repository variable in CI) |
| `CLUSTER_JOIN_THRESHOLD` / `CLUSTER_MERGE_THRESHOLD` | 0.85 / 0.845 | re-calibrate before changing |
| `CLASSIFY_CONFIDENCE_THRESHOLD` | 0.30 | below this the category is `general` |

---

## Tests

```bash
pip install -r requirements-ingest.txt -r requirements-dev.txt
pytest tests/                                   # unit tests — offline, no ML stack needed

# + SQL integration tests against real Postgres/pgvector (CI runs these too):
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg17
WIZER_TEST_DSN="host=localhost port=5432 user=postgres password=postgres" pytest tests/
```

381 tests in total:

- Layer 1 regression tests: dedup, crawler, poller (against a local HTTP server that hangs,
  drips and oversizes), DB layer, circuit breaker.
- Layer 2 steps, runner semantics (claiming, ordering, crash/time-budget/signal handling)
  and clustering logic.
- 54 SQL integration tests: the assignment maths, 8-way concurrency, merges, the claim
  queue, grants, migration idempotency, and PostgREST argument decoding.

`tools/e2e_local.py` runs the real runner with the real models against a local Postgres.

## Repository layout

```
main.py                 Layer 1 CLI          enrich.py       Layer 2 CLI
push_feeds.py           CSV → feeds          cluster.py      clustering ops CLI
archive.py              archive CLI          archiver/       Parquet export, verify, prune
pipeline/               Layer 1 modules      enrichment/     Layer 2 modules
  config, poller, crawler, db, dedup,          config, db, runner, clustering,
  circuit_breaker                              cluster_maintenance, steps/*
docs/                   SQL migrations, CLUSTERING.md, MIGRATIONS.md
tools/                  db_migrations (order), pg_backend (direct-Postgres adapter),
                        e2e_local, cluster_eval/ (clustering calibration),
                        classifier_eval/ (category calibration)
tests/                  unit + SQL integration tests
.github/workflows/      ingest_* (6), enrich, cluster, cluster_maintenance, archive, keepalive, tests
```
