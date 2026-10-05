# Database migrations and rollout runbook

Every file in `docs/*.sql` is **idempotent** on a database built from this repository.
`tests/test_clustering_sql.py` proves it by applying the whole chain twice to an empty
Postgres in CI. **Production is different**: its schema predates some of these files, so
see the rollout section below for exactly which files to run there. The canonical order also lives in code:
`tools/db_migrations.py → MIGRATION_ORDER`.

## Order

| # | file | what it does |
|---|---|---|
| 1 | `migration.sql` | Layer 1: `feeds`, `articles`, `pipeline_runs` |
| 2 | `ingestion_fixes_migration.sql` | Layer 1 fixes: `last_new_article_at`, `disabled_reason`, cadence default, `pipeline_runs.tier`, `wizer_prune_articles()` |
| 3 | `enrichment_migration.sql` | Layer 2: enrichment columns, `article_entities`, `article_clusters` (base table) |
| 4 | `enrichment_outputs_migration.sql` | `sentiment_stats`, `ai_summary`, `ai_tag`, `ai_region`, `ai_org` (the code wrote these, but no migration existed) |
| 5 | `enrichment_queue_migration.sql` | work queue: `wizer_claim_enrichment_batch()`, release, `enrichment_runs`, `enrichment_queue_health` |
| 6 | `embedding_migration.sql` | v1 clustering history (`canonical_embedding`) |
| 7 | `pgvector_migration.sql` | v1 clustering history (`embedding_vec`, `find_nearest_cluster`) |
| 8 | `tier2_clustering_migration.sql` | v1 clustering history (`outlet_set`, `find_duplicate_clusters`) |
| 9 | `cluster_index_migration.sql` | v1 clustering history (index) |
| 10 | `propensity_migration.sql` | `propensity_score` column |
| 11 | `clustering_v2_migration.sql` | **story clustering v2** (functions incl. the batched job feed, columns, indexes, views) |
| 12 | `archive_migration.sql` | article archive: `article_archive_log`, export pages, verified-only `wizer_prune_archived_day()` |
| 13 | `enrichment_queue_v2_migration.sql` | queue v2: oldest-first by ingestion time, no age gate, crawl failures claimed, daily dead-letter retries, new `enrichment_queue_health` |
| 14 | `bulk_io_migration.sql` | bulk I/O: cluster state fetch, `wizer_apply_cluster_changes`, `wizer_fetch_unclustered_ingested`, `wizer_save_enrichment_batch`, `wizer_record_feed_polls` |

Files 6–9 are kept so the history replays cleanly on a fresh database. Their functions are
not used any more. Two of them previously **could not be re-run**: each redefined
`find_nearest_cluster` with a different return type. Both now drop the function first.

## Rolling this release out to the live Supabase project

Run it in this order. Each step is safe to repeat.

**1. Check pgvector is ≥ 0.7** (clustering v2 needs `halfvec` / `l2_normalize`):
```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';   -- expect 0.8.x
-- if older:  ALTER EXTENSION vector UPDATE;
```
The v2 migration refuses to run on an older version and says why.

**2. Apply the five new files**, in this order: `ingestion_fixes_migration.sql`,
`enrichment_outputs_migration.sql`, `enrichment_queue_migration.sql`,
`clustering_v2_migration.sql`, `archive_migration.sql`. This exact sequence was verified twice (fresh and repeated)
against a clone of the production schema, made with `pg_dump --schema-only` on 2026-10-03.

**Do not re-run `migration.sql` on production.** Production's views have drifted from it
(`CREATE OR REPLACE VIEW` fails with *"cannot change name of view column"*), and its
`CREATE TABLE`s describe `uuid` ids where production uses `bigint`. The other history files
(3, 6–10) are already applied and need nothing.

Run the files with `psql` or a script rather than the SQL Editor. `ingestion_fixes`
back-fills `feeds.last_new_article_at` from the 2.9M-row `articles` table, which can exceed
the editor's statement timeout.

**3. Deploy the code** (merge to `main`) and **re-enable the workflows**: GitHub disabled
them for inactivity on 2026-06-07, and a push does not re-enable them. Run
`gh workflow enable <id>` for each, or trigger `keepalive.yml` once by hand. From then on:
- the `ingest_*` workflows install only `requirements-ingest.txt`
- `cluster.yml` clusters every new article (every 30 min and after ingestion)
- the `enrich` workflow claims NLP work from the queue
- `cluster_maintenance.yml` runs every 3 h, and `keepalive.yml` weekly

**4. Backfill clusters** for articles enriched since clustering was removed. Run it once,
from any machine that has the service key:
```bash
pip install -r requirements.txt
python cluster.py backfill --hours 72 --dry-run   # count
python cluster.py backfill --hours 72
python cluster.py maintain
python cluster.py report
```

**5. Label legacy dormant feeds (optional).** Feeds the old code disabled for dormancy
can't be told apart from feeds paused by hand, so they are not re-checked automatically.
To opt them in to the weekly dormant re-check:
```sql
UPDATE feeds SET disabled_reason = 'dormant'
WHERE  is_active = false AND disabled_reason IS NULL AND fail_count < 5;   -- review first
```

**6. First archive run, done cautiously.** Run it once with `--no-prune` (or trigger
`archive.yml` with `no_prune=true`). Check `python archive.py status` and spot-check a fetched
day in DuckDB. Then let the daily job prune. When the backlog has been pruned (about 2.3M
rows older than 30 days), reclaim the disk space during a pause in ingestion:
```sql
VACUUM (FULL, ANALYZE) articles;          -- rewrites the table; locks it while running
VACUUM (FULL, ANALYZE) article_entities;
```
(or `pg_repack`, which locks only briefly). Afterwards, reduce the provisioned disk in the
Supabase dashboard if it doesn't shrink on its own.

**7. Watch the first day:**
```sql
SELECT * FROM enrichment_queue_health;   -- pending should drain; dead_letter ~0
SELECT * FROM cluster_health;            -- clustered_pct ~100, seed_pct 60-80 %
SELECT * FROM top_stories_24h LIMIT 20;
SELECT * FROM enrichment_runs ORDER BY started_at DESC LIMIT 10;
```

## Runners do the work, the DB is a sink (2026-10-05)

Order:

1. Apply `enrichment_queue_v2_migration.sql`, then `bulk_io_migration.sql`.
2. Spread out the first poll after the 2026-10-05 reset. Otherwise every active feed is due
   at once, and that spike alone saturates Micro:
   ```sql
   UPDATE feeds
      SET last_polled_at = now() - random() * coalesce(poll_interval_mins, 720) * interval '1 minute'
    WHERE is_active AND last_polled_at IS NULL;
   ```
3. Push the code, then enable the workflows: `ingest-*`, `process.yml`, `enrichment.yml`
   (sweeper), `cluster.yml` and `cluster_maintenance.yml`.
4. Watch the first runs:
   - the ingest log shows `Hand-off: wrote N articles`
   - the `process.yml` cluster job does a cold full load once, then logs `applied N changed clusters`
   - each enrich shard logs `saved`
   - `SELECT * FROM enrichment_queue_health;` shows `unenriched_over_24h` = 0

## Enrichment queue v2 (2026-10-05)

Apply `enrichment_queue_v2_migration.sql`, then deploy the code — in that order: the new code
calls the 6-argument `wizer_claim_enrichment_batch`, which the migration creates (and it drops
the 4-argument v1). Between the two steps the old code's claims fail loudly and nothing is
lost; the queue simply waits.

```sql
SELECT * FROM enrichment_queue_health;   -- unenriched_over_24h must stay 0; given_up ~0
```

## Rollback switches (no redeploy needed)

| problem | switch |
|---|---|
| clustering misbehaves | repository variable `CLUSTERING_ENABLED=false` (Settings → Variables; read by enrichment.yml) (enrichment continues, articles are saved without a cluster; backfill later) |
| clusters too coarse or too fine | `CLUSTER_JOIN_THRESHOLD`, `CLUSTER_MERGE_THRESHOLD` (re-calibrate first, see `docs/CLUSTERING.md` §6) |
| maintenance merges look wrong | `workflow_dispatch` with `dry_run=true`, or disable the workflow |
| URL hash transition | `LEGACY_URL_HASH_CHECK=1` (default) keeps old hashes deduplicating; turn it off about 90 days after deploy |
| table size | pruning is **off** by default (`ARTICLE_HARD_LIMIT=0`). Setting a limit deletes the oldest rows on the next ingestion run; production held 2.98M rows on 2026-10-03 |

The v1 `article_clusters` rows are marked `status = 'legacy'`. They are kept, never searched,
and pruned by maintenance once no article points at them.
