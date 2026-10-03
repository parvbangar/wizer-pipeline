-- =============================================================================
-- docs/enrichment_queue_migration.sql
-- Layer 2 work queue: atomic batch claiming, crash-safe leases, dead letters,
-- and a per-run log.
--
-- Run AFTER docs/enrichment_migration.sql. Idempotent.
--
-- THE PROBLEM THIS SOLVES:
--   enrich.yml runs two shards that each read "unenriched articles ORDER BY
--   published_at DESC" with OFFSET 0 and OFFSET 1000. That is only disjoint if
--   nothing else runs — but enrichment is triggered by workflow_run from SIX
--   ingestion workflows plus an hourly cron, with no concurrency group, so
--   several runs overlap routinely. Overlapping runs read the same rows and
--   enrich the same articles twice (wasted CPU), and — with story clustering —
--   would count the same article into a cluster twice.
--   Offsets also drift: as one runner commits enriched_at, the row set the
--   other runner is paging through shifts underneath it, so rows get skipped.
--
-- THE FIX — a claim queue (the standard Postgres job-queue pattern):
--   wizer_claim_enrichment_batch() picks the newest eligible articles with
--   FOR UPDATE SKIP LOCKED and stamps enrich_claimed_at in the same statement.
--   Concurrent callers never see each other's rows. A claim is a LEASE: if a
--   runner dies (OOM, cancelled job, runner lost) the rows become claimable
--   again once the lease expires — nothing is lost and nothing is stuck.
--
--   enrich_attempts counts claims. An article whose processing kills the
--   process every time (a poison pill) stops being claimed after
--   p_max_attempts and shows up in enrichment_queue_health.dead_letter
--   instead of crashing every future run.
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Queue bookkeeping columns on articles
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE articles ADD COLUMN IF NOT EXISTS enrich_claimed_at timestamptz;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS enrich_attempts   smallint NOT NULL DEFAULT 0;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS enrich_error      text;

-- The claim query's access path: unenriched, crawled, newest first.
-- (articles_enriched_at_idx from enrichment_migration.sql covers the same
--  predicate but not is_crawled; this one lets the planner stop after LIMIT.)
CREATE INDEX IF NOT EXISTS articles_enrich_queue_idx
  ON articles (published_at DESC NULLS LAST)
  WHERE enriched_at IS NULL AND is_crawled AND title IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. wizer_claim_enrichment_batch — atomically lease a batch of articles
--
--   p_limit          rows to claim (PostgREST caps responses at max_rows,
--                    1000 by default — the Python caller pages under that)
--   p_max_age_hours  only articles published within N hours (0 = no limit)
--   p_lease_minutes  a claim older than this is considered abandoned
--                    (must exceed the longest possible run: job timeout)
--   p_max_attempts   stop claiming an article after this many claims
--
-- Returns full article rows (SETOF articles) so new columns never need a
-- signature change. The caller orders the batch itself.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_claim_enrichment_batch(
  p_limit          integer,
  p_max_age_hours  integer DEFAULT 48,
  p_lease_minutes  integer DEFAULT 150,
  p_max_attempts   integer DEFAULT 3
)
RETURNS SETOF articles
LANGUAGE sql
VOLATILE
SET search_path = public, extensions
AS $$
  UPDATE articles AS a
     SET enrich_claimed_at = now(),
         enrich_attempts   = a.enrich_attempts + 1
   WHERE a.id IN (
           SELECT q.id
             FROM articles AS q
            WHERE q.enriched_at IS NULL
              AND q.is_crawled
              AND q.title IS NOT NULL
              AND (p_max_age_hours <= 0
                   OR q.published_at > now() - make_interval(hours => p_max_age_hours))
              AND (q.enrich_claimed_at IS NULL
                   OR q.enrich_claimed_at < now() - make_interval(mins => p_lease_minutes))
              AND q.enrich_attempts < p_max_attempts
            ORDER BY q.published_at DESC NULLS LAST
            LIMIT greatest(p_limit, 0)
              FOR UPDATE SKIP LOCKED
         )
  RETURNING a.*;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. wizer_release_enrichment_claims — hand unprocessed rows back
--
-- Called when a run stops early (time budget reached, SIGTERM from a cancelled
-- job, Ctrl+C). The attempt is refunded because the article was never tried.
-- Rows that were enriched in the meantime are left alone.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_release_enrichment_claims(p_ids bigint[])
RETURNS integer
LANGUAGE sql
VOLATILE
SET search_path = public, extensions
AS $$
  WITH released AS (
    UPDATE articles
       SET enrich_claimed_at = NULL,
           enrich_attempts   = greatest(enrich_attempts - 1, 0)
     WHERE id = ANY (p_ids)
       AND enriched_at IS NULL
       AND enrich_claimed_at IS NOT NULL      -- only rows that actually hold a claim
    RETURNING 1
  )
  SELECT count(*)::integer FROM released;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. enrichment_runs — one row per enrich.py run
--
-- Answers the questions the pipeline could not answer before: how deep is the
-- queue, is it growing, how fast are we consuming it, and how did clustering
-- behave in each run.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS enrichment_runs (
  id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  started_at          timestamptz NOT NULL DEFAULT now(),
  finished_at         timestamptz,
  trigger             text,                 -- GitHub event name / 'local'
  runner              text,                 -- e.g. 'shard-0'
  dry_run             boolean     NOT NULL DEFAULT false,
  queue_depth_start   integer,              -- eligible + unclaimed at start
  claimed             integer     NOT NULL DEFAULT 0,
  processed           integer     NOT NULL DEFAULT 0,
  failed              integer     NOT NULL DEFAULT 0,
  released            integer     NOT NULL DEFAULT 0,
  cluster_joined      integer     NOT NULL DEFAULT 0,
  cluster_gray_joined integer     NOT NULL DEFAULT 0,
  cluster_created     integer     NOT NULL DEFAULT 0,
  cluster_skipped     integer     NOT NULL DEFAULT 0,
  duration_s          float,
  stop_reason         text,                 -- 'drained' | 'time_budget' | 'signal' | 'error'
  error               text
);

CREATE INDEX IF NOT EXISTS enrichment_runs_started_idx
  ON enrichment_runs (started_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. wizer_enrichment_queue_depth — cheap count used at run start
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_enrichment_queue_depth(
  p_max_age_hours integer DEFAULT 48,
  p_max_attempts  integer DEFAULT 3
)
RETURNS integer
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT count(*)::integer
    FROM articles
   WHERE enriched_at IS NULL
     AND is_crawled
     AND title IS NOT NULL
     AND enrich_attempts < p_max_attempts
     AND (p_max_age_hours <= 0
          OR published_at > now() - make_interval(hours => p_max_age_hours));
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. enrichment_queue_health — one-row dashboard
--
--   pending          eligible and waiting (published in the last 48 h)
--   in_flight        currently leased by a runner
--   dead_letter      gave up after 3 attempts — inspect enrich_error / logs
--   expired_unenriched  crawled articles that aged out of the 48 h window
--                    before any runner reached them (= throughput shortfall)
--   oldest_pending_minutes  freshness lag of the queue head
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW enrichment_queue_health AS
SELECT
  count(*) FILTER (WHERE enrich_attempts < 3
                     AND published_at > now() - interval '48 hours'
                     AND (enrich_claimed_at IS NULL
                          OR enrich_claimed_at < now() - interval '150 minutes'))  AS pending,
  count(*) FILTER (WHERE enrich_claimed_at >= now() - interval '150 minutes')      AS in_flight,
  count(*) FILTER (WHERE enrich_attempts >= 3)                                       AS dead_letter,
  count(*) FILTER (WHERE published_at <= now() - interval '48 hours'
                     AND published_at >  now() - interval '7 days')                  AS expired_unenriched_7d,
  round(extract(epoch FROM now() - min(published_at) FILTER (
          WHERE published_at > now() - interval '48 hours')) / 60)::integer          AS oldest_pending_minutes
FROM articles
WHERE enriched_at IS NULL
  AND is_crawled
  AND title IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- 7. Permissions — only the pipeline's service role may call the queue RPCs.
--
-- Postgres grants EXECUTE on new functions to PUBLIC by default, and PostgREST
-- exposes every function in `public` — without this, anyone holding the anon
-- key could claim or release the queue.
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
DECLARE
  fn text;
BEGIN
  FOREACH fn IN ARRAY ARRAY[
    'wizer_claim_enrichment_batch(integer, integer, integer, integer)',
    'wizer_release_enrichment_claims(bigint[])',
    'wizer_enrichment_queue_depth(integer, integer)'
  ] LOOP
    EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', fn);
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
      EXECUTE format('REVOKE ALL ON FUNCTION %s FROM anon, authenticated', fn);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
      EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', fn);
    END IF;
  END LOOP;
END $$;
