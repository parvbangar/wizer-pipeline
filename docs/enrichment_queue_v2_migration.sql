-- =============================================================================
-- docs/enrichment_queue_v2_migration.sql
-- Enrichment queue v2: every ingested article is enriched, oldest first.
--
-- Run AFTER docs/archive_migration.sql. Idempotent.
--
-- WHAT WAS WRONG WITH v1 (docs/enrichment_queue_migration.sql):
--   1. Newest-first + a 48 h window on published_at. Whenever enrichment ran
--      behind ingestion, the oldest pending articles aged out of the window
--      and were never claimed again.
--   2. published_at IS NULL never satisfies "published_at > x", so undated
--      articles were never claimed — and were invisible to the queue health
--      view. Feeds that republish old items (stale published_at) were already
--      outside the window on arrival.
--   3. is_crawled was required. Articles whose crawl failed (dead site,
--      tripped domain, empty extraction) were never enriched, and nothing
--      re-crawls them.
--   4. After 3 failed attempts an article was parked with enriched_at set —
--      counted as "enriched" with no enrichment, never retried.
--
-- v2:
--   - Queue key is ingestion time, coalesce(crawled_at, created_at): always
--     present, never in the future, monotonic. Oldest first, so a backlog
--     drains in order instead of silently dropping its tail.
--   - No age gate by default (p_max_age_hours = 0). The gate, if used, also
--     applies to ingestion time.
--   - Crawl failures are claimed; the runner enriches them from title +
--     description.
--   - Dead letters: after p_max_attempts the article is retried once every
--     p_retry_hours, up to p_max_retries more times. enriched_at stays NULL
--     throughout, so a failed article is never counted as enriched.
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Queue index: unenriched articles in ingestion order
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS articles_enrich_queue_v2_idx
  ON articles ((coalesce(crawled_at, created_at)), id)
  WHERE enriched_at IS NULL AND title IS NOT NULL;

-- v1's index (published_at DESC, is_crawled only) has no reader left.
DROP INDEX IF EXISTS articles_enrich_queue_idx;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. wizer_claim_enrichment_batch — atomically lease a batch of articles
--
--   p_limit          rows to claim (the Python caller pages under PostgREST's
--                    1000-row cap)
--   p_max_age_hours  only articles INGESTED within N hours (0 = no limit)
--   p_lease_minutes  a claim older than this is considered abandoned
--                    (must exceed the longest possible run: job timeout)
--   p_max_attempts   regular attempts before an article is a dead letter
--   p_retry_hours    a dead letter is retried once per this many hours …
--   p_max_retries    … at most this many extra times, then it is given up
--                    (enrichment_queue_health.given_up)
--   p_min_age_minutes only articles ingested at least this long ago: the
--                    SWEEPER passes this, so it leaves fresh articles to the
--                    hand-off runners that already hold their body
--                    (docs/bulk_io_migration.sql, enrichment/handoff_runner.py)
--
-- The signature changed (new parameters), so v1 is dropped first —
-- CREATE OR REPLACE cannot add parameters to an existing function.
-- ─────────────────────────────────────────────────────────────────────────────
DROP FUNCTION IF EXISTS wizer_claim_enrichment_batch(integer, integer, integer, integer);

CREATE OR REPLACE FUNCTION wizer_claim_enrichment_batch(
  p_limit          integer,
  p_max_age_hours  integer DEFAULT 0,
  p_lease_minutes  integer DEFAULT 150,
  p_max_attempts   integer DEFAULT 3,
  p_retry_hours    integer DEFAULT 24,
  p_max_retries    integer DEFAULT 7,
  p_min_age_minutes integer DEFAULT 0
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
              AND q.title IS NOT NULL
              AND (p_max_age_hours <= 0
                   OR coalesce(q.crawled_at, q.created_at) > now() - make_interval(hours => p_max_age_hours))
              AND coalesce(q.crawled_at, q.created_at) <= now() - make_interval(mins => greatest(p_min_age_minutes, 0))
              AND (q.enrich_claimed_at IS NULL
                   OR q.enrich_claimed_at < now() - make_interval(mins => p_lease_minutes))
              AND (q.enrich_attempts < p_max_attempts
                   OR (q.enrich_attempts < p_max_attempts + p_max_retries
                       AND q.enrich_claimed_at < now() - make_interval(hours => p_retry_hours)))
            ORDER BY coalesce(q.crawled_at, q.created_at), q.id
            LIMIT greatest(p_limit, 0)
              FOR UPDATE SKIP LOCKED
         )
  RETURNING a.*;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. wizer_enrichment_queue_depth — articles claimable right now
--    (ignores leases: in-flight rows still count as work to do)
-- ─────────────────────────────────────────────────────────────────────────────
DROP FUNCTION IF EXISTS wizer_enrichment_queue_depth(integer, integer);

CREATE OR REPLACE FUNCTION wizer_enrichment_queue_depth(
  p_max_age_hours integer DEFAULT 0,
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
     AND title IS NOT NULL
     AND enrich_attempts < p_max_attempts
     AND (p_max_age_hours <= 0
          OR coalesce(crawled_at, created_at) > now() - make_interval(hours => p_max_age_hours));
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. enrichment_queue_health — one-row dashboard
--
--   pending                 waiting for a regular attempt (not leased)
--   in_flight               currently leased by a runner
--   dead_letter             failed 3 attempts; retried once a day
--   given_up                failed every retry — needs a look (enrich_error)
--   uncrawled_pending       pending articles with no full text (crawl failed);
--                           enriched from title + description
--   unenriched_over_24h     ingested > 24 h ago and still not enriched — the
--                           number that must stay at 0
--   oldest_pending_minutes  age of the queue head (ingestion time)
--
-- The literals mirror the defaults of the functions above (3 attempts, 7
-- retries, 150-min lease).
-- ─────────────────────────────────────────────────────────────────────────────
DROP VIEW IF EXISTS enrichment_queue_health;

CREATE VIEW enrichment_queue_health AS
SELECT
  count(*) FILTER (WHERE enrich_attempts < 3
                     AND (enrich_claimed_at IS NULL
                          OR enrich_claimed_at < now() - interval '150 minutes'))     AS pending,
  count(*) FILTER (WHERE enrich_claimed_at >= now() - interval '150 minutes')         AS in_flight,
  count(*) FILTER (WHERE enrich_attempts >= 3 AND enrich_attempts < 10)                AS dead_letter,
  count(*) FILTER (WHERE enrich_attempts >= 10)                                        AS given_up,
  count(*) FILTER (WHERE NOT coalesce(is_crawled, false) AND enrich_attempts < 3)      AS uncrawled_pending,
  count(*) FILTER (WHERE coalesce(crawled_at, created_at) < now() - interval '24 hours') AS unenriched_over_24h,
  round(extract(epoch FROM now() - min(coalesce(crawled_at, created_at))
          FILTER (WHERE enrich_attempts < 3)) / 60)::integer                           AS oldest_pending_minutes
FROM articles
WHERE enriched_at IS NULL
  AND title IS NOT NULL;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    EXECUTE 'REVOKE ALL ON TABLE enrichment_queue_health FROM anon, authenticated';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'GRANT SELECT ON TABLE enrichment_queue_health TO service_role';
  END IF;
END $$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. Permissions — only the pipeline's service role may call the queue RPCs
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
DECLARE
  fn text;
BEGIN
  FOREACH fn IN ARRAY ARRAY[
    'wizer_claim_enrichment_batch(integer, integer, integer, integer, integer, integer, integer)',
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
