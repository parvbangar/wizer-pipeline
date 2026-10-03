-- =============================================================================
-- docs/ingestion_fixes_migration.sql
-- Run AFTER docs/migration.sql, ONCE, in Supabase -> SQL Editor.
-- Every statement is idempotent (safe to re-run).
--
-- The Layer 1 code tolerates these columns being absent (it logs a warning and
-- degrades), but dormancy detection and the weekly dormant re-check only work
-- once this has been applied.
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 1: feeds.last_new_article_at
--
-- WHY?
--   circuit_breaker used feeds.last_success_at as "last new article", but
--   last_success_at advances on EVERY successful fetch, even with 0 new
--   articles, so dormancy could never trigger.  last_new_article_at is only
--   advanced when a poll actually inserts >0 articles.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE feeds ADD COLUMN IF NOT EXISTS last_new_article_at timestamptz;

-- Backfill from the articles actually stored per feed.  Without this, every
-- feed would look "never produced an article" and fall back to the created_at
-- grace period, which would wrongly mark old-but-healthy feeds dormant.
UPDATE feeds f
SET    last_new_article_at = a.last_created
FROM  (SELECT feed_id, max(created_at) AS last_created
       FROM   articles
       GROUP  BY feed_id) a
WHERE  a.feed_id = f.id
  AND  f.last_new_article_at IS NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 2: feeds.created_at (the dormancy grace-period fallback reads it)
-- On a table that already has the column this is a no-op.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE feeds ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 3: feeds.disabled_reason -- tells dormancy apart from error-disable
--
--   'dormant' -> no new articles for DORMANCY_DAYS.  Re-polled once every
--                DORMANT_RECHECK_INTERVAL_DAYS; re-activated automatically if
--                it produces an article again.
--   'errors'  -> MAX_ERRORS consecutive fetch failures.  Never auto-polled;
--                needs a manual reset:
--                  UPDATE feeds SET is_active=true, fail_count=0,
--                                   disabled_reason=NULL WHERE id='...';
--   NULL      -> active, or paused by hand (never auto re-checked).
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE feeds ADD COLUMN IF NOT EXISTS disabled_reason text;

-- Feeds already disabled by an error streak can be labelled reliably.
-- Feeds disabled by the OLD dormancy code cannot be told apart from manually
-- paused feeds, so they are left NULL.  If you know which ones were dormant:
--   UPDATE feeds SET disabled_reason='dormant' WHERE id IN (...);
UPDATE feeds
SET    disabled_reason = 'errors'
WHERE  is_active = false
  AND  disabled_reason IS NULL
  AND  fail_count >= 5;

CREATE INDEX IF NOT EXISTS feeds_dormant_recheck_idx
  ON feeds (last_polled_at NULLS FIRST)
  WHERE is_active = false AND disabled_reason = 'dormant';


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 4: update_cadence must never be NULL
--
-- No workflow polls a NULL cadence (ingest_daily.yml polls 'daily' AND
-- 'unknown').  push_feeds.py now defaults to 'unknown'; this fixes old rows
-- and makes the database do it too.
-- ─────────────────────────────────────────────────────────────────────────────

UPDATE feeds SET update_cadence = 'unknown'
WHERE  update_cadence IS NULL OR btrim(update_cadence) = '';

ALTER TABLE feeds ALTER COLUMN update_cadence SET DEFAULT 'unknown';


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 5: pipeline_runs.tier
--
-- log_run_start/log_run_finish always send 'tier' (legacy name of 'cadence').
-- docs/migration.sql now adds it too; repeated here for deployments that ran
-- the older migration.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS tier text;
ALTER TABLE pipeline_runs ADD COLUMN IF NOT EXISTS cadence text;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 6: wizer_prune_articles — table-size cap that actually converges
--
-- The old Python pruning asked PostgREST for the `excess` oldest ids, but
-- PostgREST caps every response at max_rows (1000 on Supabase): once the
-- table passed ARTICLE_HARD_LIMIT, each run deleted at most 1000 rows while
-- ingestion added far more, so the table never came back under the limit.
-- It also sorted 2M rows by created_at, which has no index.
--
-- This function walks the PRIMARY KEY instead (articles.id is an identity
-- column, so id order IS insertion order) and deletes at most p_max_delete
-- rows per call, keeping each statement short; the caller loops until it
-- returns fewer than p_max_delete.
-- article_entities rows go with their article (ON DELETE CASCADE); story
-- clusters are reconciled by `cluster.py maintain`.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION wizer_prune_articles(
  p_hard_limit  bigint,
  p_target      bigint,
  p_max_delete  integer DEFAULT 50000
)
RETURNS integer
LANGUAGE plpgsql
VOLATILE
SET search_path = public
AS $$
DECLARE
  v_total   bigint;
  v_cutoff  bigint;
  v_deleted integer;
BEGIN
  SELECT count(*) INTO v_total FROM articles;
  IF v_total <= p_hard_limit THEN
    RETURN 0;
  END IF;
  -- Newest p_target rows are kept; everything at or below the cutoff id goes.
  SELECT id INTO v_cutoff FROM articles ORDER BY id DESC OFFSET p_target LIMIT 1;
  IF v_cutoff IS NULL THEN
    RETURN 0;
  END IF;
  WITH doomed AS (
    SELECT id FROM articles WHERE id <= v_cutoff ORDER BY id LIMIT greatest(p_max_delete, 1)
  ), gone AS (
    DELETE FROM articles a USING doomed d WHERE a.id = d.id RETURNING 1
  )
  SELECT count(*)::integer INTO v_deleted FROM gone;
  RETURN v_deleted;
END;
$$;

REVOKE ALL ON FUNCTION wizer_prune_articles(bigint, bigint, integer) FROM PUBLIC;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    REVOKE ALL ON FUNCTION wizer_prune_articles(bigint, bigint, integer) FROM anon, authenticated;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    GRANT EXECUTE ON FUNCTION wizer_prune_articles(bigint, bigint, integer) TO service_role;
  END IF;
END $$;
