-- =============================================================================
-- docs/archive_guard_migration.sql
-- The archive never moves a day out of Postgres while it still holds
-- unenriched articles.
--
-- Run AFTER docs/articles_index_cleanup_migration.sql. Idempotent.
--
-- WHY: every ingested article must be enriched (enrichment_queue_v2). The
-- archive deletes days older than the hot window (ARCHIVE_HOT_DAYS, 14 days
-- since 2026-10-06 — ~80K articles/day would outgrow the Micro disk at 30).
-- If a backlog ever outlived that window, its articles would leave Postgres
-- unenriched. archiver/runner.py asks this function first and waits.
--
-- Excluded: dead letters that have exhausted every retry (enrich_attempts
-- >= 10, enrichment_queue_health.given_up) — they will never be enriched and
-- must not block retention forever; they are archived as they are.
-- =============================================================================

CREATE OR REPLACE FUNCTION wizer_archive_day_unenriched(p_day date)
RETURNS integer
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT count(*)::integer
    FROM articles
   WHERE crawled_at >= p_day
     AND crawled_at <  p_day + 1
     AND enriched_at IS NULL
     AND title IS NOT NULL
     AND enrich_attempts < 10;
$$;

DO $$
BEGIN
  EXECUTE 'REVOKE ALL ON FUNCTION wizer_archive_day_unenriched(date) FROM PUBLIC';
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    EXECUTE 'REVOKE ALL ON FUNCTION wizer_archive_day_unenriched(date) FROM anon, authenticated';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'GRANT EXECUTE ON FUNCTION wizer_archive_day_unenriched(date) TO service_role';
  END IF;
END $$;
