-- =============================================================================
-- docs/coverage_range_migration.sql
-- wizer_domain_counts_range: articles per domain in a time window.
--
-- Run AFTER docs/cluster_state_delta_migration.sql. Idempotent.
--
-- WHY (2026-10-07): wizer_domain_counts(day) reads a whole day of articles in one
-- statement. On Micro's disk that took 25–75 s and broke the 30 s API timeout,
-- so the daily coverage audit failed. tools/coverage_audit.py now asks for one
-- hour at a time: the same pages are read in total, but each call is short.
-- Adding a covering index instead would cost every article insert.
-- =============================================================================

CREATE OR REPLACE FUNCTION wizer_domain_counts_range(p_from timestamptz, p_to timestamptz)
RETURNS TABLE (domain text, n integer)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT coalesce(nullif(lower(a.domain), ''), '(unknown)'), count(*)::integer
    FROM articles AS a
   WHERE a.crawled_at >= p_from AND a.crawled_at < p_to
   GROUP BY 1;
$$;

DO $$
DECLARE
  fn text := 'wizer_domain_counts_range(timestamptz, timestamptz)';
  r  text;
BEGIN
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', fn);
  FOREACH r IN ARRAY ARRAY['anon', 'authenticated'] LOOP
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
      EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I', fn, r);
    END IF;
  END LOOP;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', fn);
  END IF;
END $$;
