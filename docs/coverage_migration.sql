-- =============================================================================
-- docs/coverage_migration.sql
-- Daily completeness measurements (tools/coverage_audit.py, coverage.yml).
--
-- Run AFTER docs/archive_guard_migration.sql. Idempotent.
--
-- One row per (day, domain, reference):
--   reference 'sitemap'  the publisher's own news sitemap for that day is the
--                        ground truth: missing = listed but not stored.
--   reference 'gdelt'    GDELT's independent sample of the domain's URLs that
--                        day; with our own count it gives a Lincoln–Petersen
--                        estimate of the domain's true daily output
--                        (est_universe = ours × reference / both).
--   reference 'ingest'   our own stored count (yield; a sudden drop is a dead
--                        source).
-- A few hundred rows a day — negligible on Micro.
-- =============================================================================

CREATE TABLE IF NOT EXISTS coverage_daily (
  day            date        NOT NULL,
  domain         text        NOT NULL,
  reference      text        NOT NULL,          -- sitemap | gdelt | ingest
  ref_count      integer     NOT NULL DEFAULT 0,  -- URLs the reference saw
  ours           integer     NOT NULL DEFAULT 0,  -- of those, URLs we stored (ingest: all we stored)
  missing        integer     NOT NULL DEFAULT 0,
  miss_rate      real,
  est_universe   real,                            -- gdelt only: capture–recapture estimate
  sample_missing jsonb       NOT NULL DEFAULT '[]', -- up to 10 missed URLs, for diagnosis
  measured_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (day, domain, reference)
);

CREATE INDEX IF NOT EXISTS coverage_daily_day_idx ON coverage_daily (day DESC);

-- Our own daily output per domain, computed in the database (one indexed
-- range scan over a single day of crawled_at).
CREATE OR REPLACE FUNCTION wizer_domain_counts(p_day date)
RETURNS TABLE (domain text, n integer)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT coalesce(nullif(lower(domain), ''), '(unknown)'), count(*)::integer
    FROM articles
   WHERE crawled_at >= p_day AND crawled_at < p_day + 1
   GROUP BY 1;
$$;

CREATE OR REPLACE VIEW coverage_summary AS
SELECT day, reference,
       count(*)                                         AS domains,
       sum(ref_count)                                   AS ref_urls,
       sum(ours)                                        AS ours,
       sum(missing)                                     AS missing,
       round((100.0 * sum(missing) / nullif(sum(ref_count), 0))::numeric, 2) AS miss_pct
  FROM coverage_daily
 GROUP BY day, reference;

DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    EXECUTE 'REVOKE ALL ON TABLE coverage_daily, coverage_summary FROM anon, authenticated';
    EXECUTE 'REVOKE ALL ON FUNCTION wizer_domain_counts(date) FROM anon, authenticated';
  END IF;
  EXECUTE 'REVOKE ALL ON FUNCTION wizer_domain_counts(date) FROM PUBLIC';
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE coverage_daily TO service_role';
    EXECUTE 'GRANT SELECT ON TABLE coverage_summary TO service_role';
    EXECUTE 'GRANT EXECUTE ON FUNCTION wizer_domain_counts(date) TO service_role';
  END IF;
END $$;
