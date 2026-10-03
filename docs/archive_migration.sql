-- =============================================================================
-- docs/archive_migration.sql
-- Article archive: export old articles to Parquet in Supabase Storage, then
-- delete them from Postgres — only after the export is verified.
--
-- Run AFTER clustering_v2_migration.sql. Idempotent.
--
-- WHY:
--   Production held 2.98M articles / 13.2 GB on 2026-10-03 and grows ~50K
--   articles (~0.2 GB) a day. The same rows as zstd Parquet are ~3.3x smaller
--   and Storage is ~6x cheaper per GB than database disk — and a lean articles
--   table keeps backups, vacuum and scans fast. The app and enrichment only
--   touch the last 48 h, so the database keeps a 30-day hot window.
--
-- SAFETY MODEL (enforced HERE, not just in Python):
--   archive.py writes a day's Parquet parts, re-downloads and verifies them,
--   and only then marks the day 'verified'. wizer_prune_archived_day() refuses
--   to delete anything unless the day is verified, is older than the hot
--   window, and still holds exactly the rows that were archived (so rows that
--   appeared after the export can never be deleted unarchived).
--
-- UNIT: one UTC day of articles.crawled_at (NOT NULL, set at ingestion, and
-- indexed — created_at has no index on the 13 GB production table).
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. article_archive_log — one row per archived day
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS article_archive_log (
  day              date        PRIMARY KEY,
  status           text        NOT NULL DEFAULT 'uploaded',
  article_rows     integer     NOT NULL,
  entity_rows      integer     NOT NULL DEFAULT 0,
  cluster_rows     integer     NOT NULL DEFAULT 0,
  min_article_id   bigint,
  max_article_id   bigint,
  objects          jsonb       NOT NULL DEFAULT '[]',   -- [{path, rows, bytes, sha256}]
  total_bytes      bigint      NOT NULL DEFAULT 0,
  bucket           text        NOT NULL,
  archived_at      timestamptz NOT NULL DEFAULT now(),
  verified_at      timestamptz,
  pruned_rows      integer     NOT NULL DEFAULT 0,
  pruned_at        timestamptz,
  CONSTRAINT article_archive_log_status_chk
    CHECK (status IN ('uploaded', 'verified', 'pruned'))
);

ALTER TABLE article_archive_log ENABLE ROW LEVEL SECURITY;   -- service role only

-- Explicit table grants. Do NOT rely on default privileges: on this Supabase
-- project, tables created by `postgres` get only DELETE/TRUNCATE/REFERENCES/
-- TRIGGER for service_role (no SELECT/INSERT/UPDATE → "permission denied" from
-- the API) and DELETE/TRUNCATE for anon and authenticated.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
    EXECUTE 'REVOKE ALL ON TABLE article_archive_log FROM anon, authenticated';
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE article_archive_log TO service_role';
  END IF;
END $$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. wizer_archive_day_stats — what a day holds right now
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_archive_day_stats(p_day date)
RETURNS TABLE (article_rows bigint, min_id bigint, max_id bigint)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  SELECT count(*), min(id), max(id)
    FROM articles
   WHERE crawled_at >= p_day
     AND crawled_at <  p_day + 1;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. wizer_archive_days — days that are old enough and not yet pruned
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_archive_days(p_hot_days integer DEFAULT 30)
RETURNS TABLE (day date, status text)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  WITH bounds AS (
    SELECT min(crawled_at)::date AS first_day,
           (now() AT TIME ZONE 'UTC')::date - greatest(p_hot_days, 1) AS last_day
      FROM articles
  )
  SELECT d::date, l.status
    FROM bounds b
   CROSS JOIN LATERAL generate_series(b.first_day, b.last_day - 1, interval '1 day') AS d
    LEFT JOIN article_archive_log AS l ON l.day = d::date
   WHERE b.first_day IS NOT NULL
     AND coalesce(l.status, '') <> 'pruned'
   ORDER BY d;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. wizer_archive_page — full article rows of one day, keyset-paged by id
--
-- SETOF articles, so the export always carries every column, including ones
-- added after this function was written.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_archive_page(
  p_day      date,
  p_after_id bigint  DEFAULT 0,
  p_limit    integer DEFAULT 1000
)
RETURNS SETOF articles
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  SELECT *
    FROM articles
   WHERE crawled_at >= p_day
     AND crawled_at <  p_day + 1
     AND id > p_after_id
   ORDER BY id
   LIMIT greatest(least(p_limit, 1000), 1);
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4b. Entities and clusters of one day (exported next to the articles)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_archive_entities_page(
  p_day         date,
  p_after_key   text    DEFAULT '',
  p_limit       integer DEFAULT 1000
)
RETURNS TABLE (page_key text, article_id bigint, entity_text text, entity_type text,
               salience float8, created_at timestamptz)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  -- page_key = zero-padded article_id || entity id: a total order usable as a keyset
  SELECT lpad(e.article_id::text, 20, '0') || ':' || e.id::text AS page_key,
         e.article_id, e.entity_text, e.entity_type, e.salience, e.created_at
    FROM article_entities AS e
    JOIN articles AS a ON a.id = e.article_id
   WHERE a.crawled_at >= p_day
     AND a.crawled_at <  p_day + 1
     AND lpad(e.article_id::text, 20, '0') || ':' || e.id::text > p_after_key
   ORDER BY 1
   LIMIT greatest(least(p_limit, 1000), 1);
$$;

-- Story clusters that the day's articles belong to — metadata only (the
-- 768-dim vectors are model-specific working state, not research data).
-- A cluster spanning several days appears in each day's file; dedupe by id,
-- keeping the row with the latest updated_at.
CREATE OR REPLACE FUNCTION wizer_archive_clusters(p_day date)
RETURNS TABLE (id uuid, headline text, article_count integer, outlet_count integer,
               outlet_set jsonb, entity_set jsonb, top_entities jsonb, language_set jsonb,
               canonical_article_id bigint, representative_article_id bigint,
               embedding_model text, status text, merged_into uuid,
               first_seen_at timestamptz, last_seen_at timestamptz,
               created_at timestamptz, updated_at timestamptz)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  SELECT c.id, c.headline, c.article_count, c.outlet_count, c.outlet_set, c.entity_set,
         c.top_entities, c.language_set, c.canonical_article_id, c.representative_article_id,
         c.embedding_model, c.status, c.merged_into, c.first_seen_at, c.last_seen_at,
         c.created_at, c.updated_at
    FROM article_clusters AS c
   WHERE c.id IN (SELECT DISTINCT a.cluster_id FROM articles AS a
                   WHERE a.crawled_at >= p_day AND a.crawled_at < p_day + 1
                     AND a.cluster_id IS NOT NULL);
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. wizer_prune_archived_day — delete one verified day, in bounded chunks
--
-- Refuses (raises) unless:
--   - the day's log row is 'verified' (or a partially pruned 'verified' day)
--   - the day is older than p_min_age_days (the hot window)
--   - rows still in the table + rows already pruned == rows archived
--     (anything added after the export would break this equality)
-- Deletes at most p_max_delete rows per call (stays inside the 30 s API
-- statement timeout); call until it returns 0. Entities go with their
-- articles (ON DELETE CASCADE). Marks the day 'pruned' when empty.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_prune_archived_day(
  p_day           date,
  p_min_age_days  integer DEFAULT 30,
  p_max_delete    integer DEFAULT 5000
)
RETURNS integer
LANGUAGE plpgsql
VOLATILE
SET search_path = public
AS $$
DECLARE
  l          article_archive_log%ROWTYPE;
  v_left     bigint;
  v_deleted  integer;
BEGIN
  SELECT * INTO l FROM article_archive_log WHERE day = p_day FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'archive prune: day % has no archive log row', p_day;
  END IF;
  IF l.status = 'pruned' THEN
    RETURN 0;
  END IF;
  IF l.status <> 'verified' THEN
    RAISE EXCEPTION 'archive prune: day % is %, not verified', p_day, l.status;
  END IF;
  IF p_day >= (now() AT TIME ZONE 'UTC')::date - greatest(p_min_age_days, 1) THEN
    RAISE EXCEPTION 'archive prune: day % is inside the % day hot window', p_day, p_min_age_days;
  END IF;

  SELECT count(*) INTO v_left
    FROM articles WHERE crawled_at >= p_day AND crawled_at < p_day + 1;
  IF v_left + l.pruned_rows <> l.article_rows THEN
    RAISE EXCEPTION 'archive prune: day % changed since export (% in table + % pruned <> % archived) — re-archive it',
      p_day, v_left, l.pruned_rows, l.article_rows;
  END IF;

  WITH doomed AS (
    SELECT id FROM articles
     WHERE crawled_at >= p_day AND crawled_at < p_day + 1
     ORDER BY id
     LIMIT greatest(p_max_delete, 1)
  ), gone AS (
    DELETE FROM articles a USING doomed d WHERE a.id = d.id RETURNING 1
  )
  SELECT count(*)::integer INTO v_deleted FROM gone;

  UPDATE article_archive_log
     SET pruned_rows = pruned_rows + v_deleted,
         status      = CASE WHEN v_left - v_deleted = 0 THEN 'pruned' ELSE status END,
         pruned_at   = CASE WHEN v_left - v_deleted = 0 THEN now() ELSE pruned_at END
   WHERE day = p_day;
  RETURN v_deleted;
END;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Permissions — service role only
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
DECLARE
  fn text;
BEGIN
  FOREACH fn IN ARRAY ARRAY[
    'wizer_archive_day_stats(date)',
    'wizer_archive_days(integer)',
    'wizer_archive_page(date, bigint, integer)',
    'wizer_archive_entities_page(date, text, integer)',
    'wizer_archive_clusters(date)',
    'wizer_prune_archived_day(date, integer, integer)'
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
