-- =============================================================================
-- docs/cluster_state_delta_migration.sql
-- wizer_cluster_state: an index for the delta, and one plan per branch.
--
-- Run AFTER docs/cluster_hnsw_drop_migration.sql. Idempotent.
--
-- WHY (2026-10-07): the delta read (p_changed_since set: "clusters changed since
-- the bookmark", used after every twin merge and at every job start) hit the
-- 30 s API statement timeout and crashed a clustering run at its very end.
--   - Nothing indexed article_clusters.updated_at.
--   - The single SQL statement chose its filter with a CASE expression, which
--     no index can serve, so each 200-row page walked the primary key over the
--     whole table, and each row is ~3 KB of vectors.
-- Now: an updated_at index, and a plpgsql body with one RETURN QUERY per branch,
-- each planned for its own filter (delta: updated_at; full load: last_seen_at,
-- idx_clusters_last_seen_at). The signature, result columns and order are unchanged.
-- =============================================================================

CREATE INDEX IF NOT EXISTS article_clusters_updated_at_idx ON article_clusters (updated_at);

CREATE OR REPLACE FUNCTION wizer_cluster_state(
  p_model          text,
  p_changed_since  timestamptz,
  p_window_since   timestamptz,
  p_limit          integer DEFAULT 200,
  p_after_id       uuid    DEFAULT '00000000-0000-0000-0000-000000000000'
)
RETURNS TABLE (
  id                         uuid,
  status                     text,
  embedding_model            text,
  centroid_sum               text,
  anchor                     text,
  representative             text,
  representative_article_id  bigint,
  headline                   text,
  article_count              integer,
  outlet_set                 jsonb,
  language_set               jsonb,
  first_seen_at              timestamptz,
  last_seen_at               timestamptz,
  updated_at                 timestamptz
)
LANGUAGE plpgsql
STABLE
SET search_path = public, extensions
AS $$
BEGIN
  IF p_changed_since IS NOT NULL THEN
    RETURN QUERY
    SELECT c.id, c.status, c.embedding_model,
           c.centroid_sum::text, c.anchor::text, c.representative::text,
           c.representative_article_id, c.headline, c.article_count,
           c.outlet_set, c.language_set, c.first_seen_at, c.last_seen_at, c.updated_at
      FROM article_clusters AS c
     WHERE c.updated_at > p_changed_since
       AND c.embedding_model = p_model
       AND c.id > p_after_id
     ORDER BY c.id
     LIMIT greatest(p_limit, 1);
  ELSE
    RETURN QUERY
    SELECT c.id, c.status, c.embedding_model,
           c.centroid_sum::text, c.anchor::text, c.representative::text,
           c.representative_article_id, c.headline, c.article_count,
           c.outlet_set, c.language_set, c.first_seen_at, c.last_seen_at, c.updated_at
      FROM article_clusters AS c
     WHERE c.last_seen_at >= p_window_since
       AND c.status = 'active'
       AND c.embedding_model = p_model
       AND c.id > p_after_id
     ORDER BY c.id
     LIMIT greatest(p_limit, 1);
  END IF;
END;
$$;

DO $$
DECLARE
  fn text := 'wizer_cluster_state(text, timestamptz, timestamptz, integer, uuid)';
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
