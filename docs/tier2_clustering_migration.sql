-- tier2_clustering_migration.sql
-- Run once in Supabase SQL Editor.
--
-- WHAT THIS DOES:
--   1. Adds outlet_set jsonb column to article_clusters
--      — tracks which domains have contributed to a cluster so
--        outlet_count is incremented once per domain, not per article
--   2. Updates find_nearest_cluster() to return outlet_set
--   3. Adds find_duplicate_clusters() for the weekly merge job
--
-- WHY outlet_set IS NEEDED:
--   Before this migration, outlet_count was incremented for every article
--   because the domain dedup check compared against entity_set (entity texts).
--   Domains ("thehindu.com") never appear in entity text ("Narendra Modi"),
--   so the check always passed and outlet_count == article_count.
--   outlet_set is the correct dedup mechanism.


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Add outlet_set column
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE article_clusters
  ADD COLUMN IF NOT EXISTS outlet_set jsonb NOT NULL DEFAULT '[]';

-- GIN index for fast containment checks: "is domain X in outlet_set?"
CREATE INDEX IF NOT EXISTS idx_clusters_outlet_set
  ON article_clusters USING gin (outlet_set);


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Update find_nearest_cluster() to include outlet_set in result
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION find_nearest_cluster(
  query_embedding      vector(768),
  similarity_threshold float   DEFAULT 0.82,
  window_hours         integer DEFAULT 48
)
RETURNS TABLE (
  id                   uuid,
  canonical_article_id bigint,
  headline             text,
  outlet_count         integer,
  article_count        integer,
  entity_set           jsonb,
  outlet_set           jsonb,
  top_entities         jsonb,
  canonical_embedding  jsonb,
  first_seen_at        timestamptz,
  last_seen_at         timestamptz,
  similarity           float
)
LANGUAGE sql STABLE
AS $$
  SELECT
    id,
    canonical_article_id,
    headline,
    outlet_count,
    article_count,
    entity_set,
    outlet_set,
    top_entities,
    canonical_embedding,
    first_seen_at,
    last_seen_at,
    1 - (embedding_vec <=> query_embedding) AS similarity
  FROM article_clusters
  WHERE embedding_vec IS NOT NULL
    AND (
      window_hours = 0
      OR last_seen_at > now() - make_interval(hours => window_hours)
    )
    AND 1 - (embedding_vec <=> query_embedding) >= similarity_threshold
  ORDER BY embedding_vec <=> query_embedding
  LIMIT 1;
$$;

GRANT EXECUTE ON FUNCTION find_nearest_cluster TO service_role;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. find_duplicate_clusters() — used by weekly merge job
--
-- Finds pairs of clusters whose embeddings are above similarity_threshold.
-- Uses the HNSW index via LATERAL join — efficient for large tables.
-- Returns pairs (cluster_a, cluster_b) where cluster_a.id < cluster_b.id
-- so each pair appears exactly once.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION find_duplicate_clusters(
    similarity_threshold float   DEFAULT 0.87,
    max_pairs            integer DEFAULT 200
)
RETURNS TABLE (
    cluster_a   uuid,
    cluster_b   uuid,
    similarity  float,
    a_count     integer,
    b_count     integer
)
LANGUAGE sql STABLE
AS $$
    SELECT
        a.id                                          AS cluster_a,
        near.id                                       AS cluster_b,
        1 - (a.embedding_vec <=> near.embedding_vec)  AS similarity,
        a.article_count,
        near.article_count
    FROM article_clusters a
    CROSS JOIN LATERAL (
        SELECT id, embedding_vec, article_count
        FROM article_clusters c
        WHERE c.id != a.id
          AND c.embedding_vec IS NOT NULL
          -- Use string cast so uuid comparisons are consistent
          AND c.id::text > a.id::text
          AND 1 - (c.embedding_vec <=> a.embedding_vec) >= similarity_threshold
        ORDER BY c.embedding_vec <=> a.embedding_vec
        LIMIT 1
    ) near
    WHERE a.embedding_vec IS NOT NULL
    LIMIT max_pairs;
$$;

GRANT EXECUTE ON FUNCTION find_duplicate_clusters TO service_role;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Verify
-- ─────────────────────────────────────────────────────────────────────────────
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'article_clusters'
  AND column_name = 'outlet_set';
-- Should return: outlet_set | jsonb
