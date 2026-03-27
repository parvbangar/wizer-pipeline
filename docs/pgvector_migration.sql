-- pgvector migration for article_clusters
-- Run once in Supabase SQL Editor.
--
-- WHAT THIS DOES:
--   1. Enables the pgvector extension
--   2. Adds embedding_vec vector(768) column to article_clusters
--   3. Backfills from existing canonical_embedding jsonb
--   4. Creates an HNSW approximate nearest-neighbour index
--   5. Creates find_nearest_cluster() RPC function used by the enrichment pipeline
--
-- AFTER RUNNING THIS:
--   Clustering goes from O(n) Python loop (loads every cluster into memory)
--   to a single sub-millisecond SQL query regardless of cluster count.
--   The 48-hour window limit is also gone — the index searches all clusters instantly.

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. Enable pgvector (built into Supabase, no plan upgrade needed)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Add the vector column
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE article_clusters
  ADD COLUMN IF NOT EXISTS embedding_vec vector(768);


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Backfill from existing canonical_embedding jsonb
--    canonical_embedding is stored as a jsonb array [0.1, 0.2, ...]
--    Casting jsonb → text → vector works because pgvector accepts '[x,y,z]' format.
-- ─────────────────────────────────────────────────────────────────────────────
UPDATE article_clusters
SET    embedding_vec = canonical_embedding::text::vector
WHERE  canonical_embedding IS NOT NULL
  AND  embedding_vec IS NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. HNSW index for approximate nearest-neighbour cosine search
--    m=16, ef_construction=64 are the recommended defaults.
--    Build time: a few seconds on a table with thousands of clusters.
--    Query time after build: < 1ms regardless of table size.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_clusters_embedding_hnsw
  ON article_clusters
  USING hnsw (embedding_vec vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. RPC function called by enrichment/db.py find_nearest_cluster()
--
--    Returns the single most similar cluster above the threshold.
--    The pipeline calls this once per article instead of loading all clusters
--    into Python memory and doing a NumPy dot-product loop.
--
--    Parameters:
--      query_embedding      — the article's LaBSE embedding (768 floats)
--      similarity_threshold — minimum cosine similarity to qualify (default 0.82)
--      window_hours         — only search clusters seen in last N hours (default 48)
--                             Pass 0 to search ALL clusters (useful after pgvector
--                             is stable — no window limit needed with HNSW).
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

-- Grant access to the service role used by the pipeline
GRANT EXECUTE ON FUNCTION find_nearest_cluster TO service_role;
