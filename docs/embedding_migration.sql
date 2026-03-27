-- =============================================================================
-- docs/embedding_migration.sql
-- Add canonical_embedding column to article_clusters
--
-- Run ONCE in Supabase → SQL Editor.
--
-- WHY:
--   Clustering now uses LaBSE sentence embeddings instead of entity Jaccard.
--   Each cluster stores its canonical article's 768-dim embedding vector so
--   the enrichment runner can load recent clusters into memory and compare
--   new articles against them via cosine similarity.
--
--   Stored as jsonb (array of 768 floats). PostgreSQL pgvector would be faster
--   for large-scale similarity search, but jsonb + in-memory comparison is
--   sufficient for the current batch size (500 articles × ~10K clusters).
-- =============================================================================

ALTER TABLE article_clusters
  ADD COLUMN IF NOT EXISTS canonical_embedding jsonb;

-- Index is not useful here — jsonb similarity search is always a full scan.
-- When you upgrade to Supabase Pro, enable pgvector and migrate:
--   ALTER TABLE article_clusters ADD COLUMN embedding vector(768);
--   CREATE INDEX ON article_clusters USING ivfflat (embedding vector_cosine_ops);
