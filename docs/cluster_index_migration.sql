-- Fix: add index on article_clusters.last_seen_at
-- Without this, fetch_recent_clusters does a full table scan and times out.
-- Run once in Supabase SQL Editor.

CREATE INDEX IF NOT EXISTS idx_clusters_last_seen_at
  ON article_clusters (last_seen_at DESC);

-- Also index canonical_embedding to speed up future pgvector queries
-- (no-op until you add pgvector, but good to have)
CREATE INDEX IF NOT EXISTS idx_clusters_canonical_article
  ON article_clusters (canonical_article_id);
