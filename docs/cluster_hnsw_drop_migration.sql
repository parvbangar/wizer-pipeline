-- =============================================================================
-- docs/cluster_hnsw_drop_migration.sql
-- No approximate-nearest-neighbour index on article_clusters.
--
-- Run AFTER docs/coverage_migration.sql. Idempotent.
--
-- WHY: every cluster write maintained an HNSW graph over the centroids, the
-- most CPU-expensive index there is. On the Micro instance a single 500-article
-- page of clustering results took over 12 minutes to write (2026-10-06).
-- Nothing on the hot path searches vectors in SQL any more:
--   - story assignment runs in memory (enrichment/memory_clustering.py);
--   - twin detection (the maintenance merge sweep, the index's only user)
--     runs in memory too (ClusterState.merge_candidates, cluster_job.merge_twins).
-- wizer_find_cluster_merge_candidates still works without the index (an
-- exact scan over the time window) for manual / test use.
-- =============================================================================

DROP INDEX IF EXISTS article_clusters_v2_centroid_hnsw;
DROP INDEX IF EXISTS idx_clusters_embedding_hnsw;      -- v1 (pgvector_migration.sql)

-- Indexes production never scanned (pg_stat_user_indexes, 2026-10-06) that
-- every cluster write still maintained:
DROP INDEX IF EXISTS article_clusters_last_seen_idx;   -- duplicate of idx_clusters_last_seen_at
DROP INDEX IF EXISTS idx_clusters_outlet_set;          -- v1 GIN over outlet_set, rewritten on every join
DROP INDEX IF EXISTS article_clusters_simhash_idx;     -- v1, canonical_simhash is unused
