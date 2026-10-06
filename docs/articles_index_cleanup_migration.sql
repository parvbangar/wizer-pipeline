-- =============================================================================
-- docs/articles_index_cleanup_migration.sql
-- Fewer indexes on articles: every insert maintains every index.
--
-- Run AFTER docs/bulk_io_migration.sql. Idempotent.
--
-- On 2026-10-05 articles carried 26 indexes. On the Supabase Micro instance
-- (burstable CPU / disk I/O) each inserted row paid for all of them, including
-- a GIN full-text index. Dropped here:
--   - exact duplicates: idx_articles_url_hash (= articles_url_hash_uidx),
--     idx_articles_is_crawled (= articles_is_crawled_idx), idx_articles_feed_id
--     (prefix of idx_articles_feed_id_crawled), articles_enriched_at_idx
--     (replaced by articles_enrich_queue_v2_idx);
--   - indexes production had never scanned (pg_stat_user_indexes.idx_scan = 0)
--     and no pipeline query needs.
-- Kept: the primary key, the url / url_hash unique indexes (dedup), crawled_at,
-- published_at, feed_id + crawled_at, cluster_id and the two queue indexes.
-- Recreate any of these if an app starts querying by that column.
--
-- The earlier migrations still create some of them; on a replay they are
-- created and then dropped again here.
-- =============================================================================

DROP INDEX IF EXISTS idx_articles_url_hash;
DROP INDEX IF EXISTS idx_articles_is_crawled;
DROP INDEX IF EXISTS articles_is_crawled_idx;
DROP INDEX IF EXISTS idx_articles_feed_id;
DROP INDEX IF EXISTS articles_enriched_at_idx;
DROP INDEX IF EXISTS articles_title_fts_idx;
DROP INDEX IF EXISTS articles_domain_published_idx;
DROP INDEX IF EXISTS articles_iab_published_idx;
DROP INDEX IF EXISTS articles_lang_published_idx;
DROP INDEX IF EXISTS articles_feed_published_idx;
DROP INDEX IF EXISTS articles_title_simhash_idx;
DROP INDEX IF EXISTS articles_crawl_strategy_idx;
DROP INDEX IF EXISTS idx_articles_story_id;
DROP INDEX IF EXISTS articles_is_duplicate_idx;
DROP INDEX IF EXISTS articles_category_published_idx;
DROP INDEX IF EXISTS articles_image_phash_idx;
DROP INDEX IF EXISTS articles_sentiment_published_idx;
