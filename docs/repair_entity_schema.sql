-- repair_entity_schema.sql
--
-- STATUS: NO REPAIR NEEDED.
--
-- Investigation result (2026-03-28):
--   articles.id                          → bigint  ✓
--   article_entities.article_id          → bigint  ✓  (matches articles.id)
--   article_clusters.canonical_article_id → bigint  ✓  (matches articles.id)
--   article_entities.id                  → uuid    ✓
--   article_clusters.id                  → uuid    ✓
--
-- The live schema is internally consistent. The earlier hypothesis that
-- articles.id was uuid was wrong — migration.sql had an error (uuid) that
-- was fixed to match the live schema (bigint GENERATED ALWAYS AS IDENTITY).
--
-- This file is kept for audit trail. No SQL needs to be run.

-- Verify schema is healthy (run this to confirm):
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name   IN ('articles', 'article_entities', 'article_clusters')
  AND column_name  IN ('id', 'article_id', 'canonical_article_id')
ORDER BY table_name, column_name;

-- Expected:
--   article_clusters  | canonical_article_id | bigint
--   article_clusters  | id                   | uuid
--   article_entities  | article_id           | bigint
--   article_entities  | id                   | uuid
--   articles          | id                   | bigint
