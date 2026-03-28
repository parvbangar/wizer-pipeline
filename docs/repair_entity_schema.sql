-- repair_entity_schema.sql
-- Run this in Supabase SQL Editor to check and fix the bigint→uuid type bug.
--
-- BACKGROUND:
--   enrichment_migration.sql had article_entities.article_id declared as bigint
--   and article_clusters.canonical_article_id as bigint. Both should be uuid
--   because articles.id is uuid. PostgreSQL silently drops FK constraints that
--   have type mismatches in some cases, leaving the column with the wrong type.
--
-- STEP 1: Check what types you actually have in the live DB
-- ──────────────────────────────────────────────────────────
SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name   IN ('article_entities', 'article_clusters', 'articles')
  AND column_name  IN ('id', 'article_id', 'canonical_article_id')
ORDER BY table_name, column_name;

-- Expected result:
--   article_clusters  | canonical_article_id | uuid
--   article_entities  | article_id           | uuid
--   articles          | id                   | uuid
--
-- If article_entities.article_id shows "bigint", run STEP 2.
-- If it already shows "uuid", you are fine — skip to STEP 3.


-- STEP 2: Fix article_entities (only if article_id is bigint)
-- ─────────────────────────────────────────────────────────────
-- Since bigint values cannot be cast to uuid, we drop and recreate the table.
-- This is safe because you are planning to truncate and restart anyway.

DROP TABLE IF EXISTS article_entities CASCADE;

CREATE TABLE article_entities (
  id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  article_id   uuid        NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  entity_text  text        NOT NULL,
  entity_type  text        NOT NULL,
  salience     float       NOT NULL DEFAULT 0.5,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS article_entities_article_idx
  ON article_entities (article_id);

CREATE INDEX IF NOT EXISTS article_entities_type_text_idx
  ON article_entities (entity_type, entity_text);

CREATE INDEX IF NOT EXISTS article_entities_salience_idx
  ON article_entities (entity_text, salience DESC)
  WHERE salience >= 0.5;


-- STEP 3: Fix article_clusters.canonical_article_id (only if bigint)
-- ─────────────────────────────────────────────────────────────────────
-- canonical_article_id is a soft FK (no constraint), so we can ALTER in-place.
-- Any existing bigint values will be set to NULL (they were invalid anyway).

ALTER TABLE article_clusters
  ALTER COLUMN canonical_article_id TYPE uuid USING NULL;

-- Verify:
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'article_clusters'
  AND column_name = 'canonical_article_id';
-- Should show: uuid


-- STEP 4: Confirm both tables look correct
-- ──────────────────────────────────────────
SELECT
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name   IN ('article_entities', 'article_clusters')
  AND column_name  IN ('article_id', 'canonical_article_id')
ORDER BY table_name;
