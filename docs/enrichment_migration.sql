-- =============================================================================
-- docs/enrichment_migration.sql
-- Layer 2 — Metadata Enrichment Schema
--
-- Run this ONCE in Supabase → SQL Editor BEFORE running enrich.py.
-- All statements are idempotent (safe to run multiple times).
--
-- WHAT THIS CREATES:
--   1. New columns on the existing articles table (enrichment outputs)
--   2. article_entities  — one row per named entity per article (NER results)
--   3. article_clusters  — one row per story cluster (groups same-event articles)
--   4. Indexes to make Layer 3 (propensity scoring) queries fast
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 1: Add enrichment columns to articles
--
-- WHY ALTER INSTEAD OF RECREATE?
--   The articles table already has data. ALTER TABLE ... ADD COLUMN IF NOT EXISTS
--   is non-destructive — it adds the column with NULL for all existing rows.
--   We never touch existing columns.
--
-- COLUMN DECISIONS:
--   enriched_at       — NULL means "not yet processed by Layer 2". The enrichment
--                        runner uses this as its cursor: WHERE enriched_at IS NULL.
--   word_count        — derived from full_text. Used in propensity scoring
--                        (longer ≠ better, but very short articles are low-signal).
--   reading_time_mins — word_count / 200 (average adult reading speed).
--                        Useful for UX and as a propensity feature.
--   language_detected — the ACTUAL language of the article body, detected by
--                        langdetect. Differs from language_code (feed metadata)
--                        more often than you'd expect.
--   sentiment         — 'positive', 'negative', or 'neutral' (VADER score).
--                        NULL for non-English articles (VADER is English-only).
--   sentiment_score   — raw VADER compound score: -1.0 (most negative) to
--                        +1.0 (most positive). NULL for non-English.
--   category          — article-level topic category. More granular than the
--                        feed-level iab_tier1/iab_tier2 columns.
--                        e.g. 'cricket', 'politics', 'bollywood', 'business'
--   keywords          — jsonb array of top 10 keywords/phrases from the article.
--                        e.g. ["Narendra Modi", "budget 2024", "GST reform"]
--   cluster_id        — soft FK to article_clusters.id. NULL until clustering runs.
--                        Intentionally NOT a hard FK to avoid circular dependency
--                        (article_clusters.canonical_article_id → articles).
--   image_phash       — 64-bit perceptual hash of the top image. Stored as bigint.
--                        Used for image-level dedup and VLM pipeline prep.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE articles ADD COLUMN IF NOT EXISTS enriched_at       timestamptz;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS word_count        integer;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS reading_time_mins float;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS language_detected text;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS sentiment         text;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS sentiment_score   float;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS category          text;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS keywords          jsonb;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_id        uuid;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS image_phash       bigint;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 2: article_entities table
--
-- WHY A SEPARATE TABLE (not jsonb on articles)?
--   Named entities need to be QUERIED, not just stored.
--   "Give me all articles mentioning Narendra Modi from this week" requires
--   an index. You can't efficiently index inside a jsonb array.
--   A separate table with (entity_text, entity_type) columns is fast with
--   a standard B-tree index.
--
-- ENTITY TYPES (spaCy labels):
--   PERSON   — real people: politicians, celebrities, athletes
--   ORG      — organizations: BJP, BCCI, Reliance, RBI
--   GPE      — geo-political entities: India, Delhi, Maharashtra
--   EVENT    — named events: Budget 2024, IPL 2024, G20 Summit
--   PRODUCT  — products, apps, platforms: iPhone, WhatsApp, UPI
--   LAW      — named laws, acts: CAA, RTI, Article 370
--
-- SALIENCE (0.0 to 1.0):
--   How central is this entity to the article?
--   Entity in the title = 0.8–1.0. Mentioned once in body = 0.1.
--   Used to rank entities when showing article tags.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS article_entities (
  id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  article_id   uuid        NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
  entity_text  text        NOT NULL,
  entity_type  text        NOT NULL,
  salience     float       NOT NULL DEFAULT 0.5,
  created_at   timestamptz NOT NULL DEFAULT now()
);

-- Primary lookup: "all entities for article X"
CREATE INDEX IF NOT EXISTS article_entities_article_idx
  ON article_entities (article_id);

-- Cross-article query: "all articles mentioning entity Y of type Z"
-- This is what Layer 3 (propensity scoring) and story clustering query heavily.
CREATE INDEX IF NOT EXISTS article_entities_type_text_idx
  ON article_entities (entity_type, entity_text);

-- High-salience entities only (for fast "top entities" queries)
CREATE INDEX IF NOT EXISTS article_entities_salience_idx
  ON article_entities (entity_text, salience DESC)
  WHERE salience >= 0.5;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 3: article_clusters table
--
-- WHY STORY CLUSTERING?
--   The same PTI wire story gets republished by 50 Indian outlets.
--   Without clustering, your propensity model sees 50 identical signals
--   and thinks the story is 50x more important than it actually is.
--   With clustering, you see: "1 story, covered by 50 outlets" —
--   which IS a strong virality signal, but measured correctly.
--
-- HOW A CLUSTER IS DEFINED:
--   Articles belong to the same cluster if they cover the same real-world event.
--   Detection uses: entity overlap (≥2 shared entities) + title SimHash similarity.
--
-- COLUMN DECISIONS:
--   canonical_article_id — the first/best article in the cluster. Used as the
--                           representative when showing "one story" to the user.
--                           Soft FK (no constraint) to avoid circular dependency.
--   headline             — best headline for this story (from canonical article).
--   outlet_count         — how many DISTINCT domains covered this story.
--                           This is your primary virality signal.
--   article_count        — total articles in the cluster (includes syndication).
--   entity_set           — jsonb list of entity_text values in this cluster.
--                           Used during matching: does new article share entities?
--                           e.g. ["Narendra Modi", "BJP", "Parliament"]
--   top_entities         — jsonb with entity counts for display.
--                           e.g. [{"text": "Modi", "type": "PERSON", "count": 12}]
--   canonical_simhash    — SimHash of canonical article title. Used for fast
--                           title-similarity matching when clustering new articles.
--   gnews_data           — RESERVED for Layer 2h (GNews/SerpAPI enrichment).
--                           NULL until you implement that step.
--                           Will store: outlet list, coverage timeline, etc.
--   first_seen_at        — when the first article in this cluster was published.
--   last_seen_at         — when the most recent article was published.
--                           last_seen_at - first_seen_at = story lifetime.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS article_clusters (
  id                   uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  canonical_article_id uuid,                    -- soft FK → articles.id (uuid)
  headline             text,
  outlet_count         integer     NOT NULL DEFAULT 1,
  article_count        integer     NOT NULL DEFAULT 1,
  entity_set           jsonb       NOT NULL DEFAULT '[]',
  top_entities         jsonb       NOT NULL DEFAULT '[]',
  canonical_simhash    bigint,
  gnews_data           jsonb,                    -- reserved for GNews step
  first_seen_at        timestamptz,
  last_seen_at         timestamptz,
  created_at           timestamptz NOT NULL DEFAULT now(),
  updated_at           timestamptz NOT NULL DEFAULT now()
);

-- "Give me all active clusters from the last 48 hours" — used during clustering
CREATE INDEX IF NOT EXISTS article_clusters_last_seen_idx
  ON article_clusters (last_seen_at DESC NULLS LAST);

-- SimHash lookup during cluster matching
CREATE INDEX IF NOT EXISTS article_clusters_simhash_idx
  ON article_clusters (canonical_simhash)
  WHERE canonical_simhash IS NOT NULL;

-- Story size — "what are the biggest stories right now?"
CREATE INDEX IF NOT EXISTS article_clusters_outlet_count_idx
  ON article_clusters (outlet_count DESC, last_seen_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 4: Indexes on articles for enrichment queries
--
-- These make the enrichment runner's core query fast:
--   SELECT * FROM articles WHERE enriched_at IS NULL ORDER BY published_at DESC
-- ─────────────────────────────────────────────────────────────────────────────

-- Primary enrichment cursor: unenriched articles, newest first
CREATE INDEX IF NOT EXISTS articles_enriched_at_idx
  ON articles (enriched_at NULLS FIRST, published_at DESC NULLS LAST)
  WHERE enriched_at IS NULL;

-- Category queries: "show me recent politics articles"
CREATE INDEX IF NOT EXISTS articles_category_published_idx
  ON articles (category, published_at DESC NULLS LAST)
  WHERE category IS NOT NULL;

-- Cluster membership: "all articles in cluster X"
CREATE INDEX IF NOT EXISTS articles_cluster_id_idx
  ON articles (cluster_id)
  WHERE cluster_id IS NOT NULL;

-- Image hash lookup for image-level dedup
CREATE INDEX IF NOT EXISTS articles_image_phash_idx
  ON articles (image_phash)
  WHERE image_phash IS NOT NULL;

-- Sentiment analysis queries
CREATE INDEX IF NOT EXISTS articles_sentiment_published_idx
  ON articles (sentiment, published_at DESC NULLS LAST)
  WHERE sentiment IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 5: Update articles_ist view to include new columns
--
-- The view was created earlier. We recreate it to expose the new enrichment
-- columns with IST timestamps (useful for the Indian market app layer).
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW articles_ist AS
SELECT
    id,
    feed_id,
    url,
    title,
    description,
    full_text,
    (published_at AT TIME ZONE 'Asia/Kolkata') AS published_at_ist,
    (crawled_at  AT TIME ZONE 'Asia/Kolkata')  AS crawled_at_ist,
    (created_at  AT TIME ZONE 'Asia/Kolkata')  AS created_at_ist,
    (enriched_at AT TIME ZONE 'Asia/Kolkata')  AS enriched_at_ist,
    is_duplicate,
    is_crawled,
    language,
    language_code,
    language_detected,
    country_code,
    domain,
    publisher_name,
    iab_tier1,
    iab_tier2,
    category,
    keywords,
    word_count,
    reading_time_mins,
    sentiment,
    sentiment_score,
    cluster_id,
    image_phash,
    top_image_url,
    propensity_score,
    story_id
FROM articles;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 6: Monitoring view for enrichment health
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW enrichment_health AS
SELECT
    count(*)                                                           AS total_articles,
    count(*) FILTER (WHERE enriched_at IS NOT NULL)                    AS enriched,
    count(*) FILTER (WHERE enriched_at IS NULL)                        AS pending,
    round(
        count(*) FILTER (WHERE enriched_at IS NOT NULL)::numeric /
        nullif(count(*), 0) * 100
    )::int                                                             AS enriched_pct,
    count(*) FILTER (WHERE category IS NOT NULL)                       AS has_category,
    count(*) FILTER (WHERE cluster_id IS NOT NULL)                     AS clustered,
    count(*) FILTER (WHERE language_detected IS NOT NULL)              AS language_detected,
    count(*) FILTER (WHERE image_phash IS NOT NULL)                    AS has_image_hash,
    (SELECT count(*) FROM article_clusters)                            AS total_clusters,
    (SELECT count(*) FROM article_entities)                            AS total_entities
FROM articles;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 7: Verify — run this to confirm everything was created
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    table_name,
    pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('articles', 'article_entities', 'article_clusters', 'pipeline_runs', 'feeds')
ORDER BY table_name;
