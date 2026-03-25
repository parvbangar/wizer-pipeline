-- =============================================================================
-- docs/migration.sql
-- Run this ONCE in Supabase → SQL Editor.
-- 
-- YOUR FEEDS TABLE ALREADY EXISTS — this script only ADDS what's missing.
-- All statements use IF NOT EXISTS — safe to run multiple times.
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 1: Add the pipeline_runs table
-- This stores one row per pipeline run for monitoring.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_runs (
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  cadence          text,                        -- 'breaking_news', 'multiple_daily', 'daily', etc.
  feeds_attempted  integer     NOT NULL DEFAULT 0,
  feeds_skipped    integer     NOT NULL DEFAULT 0,
  new_articles     integer     NOT NULL DEFAULT 0,
  near_duplicates  integer     NOT NULL DEFAULT 0,
  exact_duplicates integer     NOT NULL DEFAULT 0,
  errors           integer     NOT NULL DEFAULT 0,
  duration_s       float,
  run_at           timestamptz NOT NULL DEFAULT now(),
  dry_run          boolean     NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS pipeline_runs_run_at_idx
  ON pipeline_runs (run_at DESC);


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 2: Create the articles table with YOUR exact column schema
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS articles (
  -- Core identity
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  feed_id          uuid        REFERENCES feeds(id) ON DELETE CASCADE,
  url              text        NOT NULL,
  url_hash         bigint      NOT NULL,         -- MurmurHash3 of normalised URL
                                                 -- UNIQUE constraint = dedup key

  -- Content
  title            text,
  title_simhash    bigint,                       -- SimHash of title for near-dedup
  description      text,
  full_text        text,
  top_image_url    text,
  author           text,

  -- Dates
  published_at     timestamptz,
  crawled_at       timestamptz NOT NULL DEFAULT now(),

  -- Classification
  language         text,                         -- e.g. "English", "Hindi"
  country_code     text,                         -- e.g. "IND", "USA"
  og_tags          jsonb,                        -- full Open Graph tag dump

  -- Status flags
  is_crawled       boolean     NOT NULL DEFAULT false,   -- was full text extracted?
  is_duplicate     boolean     NOT NULL DEFAULT false,   -- near-duplicate of another?

  -- Future fields (leave empty for now)
  story_id         uuid,                         -- for story clustering (Layer 3)
  propensity_score float,                        -- for virality scoring (Layer 6)

  -- Timestamps
  created_at       timestamptz NOT NULL DEFAULT now(),

  -- Denormalised from feeds (for easier querying without joins)
  feed_url         text,
  domain           text,
  publisher_name   text,
  iab_tier1        text,
  iab_tier2        text,
  language_code    text                          -- e.g. "en", "hi", "ta"
);


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 3: Indexes on articles
--
-- WHY THESE SPECIFIC INDEXES?
--   url_hash: PRIMARY dedup check — all inserts check this. MUST be unique.
--   feed_id + published_at: "show me latest articles from this feed"
--   language_code + published_at: "show me latest Hindi articles"
--   is_duplicate: "show me only original articles (not near-dups)"
--   is_crawled: "show me articles we still need to crawl"
--   title_simhash: future use for near-dedup queries
-- ─────────────────────────────────────────────────────────────────────────────

CREATE UNIQUE INDEX IF NOT EXISTS articles_url_hash_uidx
  ON articles (url_hash);

CREATE INDEX IF NOT EXISTS articles_feed_published_idx
  ON articles (feed_id, published_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS articles_lang_published_idx
  ON articles (language_code, published_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS articles_domain_published_idx
  ON articles (domain, published_at DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS articles_is_duplicate_idx
  ON articles (is_duplicate) WHERE is_duplicate = false;

CREATE INDEX IF NOT EXISTS articles_is_crawled_idx
  ON articles (is_crawled) WHERE is_crawled = false;

CREATE INDEX IF NOT EXISTS articles_iab_published_idx
  ON articles (iab_tier1, published_at DESC NULLS LAST);

-- Full-text search on title (useful for your app's search feature later)
CREATE INDEX IF NOT EXISTS articles_title_fts_idx
  ON articles USING gin(to_tsvector('english', coalesce(title, '')));

-- SimHash index for near-duplicate lookups
CREATE INDEX IF NOT EXISTS articles_title_simhash_idx
  ON articles (title_simhash) WHERE title_simhash IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 4: Indexes on your existing feeds table
-- These make the pipeline's "get due feeds" query much faster.
-- ─────────────────────────────────────────────────────────────────────────────

-- "Get all active feeds for a cadence, ordered by priority"
CREATE INDEX IF NOT EXISTS feeds_active_cadence_priority_idx
  ON feeds (is_active, update_cadence, priority_score DESC NULLS LAST)
  WHERE is_active = true;

-- Kept for backward compatibility / monitoring views
CREATE INDEX IF NOT EXISTS feeds_active_tier_priority_idx
  ON feeds (is_active, validation_tier, priority_score DESC NULLS LAST)
  WHERE is_active = true;

-- "When is this feed due next?" — for scheduling
CREATE INDEX IF NOT EXISTS feeds_active_last_polled_idx
  ON feeds (is_active, last_polled_at NULLS FIRST)
  WHERE is_active = true;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 5: Row Level Security (RLS)
--
-- WHAT IS RLS?
--   Controls who can read/write which rows.
--   Service key (used by the pipeline) = full access.
--   Anon key (used by your app/users) = read non-paywalled articles only.
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE articles       ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs  ENABLE ROW LEVEL SECURITY;

-- Anon users can only read non-duplicate, non-paywalled articles
-- (is_duplicate check: don't show duplicate wire stories to end users)
DROP POLICY IF EXISTS "anon_read_articles" ON articles;
CREATE POLICY "anon_read_articles" ON articles
  FOR SELECT TO anon
  USING (
    is_duplicate = false
    -- Add a join to feeds.has_paywall here when you build the app layer
  );

-- Anon users can see pipeline run stats (useful for status pages)
DROP POLICY IF EXISTS "anon_read_runs" ON pipeline_runs;
CREATE POLICY "anon_read_runs" ON pipeline_runs
  FOR SELECT TO anon USING (true);


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 6: Monitoring views
-- Query these in Supabase Table Editor or SQL Editor to monitor health.
-- ─────────────────────────────────────────────────────────────────────────────

-- Feed health dashboard
CREATE OR REPLACE VIEW feed_health AS
SELECT
  f.id,
  f.domain,
  f.publisher_name,
  f.validation_tier                                           AS tier,
  f.is_active,
  f.fail_count,
  f.language_code,
  f.country_code,
  f.has_paywall,
  f.last_polled_at,
  f.last_success_at,
  f.articles_found,
  f.priority_score,
  f.freshness_score,
  round(
    EXTRACT(EPOCH FROM (now() - f.last_success_at)) / 86400
  )::int                                                      AS days_since_new_article,
  count(a.id)                                                 AS total_articles_in_db
FROM feeds f
LEFT JOIN articles a ON a.feed_id = f.id
GROUP BY f.id
ORDER BY f.priority_score DESC NULLS LAST;

-- Feeds approaching dormancy (>20 days without new articles — warning zone)
CREATE OR REPLACE VIEW dormancy_watch AS
SELECT
  domain,
  publisher_name,
  validation_tier                                             AS tier,
  language_code,
  is_active,
  fail_count,
  last_success_at,
  round(
    EXTRACT(EPOCH FROM (now() - last_success_at)) / 86400
  )::int                                                      AS days_stale
FROM feeds
WHERE
  is_active = true
  AND last_success_at IS NOT NULL
  AND (now() - last_success_at) > interval '20 days'
ORDER BY last_success_at ASC;

-- Article quality stats
CREATE OR REPLACE VIEW article_stats AS
SELECT
  domain,
  publisher_name,
  count(*)                                                    AS total,
  count(*) FILTER (WHERE is_crawled = true)                   AS crawled,
  count(*) FILTER (WHERE is_duplicate = true)                 AS near_duplicates,
  count(*) FILTER (WHERE published_at > now() - interval '24 hours') AS last_24h,
  round(
    count(*) FILTER (WHERE is_crawled = true)::numeric /
    nullif(count(*), 0) * 100
  )::int                                                      AS crawl_rate_pct
FROM articles
GROUP BY domain, publisher_name
ORDER BY total DESC;

-- Recent pipeline runs
CREATE OR REPLACE VIEW recent_runs AS
SELECT * FROM pipeline_runs ORDER BY run_at DESC LIMIT 100;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 7: Migrate pipeline_runs.tier → cadence
-- The pipeline now groups by update_cadence instead of validation_tier.
-- This renames the column safely — only runs if 'tier' still exists.
-- ─────────────────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'pipeline_runs' AND column_name = 'tier'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'pipeline_runs' AND column_name = 'cadence'
    ) THEN
        ALTER TABLE pipeline_runs RENAME COLUMN tier TO cadence;
    END IF;
END $$;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 8: Verify setup — run this to confirm everything was created
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
  table_name,
  pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) AS size
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('feeds', 'articles', 'pipeline_runs')
ORDER BY table_name;
