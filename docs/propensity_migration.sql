-- Propensity score migration
-- Run once in Supabase SQL Editor.
--
-- WHAT THIS DOES:
--   Adds propensity_score float column to articles table.
--   Propensity score (0.0–1.0) is written by the enrichment pipeline at
--   enrichment time and updated weekly by train_propensity.py.
--
-- WHY propensity_score?
--   The app layer needs a single number to rank articles in the feed,
--   trigger push notifications (score > 0.70), and gate ad placement.
--   Computed from: outlet_count (virality), category, entity salience, recency.
--   Phase 2: LightGBM model trained weekly on outlet_count as self-supervised label.

ALTER TABLE articles
  ADD COLUMN IF NOT EXISTS propensity_score float;

-- Index for fast top-N feed queries: "show me the 50 highest-propensity
-- articles published in the last 24 hours"
CREATE INDEX IF NOT EXISTS idx_articles_propensity
  ON articles (propensity_score DESC)
  WHERE propensity_score IS NOT NULL;

-- Composite index for the most common app query pattern:
--   WHERE enriched_at IS NOT NULL AND published_at > now() - interval '24 hours'
--   ORDER BY propensity_score DESC
CREATE INDEX IF NOT EXISTS idx_articles_propensity_recency
  ON articles (published_at DESC, propensity_score DESC)
  WHERE enriched_at IS NOT NULL;
