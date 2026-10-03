-- =============================================================================
-- docs/enrichment_outputs_migration.sql
-- Columns for the NewsData.io-style enrichment outputs added on 2026-04-07
-- (commit 39b0d5c) — sentiment_stats, ai_summary, ai_tag, ai_region, ai_org.
--
-- WHY THIS FILE EXISTS:
--   enrichment/runner.py writes these five columns on every article, but no
--   migration ever created them. On a database built from docs/ alone every
--   save_article_enrichment() call failed with PGRST204 ("column not found"),
--   enriched_at was never set, and the same articles were re-processed forever.
--   Production was patched by hand; this file makes the schema reproducible.
--
-- Idempotent — ADD COLUMN IF NOT EXISTS leaves existing columns (and their
-- types) untouched, so it is safe on the live database.
-- =============================================================================

-- {positive: 91.2, neutral: 6.1, negative: 2.7}  (percentages, NewsData.io format)
ALTER TABLE articles ADD COLUMN IF NOT EXISTS sentiment_stats jsonb;

-- Extractive summary: description if ≥150 chars, else first 3 body sentences.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS ai_summary text;

-- Up to 5 fine-grained zero-shot topic tags, e.g. ["elections", "government"].
ALTER TABLE articles ADD COLUMN IF NOT EXISTS ai_tag jsonb;

-- Top GPE / ORG entities (lower-cased) derived from NER, max 5 each.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS ai_region jsonb;
ALTER TABLE articles ADD COLUMN IF NOT EXISTS ai_org jsonb;
