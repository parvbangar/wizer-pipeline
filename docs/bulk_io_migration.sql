-- =============================================================================
-- docs/bulk_io_migration.sql
-- Bulk I/O: the database receives results, it no longer does the work.
--
-- Run AFTER docs/enrichment_queue_v2_migration.sql. Idempotent.
--
-- WHY:
--   On the Supabase Micro instance (burstable CPU and disk I/O) the per-row
--   path exhausted the credits: a feed-state read + write per polled feed, an
--   exact cosine scan inside wizer_assign_cluster per article, and several
--   calls per enriched article. With these functions:
--     - story clustering runs in memory on the runner
--       (enrichment/memory_clustering.py) and writes its result here in bulk;
--     - enrichment results are saved a batch at a time;
--     - feed poll outcomes are recorded once per run, arithmetic done in SQL.
--   Each function takes a JSON array and does all its work in one statement
--   per table, so one round trip handles hundreds of rows.
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. wizer_db_now — the database clock (cluster-state sync bookmarks)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_db_now()
RETURNS timestamptz
LANGUAGE sql
STABLE
AS $$ SELECT now(); $$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. wizer_cluster_state — clusters for the in-memory clusterer, keyset-paged
--
--   p_changed_since NULL  → FULL load: active clusters last seen since
--                           p_window_since (cold start, cache miss)
--   p_changed_since set   → DELTA: every cluster of the model changed after
--                           it, any status (merged tombstones tell the
--                           runner to drop them)
--
-- Vectors are returned as text; the runner parses them once per load.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_cluster_state(
  p_model          text,
  p_changed_since  timestamptz,
  p_window_since   timestamptz,
  p_limit          integer DEFAULT 200,
  p_after_id       uuid    DEFAULT '00000000-0000-0000-0000-000000000000'
)
RETURNS TABLE (
  id                         uuid,
  status                     text,
  embedding_model            text,
  centroid_sum               text,
  anchor                     text,
  representative             text,
  representative_article_id  bigint,
  headline                   text,
  article_count              integer,
  outlet_set                 jsonb,
  language_set               jsonb,
  first_seen_at              timestamptz,
  last_seen_at               timestamptz,
  updated_at                 timestamptz
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT c.id, c.status, c.embedding_model,
         c.centroid_sum::text, c.anchor::text, c.representative::text,
         c.representative_article_id, c.headline, c.article_count,
         c.outlet_set, c.language_set, c.first_seen_at, c.last_seen_at, c.updated_at
    FROM article_clusters AS c
   WHERE c.embedding_model = p_model
     AND c.id > p_after_id
     AND CASE WHEN p_changed_since IS NULL
              THEN c.status = 'active' AND c.last_seen_at >= p_window_since
              ELSE c.updated_at > p_changed_since END
   ORDER BY c.id
   LIMIT greatest(p_limit, 1);
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2b. wizer_fetch_unclustered_ingested — the in-memory clusterer's work list
--
-- Every article without a cluster INGESTED since p_since, oldest first,
-- keyset-paged on (ingestion time, id). Keyed on ingestion time, not
-- published_at (wizer_fetch_unclustered): an undated or stale-dated article
-- must still get its story. No body: it is not stored in Postgres any more —
-- the job overlays bodies from the ingest hand-off where it has them.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS articles_unclustered_ingested_idx
  ON articles ((coalesce(crawled_at, created_at)), id)
  WHERE cluster_id IS NULL AND title IS NOT NULL;

CREATE OR REPLACE FUNCTION wizer_fetch_unclustered_ingested(
  p_since     timestamptz,
  p_limit     integer     DEFAULT 500,
  p_after_ts  timestamptz DEFAULT '-infinity',
  p_after_id  bigint      DEFAULT 0
)
RETURNS TABLE (
  id            bigint,
  title         text,
  description   text,
  domain        text,
  language_code text,
  published_at  timestamptz,
  ingested_at   timestamptz
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT a.id, a.title, a.description, a.domain, a.language_code::text,
         a.published_at, coalesce(a.crawled_at, a.created_at)
    FROM articles AS a
   WHERE a.cluster_id IS NULL
     AND a.title IS NOT NULL
     AND coalesce(a.crawled_at, a.created_at) >= p_since
     AND (coalesce(a.crawled_at, a.created_at), a.id) > (p_after_ts, p_after_id)
   ORDER BY coalesce(a.crawled_at, a.created_at), a.id
   LIMIT greatest(p_limit, 1);
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. wizer_apply_cluster_changes — write the in-memory clusterer's result
--
--   p_clusters  [{id, is_new, centroid_sum, article_count, outlet_set,
--                 language_set, first_seen_at, last_seen_at, headline,
--                 representative_article_id, canonical_article_id,
--                 anchor?, representative?}]
--               The runner owns these columns (single writer — see
--               memory_clustering.py); the centroid is derived from the sum.
--               anchor / representative are sent only when they differ from
--               normalise(sum) (a singleton's are its sum). top_entities / entity_set /
--               image_hashes are NOT touched: enrichment adds those later.
--   p_articles  [{id, cluster_id, action, similarity}] — only articles that
--               have no cluster yet are stamped (idempotent on retry).
--
-- Returns the DB clock of this transaction: every row written here carries
-- updated_at = that time, so the next delta sync (updated_at > synced_at)
-- does not re-download the runner's own writes.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_apply_cluster_changes(
  p_model     text,
  p_clusters  jsonb,
  p_articles  jsonb
)
RETURNS TABLE (clusters_written integer, articles_written integer, synced_at timestamptz)
LANGUAGE plpgsql
VOLATILE
SET search_path = public, extensions
AS $$
DECLARE
  v_clusters integer := 0;
  v_articles integer := 0;
BEGIN
  WITH src AS (
    SELECT (c ->> 'id')::uuid                          AS id,
           coalesce((c ->> 'is_new')::boolean, false)  AS is_new,
           (c ->> 'centroid_sum')::vector              AS centroid_sum,
           (c ->> 'representative')::vector            AS representative,
           (c ->> 'anchor')::vector                    AS anchor,
           (c ->> 'article_count')::integer            AS article_count,
           coalesce(c -> 'outlet_set', '[]'::jsonb)    AS outlet_set,
           coalesce(c -> 'language_set', '[]'::jsonb)  AS language_set,
           (c ->> 'first_seen_at')::timestamptz        AS first_seen_at,
           (c ->> 'last_seen_at')::timestamptz         AS last_seen_at,
           left(c ->> 'headline', 500)                 AS headline,
           (c ->> 'representative_article_id')::bigint AS rep_id,
           (c ->> 'canonical_article_id')::bigint      AS seed_id
      FROM jsonb_array_elements(coalesce(p_clusters, '[]'::jsonb)) AS c
  ), inserted AS (
    -- New clusters. A retry of an already-written batch conflicts and is
    -- skipped: the first write carried the same values.
    INSERT INTO article_clusters (
      id, canonical_article_id, representative_article_id, headline,
      centroid_sum, centroid, anchor, representative, embedding_model,
      article_count, outlet_count, outlet_set, entity_set, top_entities,
      image_hashes, language_set, first_seen_at, last_seen_at, status, updated_at
    )
    SELECT s.id, coalesce(s.seed_id, s.rep_id), s.rep_id, s.headline,
           s.centroid_sum, l2_normalize(s.centroid_sum)::halfvec,
           coalesce(s.anchor, l2_normalize(s.centroid_sum))::halfvec,
           coalesce(s.representative, l2_normalize(s.centroid_sum))::halfvec,
           p_model, s.article_count, jsonb_array_length(s.outlet_set), s.outlet_set,
           '[]'::jsonb, '[]'::jsonb, '[]'::jsonb, s.language_set,
           s.first_seen_at, s.last_seen_at, 'active', now()
      FROM src AS s
     WHERE s.is_new
    ON CONFLICT (id) DO NOTHING
    RETURNING 1
  ), updated AS (
    -- Clusters that grew. Only active ones: a merged tombstone is never revived.
    UPDATE article_clusters AS t
       SET centroid_sum   = s.centroid_sum,
           centroid       = l2_normalize(s.centroid_sum)::halfvec,
           article_count  = s.article_count,
           outlet_set     = s.outlet_set,
           outlet_count   = jsonb_array_length(s.outlet_set),
           language_set   = s.language_set,
           first_seen_at  = s.first_seen_at,
           last_seen_at   = s.last_seen_at,
           headline       = coalesce(s.headline, t.headline),
           representative_article_id = s.rep_id,
           representative = coalesce(s.representative::halfvec, t.representative),
           updated_at     = now()
      FROM src AS s
     WHERE t.id = s.id
       AND NOT s.is_new
       AND t.status = 'active'
    RETURNING 1
  )
  SELECT (SELECT count(*) FROM inserted) + (SELECT count(*) FROM updated) INTO v_clusters;

  WITH r AS (
    SELECT (x ->> 'id')::bigint         AS id,
           (x ->> 'cluster_id')::uuid   AS cluster_id,
           x ->> 'action'               AS action,
           (x ->> 'similarity')::real   AS similarity
      FROM jsonb_array_elements(coalesce(p_articles, '[]'::jsonb)) AS x
  ), stamped AS (
    UPDATE articles AS a
       SET cluster_id = r.cluster_id, cluster_similarity = r.similarity,
           cluster_assignment = r.action, clustered_at = now()
      FROM r
     WHERE a.id = r.id
       AND a.cluster_id IS NULL
    RETURNING 1
  )
  SELECT count(*) INTO v_articles FROM stamped;

  RETURN QUERY SELECT v_clusters, v_articles, now();
END;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. wizer_save_enrichment_batch — enrichment results for many articles
--
--   p_items [{id, update: {<enrichment columns>}, crawl: {<crawl columns>},
--             entities: [{entity_text, entity_type, salience}],
--             cluster_entities: [{text, type}], entity_keys: [text]}]
--
--   - crawl (optional): what the deferred page crawl found
--     (pipeline.crawler.crawl_record). Only keys present overwrite: the
--     page's description / image / author / og_tags / crawl status; a title
--     only if the row has none; a publish date only if the row has none.
--     full_text is NOT stored here — bodies go to Storage (body_store.py);
--   - articles: the enrichment columns are set from `update` (a missing key
--     sets NULL — a step that failed produced nothing), enrich_error is
--     cleared and enriched_at set LAST, in the same statement;
--   - article_entities: replaced for these articles;
--   - the article's story cluster gets its entities (top_entities tally,
--     entity_set). updated_at is NOT bumped: these columns are not part of
--     the in-memory clusterer's state, so its delta sync has nothing to fetch.
--
-- Returns the number of articles updated.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_save_enrichment_batch(p_items jsonb)
RETURNS integer
LANGUAGE plpgsql
VOLATILE
SET search_path = public, extensions
AS $$
DECLARE
  v_updated integer := 0;
BEGIN
  WITH it AS (
    SELECT (x ->> 'id')::bigint AS id, coalesce(x -> 'update', '{}'::jsonb) AS u,
           coalesce(x -> 'crawl', '{}'::jsonb) AS c
      FROM jsonb_array_elements(coalesce(p_items, '[]'::jsonb)) AS x
  ), done AS (
    UPDATE articles AS a
       SET word_count        = (it.u ->> 'word_count')::integer,
           reading_time_mins = (it.u ->> 'reading_time_mins')::double precision,
           language_detected = it.u ->> 'language_detected',
           sentiment         = it.u ->> 'sentiment',
           sentiment_score   = (it.u ->> 'sentiment_score')::double precision,
           sentiment_stats   = it.u -> 'sentiment_stats',
           category          = it.u ->> 'category',
           ai_tag            = it.u -> 'ai_tag',
           ai_summary        = it.u ->> 'ai_summary',
           ai_region         = it.u -> 'ai_region',
           ai_org            = it.u -> 'ai_org',
           keywords          = it.u -> 'keywords',
           image_phash       = (it.u ->> 'image_phash')::bigint,
           title             = CASE WHEN coalesce(a.title, '') = '' AND it.c ? 'title'
                                    THEN left(it.c ->> 'title', 1000) ELSE a.title END,
           title_simhash     = CASE WHEN coalesce(a.title, '') = '' AND it.c ? 'title_simhash'
                                    THEN (it.c ->> 'title_simhash')::bigint ELSE a.title_simhash END,
           description       = CASE WHEN it.c ? 'description' THEN left(it.c ->> 'description', 2000)
                                    ELSE a.description END,
           top_image_url     = CASE WHEN it.c ? 'top_image_url' THEN left(it.c ->> 'top_image_url', 500)
                                    ELSE a.top_image_url END,
           author            = CASE WHEN it.c ? 'author' THEN left(it.c ->> 'author', 300) ELSE a.author END,
           published_at      = coalesce(a.published_at, (it.c ->> 'published_at')::timestamptz),
           og_tags           = CASE WHEN it.c ? 'og_tags' THEN it.c -> 'og_tags' ELSE a.og_tags END,
           is_crawled        = coalesce((it.c ->> 'is_crawled')::boolean, a.is_crawled),
           crawl_strategy    = coalesce(it.c ->> 'crawl_strategy', a.crawl_strategy),
           enrich_error      = NULL,
           enriched_at       = now()
      FROM it
     WHERE a.id = it.id
    RETURNING a.id
  )
  SELECT count(*) INTO v_updated FROM done;

  DELETE FROM article_entities AS e
   USING jsonb_array_elements(coalesce(p_items, '[]'::jsonb)) AS x
   WHERE e.article_id = (x ->> 'id')::bigint;

  INSERT INTO article_entities (article_id, entity_text, entity_type, salience)
  SELECT (x ->> 'id')::bigint, e ->> 'entity_text', e ->> 'entity_type', (e ->> 'salience')::float8
    FROM jsonb_array_elements(coalesce(p_items, '[]'::jsonb)) AS x,
         jsonb_array_elements(coalesce(x -> 'entities', '[]'::jsonb)) AS e
   WHERE EXISTS (SELECT 1 FROM articles AS a WHERE a.id = (x ->> 'id')::bigint);

  -- Cluster entity tallies: one update per cluster, all its new members at once.
  WITH per_article AS (
    SELECT a.cluster_id,
           coalesce(x -> 'cluster_entities', '[]'::jsonb) AS ents,
           coalesce(x -> 'entity_keys', '[]'::jsonb)      AS keys
      FROM jsonb_array_elements(coalesce(p_items, '[]'::jsonb)) AS x
      JOIN articles AS a ON a.id = (x ->> 'id')::bigint
     WHERE a.cluster_id IS NOT NULL
  ), per_cluster AS (
    SELECT cluster_id,
           coalesce(jsonb_agg(e) FILTER (WHERE e IS NOT NULL), '[]'::jsonb) AS ents,
           coalesce((SELECT jsonb_agg(DISTINCT k)
                       FROM per_article AS p2, jsonb_array_elements_text(p2.keys) AS k
                      WHERE p2.cluster_id = p.cluster_id), '[]'::jsonb)     AS keys
      FROM per_article AS p
      LEFT JOIN LATERAL jsonb_array_elements(p.ents) AS e ON true
     GROUP BY cluster_id
  )
  UPDATE article_clusters AS c
     SET top_entities = wizer_merge_top_entities(c.top_entities, pc.ents, 30),
         entity_set   = wizer_jsonb_text_union(c.entity_set, pc.keys, 200)
    FROM per_cluster AS pc
   WHERE c.id = pc.cluster_id
     AND c.status = 'active';

  RETURN v_updated;
END;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. wizer_record_feed_polls — every feed polled in a run, one call
--
--   p_items [{id, success, new_articles, reactivate, dormant}]
--     success       fetch worked (even with 0 new articles)
--     new_articles  rows inserted from this feed
--     reactivate    a dormant re-check that found articles → active again
--     dormant       the dormancy check fired → inactive, 'dormant'
--
-- Same rules as pipeline/db.update_feed_after_poll + mark_feed_dormant, but
-- the counters are incremented in SQL (no read-before-write, no lost update).
-- Returns the number of feeds updated.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_record_feed_polls(p_items jsonb, p_max_errors integer DEFAULT 5)
RETURNS integer
LANGUAGE sql
VOLATILE
SET search_path = public, extensions
AS $$
  WITH it AS (
    -- feeds.id is bigint in production and uuid in docs/migration.sql:
    -- compare as text so the function works on both.
    SELECT x ->> 'id'                                   AS id,
           coalesce((x ->> 'success')::boolean, false)  AS success,
           coalesce((x ->> 'new_articles')::integer, 0) AS new_articles,
           coalesce((x ->> 'reactivate')::boolean, false) AS reactivate,
           coalesce((x ->> 'dormant')::boolean, false)  AS dormant
      FROM jsonb_array_elements(coalesce(p_items, '[]'::jsonb)) AS x
  ), done AS (
    UPDATE feeds AS f
       SET last_polled_at      = now(),
           last_success_at     = CASE WHEN it.success THEN now() ELSE f.last_success_at END,
           fail_count          = CASE WHEN it.success THEN 0 ELSE f.fail_count + 1 END,
           articles_found      = f.articles_found + it.new_articles,
           last_new_article_at = CASE WHEN it.success AND it.new_articles > 0
                                      THEN now() ELSE f.last_new_article_at END,
           is_active           = CASE WHEN it.success AND it.reactivate THEN true
                                      WHEN NOT it.success AND f.fail_count + 1 >= p_max_errors THEN false
                                      WHEN it.success AND it.dormant THEN false
                                      ELSE f.is_active END,
           disabled_reason     = CASE WHEN it.success AND it.reactivate THEN NULL
                                      WHEN NOT it.success AND f.fail_count + 1 >= p_max_errors THEN 'errors'
                                      WHEN it.success AND it.dormant THEN 'dormant'
                                      ELSE f.disabled_reason END
      FROM it
     WHERE f.id::text = it.id
    RETURNING 1
  )
  SELECT count(*)::integer FROM done;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. Permissions — service role only
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
DECLARE
  fn text;
BEGIN
  FOREACH fn IN ARRAY ARRAY[
    'wizer_db_now()',
    'wizer_cluster_state(text, timestamptz, timestamptz, integer, uuid)',
    'wizer_fetch_unclustered_ingested(timestamptz, integer, timestamptz, bigint)',
    'wizer_apply_cluster_changes(text, jsonb, jsonb)',
    'wizer_save_enrichment_batch(jsonb)',
    'wizer_record_feed_polls(jsonb, integer)'
  ] LOOP
    EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', fn);
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
      EXECUTE format('REVOKE ALL ON FUNCTION %s FROM anon, authenticated', fn);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
      EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', fn);
    END IF;
  END LOOP;
END $$;
