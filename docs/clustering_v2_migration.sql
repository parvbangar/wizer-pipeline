-- =============================================================================
-- docs/clustering_v2_migration.sql
-- Story clustering v2 — atomic, concurrency-safe, drift-resistant.
--
-- Run AFTER docs/enrichment_migration.sql and docs/enrichment_queue_migration.sql.
-- Idempotent: safe to run repeatedly. Requires pgvector >= 0.7.0 (halfvec,
-- l2_normalize) — every Supabase project created since mid-2024 has 0.8.x;
-- older ones: `ALTER EXTENSION vector UPDATE;`.
--
-- Full design write-up: docs/CLUSTERING.md. Summary of what v1 got wrong and
-- what this file does about it:
--
--   v1 (removed in 374dfe0)                 v2 (this file)
--   ──────────────────────────────────────  ──────────────────────────────────
--   find in SQL, decide + write in Python   find + decide + write in ONE
--   → two shards / overlapping runs lose    transaction under an advisory lock
--     counter updates and create twin         (wizer_assign_cluster)
--     clusters for the same story
--   running mean re-normalised each step    exact running SUM of member vectors
--     → not the true mean, drifts           (cosine is scale-invariant, so the
--                                             sum IS the mean for search)
--   centroid-similarity decisions chain:    AVERAGE-LINK decisions (mean cosine
--     a big cluster's generic centroid        to every member, exact from the
--     absorbs neighbouring stories, and       sum) + ANCHOR guard: every member
--     the centroid wanders off its story      must resemble the seed article
--   last_seen_at overwritten with the       first/last_seen use LEAST/GREATEST;
--     article's time; enrichment runs         window is relative to the
--     newest-first, so clusters moved         ARTICLE's published_at, not now(),
--     backwards and fell out of the window    so backfills cluster correctly
--   CLUSTER_WINDOW_HOURS=0 (search all       bounded window: max gap since the
--     time) → a story could absorb          cluster's last article + max total
--     articles months later                 span, so recurring daily stories
--                                             ("Sensex closes higher") don't fuse
--   HNSW + WHERE filter → silently missed   exact search over the (small) time
--     neighbours (post-filter recall loss)    window; HNSW kept only for the
--                                             merge-candidate sweep
--   no second chance for borderline pairs   gray zone: slightly-below-threshold
--                                             matches join if they share salient
--                                             entities or a wire photo
--   duplicate clusters never reconciled     wizer_find_cluster_merge_candidates
--                                             + wizer_merge_clusters (tombstones)
--   headline = longest title                representative = member closest to
--                                             the current centroid (exact)
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 0. pgvector — create if missing (Supabase keeps extensions in `extensions`),
--    then refuse to continue on versions without halfvec / l2_normalize.
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'extensions') THEN
      EXECUTE 'CREATE EXTENSION vector WITH SCHEMA extensions';
    ELSE
      EXECUTE 'CREATE EXTENSION vector';
    END IF;
  END IF;
END $$;

DO $$
DECLARE
  v text;
BEGIN
  SELECT extversion INTO v FROM pg_extension WHERE extname = 'vector';
  IF string_to_array(split_part(v, '-', 1), '.')::int[] < ARRAY[0, 7, 0] THEN
    RAISE EXCEPTION
      'pgvector % is too old: clustering v2 needs >= 0.7.0 (halfvec, l2_normalize). '
      'Run: ALTER EXTENSION vector UPDATE;', v;
  END IF;
END $$;

SET search_path = public, extensions;


-- ─────────────────────────────────────────────────────────────────────────────
-- 1. article_clusters — v2 columns
--
-- Vector storage choices:
--   centroid_sum  vector(768)   exact float32 SUM of member embeddings. Read and
--                               written for ONE row per assignment, so its size
--                               (3 KB, TOASTed) does not matter.
--   centroid      halfvec(768)  L2-normalised centroid used for SEARCH. 1.5 KB
--                               fits in the heap page (no TOAST detour), which
--                               keeps the exact window scan fast.
--   anchor        halfvec(768)  the seed article's embedding (drift guard).
--   representative halfvec(768) embedding of the current representative
--                               article, so "is the new article more central?"
--                               is an exact comparison, not a stale one.
-- halfvec keeps ~3 significant digits — far below the gap between "same story"
-- (>0.8) and "different story" (<0.7) cosine similarities.
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS centroid_sum              vector(768);
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS centroid                  halfvec(768);
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS anchor                    halfvec(768);
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS representative            halfvec(768);
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS representative_article_id bigint;
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS embedding_model           text;
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS status                    text NOT NULL DEFAULT 'active';
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS merged_into               uuid;
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS outlet_set                jsonb NOT NULL DEFAULT '[]';
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS image_hashes              jsonb NOT NULL DEFAULT '[]';
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS language_set              jsonb NOT NULL DEFAULT '[]';
ALTER TABLE article_clusters ADD COLUMN IF NOT EXISTS gray_join_count           integer NOT NULL DEFAULT 0;

-- status: 'active' (searchable) | 'merged' (tombstone → merged_into)
--         | 'legacy' (v1 cluster without a v2 centroid — never searched)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'article_clusters_status_chk') THEN
    ALTER TABLE article_clusters
      ADD CONSTRAINT article_clusters_status_chk
      CHECK (status IN ('active', 'merged', 'legacy'));
  END IF;
END $$;

-- v1 rows have no v2 centroid; their embeddings may come from a different
-- model, and they are months old. Retire them from search, keep them for
-- history (articles.cluster_id still points at them).
UPDATE article_clusters
   SET status = 'legacy'
 WHERE status = 'active'
   AND centroid_sum IS NULL;

-- Window pre-filter for assignment: active clusters of one model, by recency.
CREATE INDEX IF NOT EXISTS article_clusters_v2_window_idx
  ON article_clusters (embedding_model, last_seen_at DESC)
  WHERE status = 'active';

-- Merge sweep probes "clusters touched since T".
CREATE INDEX IF NOT EXISTS article_clusters_v2_updated_idx
  ON article_clusters (embedding_model, updated_at)
  WHERE status = 'active';

-- ANN index for the merge sweep's nearest-cluster lookups.
CREATE INDEX IF NOT EXISTS article_clusters_v2_centroid_hnsw
  ON article_clusters USING hnsw (centroid halfvec_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE status = 'active';

CREATE INDEX IF NOT EXISTS article_clusters_merged_into_idx
  ON article_clusters (merged_into)
  WHERE merged_into IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2. articles — clustering provenance (audit trail for every assignment)
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_similarity real;   -- average-link cosine at join (1.0 for seeds)
ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_assignment text;   -- seed | join | gray_join
ALTER TABLE articles ADD COLUMN IF NOT EXISTS clustered_at       timestamptz;

-- (No "unclustered" partial index: on the live table nearly every row is
--  unclustered, so it would duplicate the 2.9M-row published_at index for
--  nothing. wizer_fetch_unclustered walks idx_articles_published_at instead.)
DROP INDEX IF EXISTS articles_unclustered_idx;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3. jsonb helpers (pure, IMMUTABLE — unit-tested in tests/test_clustering_sql.py)
-- ─────────────────────────────────────────────────────────────────────────────

-- Merge two [{text, type, count}] lists: case-insensitive key, counts summed,
-- first-seen spelling/type kept, top p_limit by count. Elements without a
-- "count" count as 1 (that is how a single article's entities arrive).
CREATE OR REPLACE FUNCTION wizer_merge_top_entities(
  p_existing jsonb,
  p_new      jsonb,
  p_limit    integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT coalesce(
           jsonb_agg(jsonb_build_object('text', m.t, 'type', m.ty, 'count', m.n)
                     ORDER BY m.n DESC, m.k),
           '[]'::jsonb)
    FROM (
      SELECT lower(s.e ->> 'text')                                  AS k,
             (array_agg(s.e ->> 'text' ORDER BY s.ord))[1]          AS t,
             (array_agg(s.e ->> 'type' ORDER BY s.ord))[1]          AS ty,
             sum(coalesce((s.e ->> 'count')::integer, 1))::integer  AS n
        FROM (
          SELECT x.e, x.ord
            FROM jsonb_array_elements(coalesce(p_existing, '[]'::jsonb)) WITH ORDINALITY AS x(e, ord)
          UNION ALL
          SELECT y.e, y.ord + 1000000
            FROM jsonb_array_elements(coalesce(p_new, '[]'::jsonb)) WITH ORDINALITY AS y(e, ord)
        ) AS s
       WHERE coalesce(btrim(s.e ->> 'text'), '') <> ''
       GROUP BY lower(s.e ->> 'text')
       ORDER BY n DESC, k
       LIMIT greatest(p_limit, 0)
    ) AS m;
$$;

-- Set-union of two jsonb string arrays, order of first appearance, capped.
CREATE OR REPLACE FUNCTION wizer_jsonb_text_union(
  p_a     jsonb,
  p_b     jsonb,
  p_limit integer DEFAULT 200
)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT coalesce(jsonb_agg(to_jsonb(u.v) ORDER BY u.first_ord), '[]'::jsonb)
    FROM (
      SELECT s.v, min(s.ord) AS first_ord
        FROM (
          SELECT x.v, x.ord
            FROM jsonb_array_elements_text(coalesce(p_a, '[]'::jsonb)) WITH ORDINALITY AS x(v, ord)
          UNION ALL
          SELECT y.v, y.ord + 1000000
            FROM jsonb_array_elements_text(coalesce(p_b, '[]'::jsonb)) WITH ORDINALITY AS y(v, ord)
        ) AS s
       WHERE s.v <> ''
       GROUP BY s.v
       ORDER BY min(s.ord)
       LIMIT greatest(p_limit, 0)
    ) AS u;
$$;

-- Union of [{h: phash, d: domain}] lists keyed on h, keeping the NEWEST
-- p_limit entries (later elements of p_b are newest).
CREATE OR REPLACE FUNCTION wizer_merge_image_hashes(
  p_a     jsonb,
  p_b     jsonb,
  p_limit integer DEFAULT 30
)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT coalesce(jsonb_agg(k.e ORDER BY k.last_ord), '[]'::jsonb)
    FROM (
      SELECT (array_agg(s.e ORDER BY s.ord DESC))[1] AS e, max(s.ord) AS last_ord
        FROM (
          SELECT x.e, x.ord
            FROM jsonb_array_elements(coalesce(p_a, '[]'::jsonb)) WITH ORDINALITY AS x(e, ord)
          UNION ALL
          SELECT y.e, y.ord + 1000000
            FROM jsonb_array_elements(coalesce(p_b, '[]'::jsonb)) WITH ORDINALITY AS y(e, ord)
        ) AS s
       WHERE s.e ? 'h'
       GROUP BY s.e ->> 'h'
       ORDER BY max(s.ord) DESC
       LIMIT greatest(p_limit, 0)
    ) AS k;
$$;

-- Hamming distance between two signed-int64 perceptual hashes.
-- bigint → bit(64) reinterprets the two's-complement bit pattern, so signed
-- storage (see enrichment/steps/images.py) compares correctly.
CREATE OR REPLACE FUNCTION wizer_phash_distance(p_a bigint, p_b bigint)
RETURNS integer
LANGUAGE sql
IMMUTABLE
STRICT
AS $$
  SELECT bit_count((p_a # p_b)::bit(64))::integer;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4. wizer_assign_cluster — THE clustering decision. One call per article.
--
-- Everything happens inside one transaction holding a global advisory lock, so
-- concurrent runners serialise on clustering only (milliseconds per article)
-- while their NLP work stays parallel. The lock makes "find nearest → decide →
-- create or update" atomic: two outlets' copies of a story processed at the
-- same instant by two runners can no longer seed two clusters.
--
-- Idempotent: an article that already has a cluster_id is returned as
-- 'existing' and changes nothing (re-enrichment with --force, or a retry after
-- the enrichment save failed, can never double-count).
--
-- Decision rule. Candidates are the p_candidates active clusters (same
-- embedding model, time-compatible) whose centroid is nearest the article.
-- Each is scored by AVERAGE-LINK similarity — the mean cosine between the
-- article and every member, computed exactly as (q · Σm) / n from the stored
-- sum. (Centroid similarity alone "chains": a large cluster's averaged
-- centroid becomes generic and attracts neighbouring stories; calibration
-- showed loose precision 0.81 → 0.89 with average-link, docs/CLUSTERING.md.)
-- In order of that score, the first candidate that passes wins:
--
--   anchor_sim  >= p_anchor_threshold                   (always required)
--   and either
--     sim >= p_join_threshold                           → 'join'
--     sim >= p_gray_threshold and corroborated          → 'gray_join'
--       corroborated = ≥ p_min_shared_entities salient entities in common
--                      OR a near-identical top image (pHash distance ≤
--                      p_image_max_distance) contributed by a DIFFERENT outlet
--                      (same-outlet matches are usually that outlet's logo /
--                      placeholder image, not evidence)
--       (the gray zone is off when p_gray_threshold = p_join_threshold —
--        the shipped default, see enrichment/config.py)
--   no candidate passes                                 → 'seed' (new cluster)
--
-- Time compatibility (article time t, cluster span [first, last]):
--   first - gap <= t <= last + gap          (story still "live")
--   max(last, t) - min(first, t) <= span    (no unbounded growth)
--
-- p_embedding must be the L2-normalised article embedding (dimension 768).
-- p_entity_keys are the article's salient entity keys, lower-cased and
-- stop-listed by the caller (enrichment/clustering.py).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_assign_cluster(
  p_article_id            bigint,
  p_embedding             vector,
  p_model                 text,
  p_published_at          timestamptz,
  p_domain                text,
  p_title                 text,
  p_language              text,
  p_entities              jsonb     DEFAULT '[]',
  p_entity_keys           text[]    DEFAULT '{}',
  p_image_phash           bigint    DEFAULT NULL,
  p_join_threshold        float8    DEFAULT 0.80,
  p_gray_threshold        float8    DEFAULT 0.70,
  p_anchor_threshold      float8    DEFAULT 0.65,
  p_min_shared_entities   integer   DEFAULT 2,
  p_image_max_distance    integer   DEFAULT 6,
  p_max_gap_hours         integer   DEFAULT 18,
  p_max_span_hours        integer   DEFAULT 120,
  p_candidates            integer   DEFAULT 5
)
RETURNS TABLE (
  cluster_id     uuid,
  action         text,
  similarity     float8,
  article_count  integer,
  outlet_count   integer
)
LANGUAGE plpgsql
VOLATILE
SET search_path = public, extensions
AS $$
DECLARE
  v_t          timestamptz := coalesce(p_published_at, now());
  v_gap        interval    := make_interval(hours => greatest(p_max_gap_hours, 0));
  v_span       interval    := make_interval(hours => greatest(p_max_span_hours, 0));
  v_q          halfvec;
  v_unit       vector;
  v_domain     text        := coalesce(nullif(btrim(lower(p_domain)), ''), '(unknown)');
  v_lang       text        := nullif(btrim(lower(p_language)), '');
  v_keys_json  jsonb;
  v_img_json   jsonb       := '[]'::jsonb;
  v_existing   uuid;
  v_found      boolean;
  v_cand       record;
  v_shared     integer;
  v_img_match  boolean;
  v_pick_id    uuid;
  v_pick_sim   float8;
  v_action     text;
  v_c          article_clusters%ROWTYPE;
  v_new_sum    vector;
  v_new_cent   vector;
  v_outlets    jsonb;
  v_rep_new    float8;
  v_rep_old    float8;
BEGIN
  IF p_embedding IS NULL THEN
    RAISE EXCEPTION 'wizer_assign_cluster: p_embedding is NULL (article %)', p_article_id;
  END IF;
  IF vector_dims(p_embedding) <> 768 THEN
    RAISE EXCEPTION 'wizer_assign_cluster: embedding has % dimensions, expected 768 (article %)',
      vector_dims(p_embedding), p_article_id;
  END IF;
  -- A zero or NaN vector has an undefined (NaN) cosine to everything, and
  -- Postgres orders NaN ABOVE every number — so `NaN >= threshold` is TRUE and
  -- a broken embedding would be joined into the nearest cluster, then poison
  -- that cluster's centroid. Refuse it outright.
  IF vector_norm(p_embedding) = 0 OR vector_norm(p_embedding) = 'NaN'::float8 THEN
    RAISE EXCEPTION 'wizer_assign_cluster: zero or NaN embedding (article %)', p_article_id;
  END IF;
  v_unit := l2_normalize(p_embedding);
  v_q    := v_unit::halfvec;

  SELECT coalesce(jsonb_agg(DISTINCT k), '[]'::jsonb)
    INTO v_keys_json
    FROM unnest(coalesce(p_entity_keys, '{}'::text[])) AS k
   WHERE k <> '';

  IF p_image_phash IS NOT NULL THEN
    v_img_json := jsonb_build_array(jsonb_build_object('h', p_image_phash, 'd', v_domain));
  END IF;

  -- Serialise all cluster mutations (assign + merge share this key).
  PERFORM pg_advisory_xact_lock(hashtextextended('wizer:article_clusters', 0));

  -- Idempotency + row lock on the article.
  SELECT a.cluster_id, true INTO v_existing, v_found
    FROM articles AS a
   WHERE a.id = p_article_id
     FOR UPDATE;
  IF NOT coalesce(v_found, false) THEN
    RAISE EXCEPTION 'wizer_assign_cluster: article % does not exist', p_article_id;
  END IF;
  IF v_existing IS NOT NULL THEN
    RETURN QUERY
      SELECT c.id, 'existing'::text, NULL::float8, c.article_count, c.outlet_count
        FROM article_clusters AS c
       WHERE c.id = v_existing;
    IF NOT FOUND THEN   -- dangling pointer (cluster deleted) → report, don't touch
      RETURN QUERY SELECT v_existing, 'existing'::text, NULL::float8, NULL::integer, NULL::integer;
    END IF;
    RETURN;
  END IF;

  -- ── Candidate search: exact cosine over the time window ───────────────────
  -- MATERIALIZED forces the window filter to run first and the distance sort
  -- to be exact. (An HNSW scan with a WHERE clause post-filters a fixed-size
  -- candidate list and silently drops matches when many out-of-window
  -- clusters are nearer — exactly the failure mode that matters here.)
  FOR v_cand IN
    WITH win AS MATERIALIZED (
      SELECT c.id, c.centroid, c.anchor
        FROM article_clusters AS c
       WHERE c.status = 'active'
         AND c.embedding_model = p_model
         AND c.last_seen_at  >= v_t - v_gap
         AND c.first_seen_at <= v_t + v_gap
         AND greatest(c.last_seen_at, v_t) - least(c.first_seen_at, v_t) <= v_span
    ), nearest AS (
      SELECT w.id, 1 - (w.anchor <=> v_q) AS anchor_sim
        FROM win AS w
       ORDER BY w.centroid <=> v_q
       LIMIT greatest(p_candidates, 1)
    )
    -- Average-link similarity = mean cosine between the article and EVERY
    -- member, computed exactly from the stored sum of unit vectors:
    --   avg_i (q · m_i) = q · Σ m_i / n          (<#> is NEGATIVE inner product)
    -- Only the few nearest clusters pay for reading their (TOASTed) sum.
    SELECT n.id,
           n.anchor_sim,
           c.top_entities,
           c.image_hashes,
           -(c.centroid_sum <#> v_unit) / c.article_count AS sim
      FROM nearest AS n
      JOIN article_clusters AS c ON c.id = n.id
     ORDER BY sim DESC
  LOOP
    EXIT WHEN v_cand.sim < p_gray_threshold;          -- sorted: nothing better follows
    CONTINUE WHEN v_cand.anchor_sim < p_anchor_threshold;

    IF v_cand.sim >= p_join_threshold THEN
      v_pick_id := v_cand.id; v_pick_sim := v_cand.sim; v_action := 'join';
      EXIT;
    END IF;

    SELECT count(*) INTO v_shared
      FROM (
        SELECT lower(e ->> 'text') FROM jsonb_array_elements(v_cand.top_entities) AS e
        INTERSECT
        SELECT k FROM unnest(coalesce(p_entity_keys, '{}'::text[])) AS k
      ) AS s;

    v_img_match := p_image_phash IS NOT NULL AND EXISTS (
      SELECT 1
        FROM jsonb_array_elements(v_cand.image_hashes) AS h
       WHERE (h ->> 'd') IS DISTINCT FROM v_domain
         AND wizer_phash_distance((h ->> 'h')::bigint, p_image_phash) <= p_image_max_distance
    );

    IF v_shared >= p_min_shared_entities OR v_img_match THEN
      v_pick_id := v_cand.id; v_pick_sim := v_cand.sim; v_action := 'gray_join';
      EXIT;
    END IF;
  END LOOP;

  -- ── Seed a new cluster ────────────────────────────────────────────────────
  IF v_pick_id IS NULL THEN
    INSERT INTO article_clusters AS c (
      canonical_article_id, representative_article_id, headline,
      centroid_sum, centroid, anchor, representative, embedding_model,
      article_count, outlet_count, outlet_set,
      entity_set, top_entities, image_hashes, language_set,
      first_seen_at, last_seen_at, status, updated_at
    ) VALUES (
      p_article_id, p_article_id, left(p_title, 500),
      l2_normalize(p_embedding), v_q, v_q, v_q, p_model,
      1, 1, jsonb_build_array(v_domain),
      v_keys_json,
      wizer_merge_top_entities('[]'::jsonb, p_entities, 30),
      v_img_json,
      CASE WHEN v_lang IS NULL THEN '[]'::jsonb ELSE jsonb_build_array(v_lang) END,
      v_t, v_t, 'active', now()
    )
    RETURNING c.id INTO v_pick_id;

    UPDATE articles
       SET cluster_id = v_pick_id, cluster_similarity = 1.0,
           cluster_assignment = 'seed', clustered_at = now()
     WHERE id = p_article_id;

    RETURN QUERY SELECT v_pick_id, 'seed'::text, 1.0::float8, 1, 1;
    RETURN;
  END IF;

  -- ── Join the chosen cluster ───────────────────────────────────────────────
  SELECT * INTO v_c FROM article_clusters WHERE id = v_pick_id FOR UPDATE;

  v_new_sum  := v_c.centroid_sum + v_unit;
  v_new_cent := l2_normalize(v_new_sum);
  v_outlets  := CASE WHEN v_c.outlet_set ? v_domain
                     THEN v_c.outlet_set
                     ELSE v_c.outlet_set || to_jsonb(v_domain) END;

  -- Representative = the member closest to the CURRENT centroid. Both sides are
  -- measured against the same, new centroid, so the comparison is exact.
  v_rep_new := 1 - (v_new_cent::halfvec <=> v_q);
  v_rep_old := CASE WHEN v_c.representative IS NULL THEN -2
                    ELSE 1 - (v_new_cent::halfvec <=> v_c.representative) END;

  UPDATE article_clusters AS c
     SET centroid_sum    = v_new_sum,
         centroid        = v_new_cent::halfvec,
         article_count   = c.article_count + 1,
         outlet_set      = v_outlets,
         outlet_count    = jsonb_array_length(v_outlets),
         entity_set      = wizer_jsonb_text_union(c.entity_set, v_keys_json, 200),
         top_entities    = wizer_merge_top_entities(c.top_entities, p_entities, 30),
         image_hashes    = wizer_merge_image_hashes(c.image_hashes, v_img_json, 30),
         language_set    = CASE WHEN v_lang IS NULL THEN c.language_set
                                ELSE wizer_jsonb_text_union(c.language_set, jsonb_build_array(v_lang), 50) END,
         first_seen_at   = least(c.first_seen_at, v_t),
         last_seen_at    = greatest(c.last_seen_at, v_t),
         gray_join_count = c.gray_join_count + (v_action = 'gray_join')::integer,
         representative            = CASE WHEN v_rep_new > v_rep_old THEN v_q ELSE c.representative END,
         representative_article_id = CASE WHEN v_rep_new > v_rep_old THEN p_article_id ELSE c.representative_article_id END,
         headline                  = CASE WHEN v_rep_new > v_rep_old AND coalesce(p_title, '') <> ''
                                          THEN left(p_title, 500) ELSE c.headline END,
         updated_at      = now()
   WHERE c.id = v_pick_id;

  UPDATE articles
     SET cluster_id = v_pick_id, cluster_similarity = v_pick_sim,
         cluster_assignment = v_action, clustered_at = now()
   WHERE id = p_article_id;

  RETURN QUERY
    SELECT c.id, v_action, v_pick_sim, c.article_count, c.outlet_count
      FROM article_clusters AS c
     WHERE c.id = v_pick_id;
END;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5. wizer_merge_clusters — fold p_loser into p_winner (maintenance)
--
-- Moves every member article, adds the centroid sums (exact), unions outlets /
-- entities / images / languages, widens the time span, and keeps whichever
-- representative is more central to the merged centroid. The loser becomes a
-- tombstone (status='merged', merged_into=winner) so any app that cached the
-- old cluster id can follow the pointer; tombstones that pointed at the loser
-- are re-pointed at the winner, so chains never form.
-- Returns false (and changes nothing) unless both are active, distinct, and
-- built with the same embedding model.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_merge_clusters(p_winner uuid, p_loser uuid)
RETURNS boolean
LANGUAGE plpgsql
VOLATILE
SET search_path = public, extensions
AS $$
DECLARE
  w        article_clusters%ROWTYPE;
  l        article_clusters%ROWTYPE;
  v_sum    vector;
  v_cent   vector;
  v_outl   jsonb;
  v_w_rep  float8;
  v_l_rep  float8;
BEGIN
  IF p_winner IS NULL OR p_loser IS NULL OR p_winner = p_loser THEN
    RETURN false;
  END IF;

  PERFORM pg_advisory_xact_lock(hashtextextended('wizer:article_clusters', 0));

  SELECT * INTO w FROM article_clusters WHERE id = p_winner FOR UPDATE;
  SELECT * INTO l FROM article_clusters WHERE id = p_loser  FOR UPDATE;
  IF w.id IS NULL OR l.id IS NULL
     OR w.status <> 'active' OR l.status <> 'active'
     OR w.embedding_model IS DISTINCT FROM l.embedding_model
     OR w.centroid_sum IS NULL OR l.centroid_sum IS NULL THEN
    RETURN false;
  END IF;

  v_sum  := w.centroid_sum + l.centroid_sum;
  v_cent := l2_normalize(v_sum);
  v_outl := wizer_jsonb_text_union(w.outlet_set, l.outlet_set, 100000);
  v_w_rep := 1 - (v_cent::halfvec <=> w.representative);
  v_l_rep := 1 - (v_cent::halfvec <=> l.representative);

  UPDATE articles SET cluster_id = p_winner WHERE cluster_id = p_loser;

  UPDATE article_clusters AS c
     SET centroid_sum    = v_sum,
         centroid        = v_cent::halfvec,
         article_count   = w.article_count + l.article_count,
         outlet_set      = v_outl,
         outlet_count    = jsonb_array_length(v_outl),
         entity_set      = wizer_jsonb_text_union(w.entity_set, l.entity_set, 200),
         top_entities    = wizer_merge_top_entities(w.top_entities, l.top_entities, 30),
         image_hashes    = wizer_merge_image_hashes(l.image_hashes, w.image_hashes, 30),
         language_set    = wizer_jsonb_text_union(w.language_set, l.language_set, 50),
         first_seen_at   = least(w.first_seen_at, l.first_seen_at),
         last_seen_at    = greatest(w.last_seen_at, l.last_seen_at),
         gray_join_count = w.gray_join_count + l.gray_join_count,
         representative            = CASE WHEN v_l_rep > v_w_rep THEN l.representative ELSE w.representative END,
         representative_article_id = CASE WHEN v_l_rep > v_w_rep THEN l.representative_article_id ELSE w.representative_article_id END,
         headline                  = CASE WHEN v_l_rep > v_w_rep THEN l.headline ELSE w.headline END,
         updated_at      = now()
   WHERE c.id = p_winner;

  UPDATE article_clusters
     SET status = 'merged', merged_into = p_winner, updated_at = now()
   WHERE id = p_loser;

  UPDATE article_clusters
     SET merged_into = p_winner
   WHERE merged_into = p_loser;

  RETURN true;
END;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6. wizer_find_cluster_merge_candidates — duplicate-cluster sweep
--
-- Online clustering is order-dependent: if the first two outlets' versions of a
-- story arrive with unusually different headlines, they seed two clusters and
-- later articles split between them. This sweep finds such twins.
--
-- Probes = active clusters touched since p_since, paged by (updated_at, id)
-- via p_after_ts / p_after_id. For each probe, the single nearest compatible
-- active cluster (same model, time-compatible, anchors agree) is returned if
-- its AVERAGE-LINK similarity (mean cosine over all cross-cluster article
-- pairs, exact from the stored sums) is ≥ p_threshold. EVERY probe is returned
-- (other_id NULL when nothing qualifies) so the caller can page with the last
-- probe's (updated_at, id). Pairs can appear in both directions; the caller
-- dedups and merges smaller → larger.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_find_cluster_merge_candidates(
  p_model            text,
  p_since            timestamptz,
  p_threshold        float8      DEFAULT 0.86,
  p_anchor_threshold float8      DEFAULT 0.65,
  p_max_gap_hours    integer     DEFAULT 18,
  p_max_span_hours   integer     DEFAULT 120,
  p_probe_limit      integer     DEFAULT 200,
  p_after_ts         timestamptz DEFAULT '-infinity',
  p_after_id         uuid        DEFAULT '00000000-0000-0000-0000-000000000000'
)
RETURNS TABLE (
  probe_id          uuid,
  probe_updated_at  timestamptz,
  probe_count       integer,
  other_id          uuid,
  other_count       integer,
  similarity        float8
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  WITH probes AS MATERIALIZED (
    SELECT p.id, p.centroid, p.anchor, p.centroid_sum, p.first_seen_at, p.last_seen_at,
           p.article_count, p.updated_at
      FROM article_clusters AS p
     WHERE p.status = 'active'
       AND p.embedding_model = p_model
       AND p.updated_at >= p_since
       AND (p.updated_at, p.id) > (p_after_ts, p_after_id)
     ORDER BY p.updated_at, p.id
     LIMIT greatest(p_probe_limit, 1)
  )
  SELECT pr.id, pr.updated_at, pr.article_count,
         nb.id, nb.article_count, nb.sim
    FROM probes AS pr
    LEFT JOIN LATERAL (
      -- nearest compatible neighbour by centroid (cheap: halfvec, in-page) …
      SELECT x.id, x.article_count,
             -- … then its exact AVERAGE-LINK similarity: mean cosine over all
             -- member pairs = (Σa · Σb) / (na · nb). Only this one row reads
             -- the TOASTed sum vector.
             -(pr.centroid_sum <#> c2.centroid_sum) / (pr.article_count * x.article_count) AS sim
        FROM (
          SELECT c.id, c.article_count
            FROM article_clusters AS c
           WHERE c.status = 'active'
             AND c.embedding_model = p_model
             AND c.id <> pr.id
             AND c.last_seen_at  >= pr.first_seen_at - make_interval(hours => p_max_gap_hours)
             AND c.first_seen_at <= pr.last_seen_at  + make_interval(hours => p_max_gap_hours)
             AND greatest(c.last_seen_at, pr.last_seen_at) - least(c.first_seen_at, pr.first_seen_at)
                   <= make_interval(hours => p_max_span_hours)
             AND 1 - (c.anchor <=> pr.anchor) >= p_anchor_threshold
           ORDER BY c.centroid <=> pr.centroid
           LIMIT 1
        ) AS x
        JOIN article_clusters AS c2 ON c2.id = x.id
    ) AS nb ON nb.sim >= p_threshold
   ORDER BY pr.updated_at, pr.id;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 7. wizer_reconcile_cluster_counts — consistency audit / repair
--
-- Recomputes article_count, outlet_set/outlet_count and the seen-span of
-- active clusters touched since p_since from their actual member articles.
-- Under normal operation this fixes nothing (returns 0) — a non-zero result
-- means something outside the assignment path changed membership, e.g.
-- Layer 1's table pruning deleting old articles.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_reconcile_cluster_counts(p_since timestamptz)
RETURNS integer
LANGUAGE plpgsql
VOLATILE
SET search_path = public, extensions
AS $$
DECLARE
  v_fixed integer;
BEGIN
  PERFORM pg_advisory_xact_lock(hashtextextended('wizer:article_clusters', 0));

  WITH truth AS (
    SELECT a.cluster_id AS id,
           count(*)::integer AS n,
           jsonb_agg(DISTINCT coalesce(nullif(btrim(lower(a.domain)), ''), '(unknown)')) AS outlets,
           min(coalesce(a.published_at, a.created_at)) AS first_t,
           max(coalesce(a.published_at, a.created_at)) AS last_t
      FROM articles AS a
      JOIN article_clusters AS c ON c.id = a.cluster_id
     WHERE c.status = 'active'
       AND c.updated_at >= p_since
     GROUP BY a.cluster_id
  ), fixed AS (
    UPDATE article_clusters AS c
       SET article_count = t.n,
           outlet_set    = t.outlets,
           outlet_count  = jsonb_array_length(t.outlets),
           first_seen_at = least(c.first_seen_at, t.first_t),
           last_seen_at  = greatest(c.last_seen_at, t.last_t)
      FROM truth AS t
     WHERE c.id = t.id
       AND (c.article_count <> t.n OR c.outlet_count <> jsonb_array_length(t.outlets))
    RETURNING 1
  )
  SELECT count(*)::integer INTO v_fixed FROM fixed;
  RETURN v_fixed;
END;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8. wizer_prune_orphan_clusters — delete clusters nobody points at
--
-- Orphans come from Layer 1 table pruning (all members deleted), from merge
-- tombstones once they have aged out, and from v1 'legacy' clusters whose
-- articles are gone. Only clusters idle for p_older_than_hours are touched.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION wizer_prune_orphan_clusters(p_older_than_hours integer DEFAULT 168)
RETURNS integer
LANGUAGE sql
VOLATILE
SET search_path = public, extensions
AS $$
  WITH gone AS (
    DELETE FROM article_clusters AS c
     WHERE c.updated_at < now() - make_interval(hours => greatest(p_older_than_hours, 1))
       AND NOT EXISTS (SELECT 1 FROM articles AS a WHERE a.cluster_id = c.id)
    RETURNING 1
  )
  SELECT count(*)::integer FROM gone;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8b. Clustering job feed + batched assignment (cluster.py run)
--
-- Clustering is decoupled from NLP enrichment: embedding a headline costs
-- ~0.1-0.3 s on CPU while the full NLP set costs ~15 s, so the clustering job
-- keeps up with ALL ingested articles (~50K/day) no matter how far NLP
-- enrichment lags. outlet_count is only meaningful if every outlet's copy of a
-- story is clustered, not just the ~5 % that get enriched.
-- ─────────────────────────────────────────────────────────────────────────────

-- Articles without a cluster published since p_since, oldest first, keyset-
-- paged on (published_at, id). Returns only what embedding needs — the body is
-- trimmed to 1,500 chars so a page of 500 rows stays small over the wire.
CREATE OR REPLACE FUNCTION wizer_fetch_unclustered(
  p_since           timestamptz,
  p_limit           integer     DEFAULT 500,
  p_after_published timestamptz DEFAULT '-infinity',
  p_after_id        bigint      DEFAULT 0
)
RETURNS TABLE (
  id                bigint,
  title             text,
  description       text,
  body_lead         text,
  domain            text,
  language_code     text,
  language_detected text,
  published_at      timestamptz,
  image_phash       bigint
)
LANGUAGE sql
STABLE
SET search_path = public, extensions
AS $$
  SELECT a.id, a.title, a.description, left(a.full_text, 1500), a.domain,
         a.language_code::text, a.language_detected, a.published_at, a.image_phash
    FROM articles AS a
   WHERE a.published_at >= p_since
     AND a.published_at <= now() + interval '1 hour'
     AND a.cluster_id IS NULL
     AND a.title IS NOT NULL
     AND (a.published_at, a.id) > (p_after_published, p_after_id)
   ORDER BY a.published_at, a.id
   LIMIT greatest(p_limit, 1);
$$;

-- Assign many articles in ONE round trip. Each item is the argument set of
-- wizer_assign_cluster (minus the shared thresholds). Every item runs in its
-- own sub-transaction: a bad item (missing article, broken vector) is reported
-- in `error` and the rest of the batch still commits.
-- Why: the database is in ap-southeast-1 and GitHub runners are mostly in the
-- US — ~200 ms per round trip. 50 articles per call turns 50 round trips into 1.
CREATE OR REPLACE FUNCTION wizer_assign_cluster_batch(
  p_items                 jsonb,
  p_model                 text,
  p_join_threshold        float8  DEFAULT 0.85,
  p_gray_threshold        float8  DEFAULT 0.85,
  p_anchor_threshold      float8  DEFAULT 0.75,
  p_min_shared_entities   integer DEFAULT 2,
  p_image_max_distance    integer DEFAULT 6,
  p_max_gap_hours         integer DEFAULT 18,
  p_max_span_hours        integer DEFAULT 120,
  p_candidates            integer DEFAULT 5
)
RETURNS TABLE (
  article_id     bigint,
  cluster_id     uuid,
  action         text,
  similarity     float8,
  article_count  integer,
  outlet_count   integer,
  error          text
)
LANGUAGE plpgsql
VOLATILE
SET search_path = public, extensions
AS $$
DECLARE
  it  jsonb;
  r   record;
BEGIN
  FOR it IN SELECT value FROM jsonb_array_elements(coalesce(p_items, '[]'::jsonb)) LOOP
    BEGIN
      SELECT * INTO r FROM wizer_assign_cluster(
        (it ->> 'p_article_id')::bigint,
        (it ->> 'p_embedding')::vector,
        p_model,
        (it ->> 'p_published_at')::timestamptz,
        it ->> 'p_domain',
        it ->> 'p_title',
        it ->> 'p_language',
        coalesce(it -> 'p_entities', '[]'::jsonb),
        coalesce(ARRAY(SELECT jsonb_array_elements_text(coalesce(it -> 'p_entity_keys', '[]'::jsonb))), '{}'),
        (it ->> 'p_image_phash')::bigint,
        p_join_threshold, p_gray_threshold, p_anchor_threshold, p_min_shared_entities,
        p_image_max_distance, p_max_gap_hours, p_max_span_hours, p_candidates);
      article_id    := (it ->> 'p_article_id')::bigint;
      cluster_id    := r.cluster_id;
      action        := r.action;
      similarity    := r.similarity;
      article_count := r.article_count;
      outlet_count  := r.outlet_count;
      error         := NULL;
    EXCEPTION WHEN OTHERS THEN
      article_id := (it ->> 'p_article_id')::bigint;
      cluster_id := NULL; action := NULL; similarity := NULL;
      article_count := NULL; outlet_count := NULL;
      error := SQLERRM;
    END;
    RETURN NEXT;
  END LOOP;
END;
$$;


-- ─────────────────────────────────────────────────────────────────────────────
-- 9. Monitoring views
-- ─────────────────────────────────────────────────────────────────────────────

-- Stories of the last 24 h, biggest first. The app-facing "top stories" query.
CREATE OR REPLACE VIEW top_stories_24h AS
SELECT c.id,
       c.headline,
       c.outlet_count,
       c.article_count,
       c.language_set,
       c.top_entities,
       c.first_seen_at,
       c.last_seen_at,
       c.representative_article_id
  FROM article_clusters AS c
 WHERE c.status = 'active'
   AND c.last_seen_at > now() - interval '24 hours'
 ORDER BY c.outlet_count DESC, c.article_count DESC, c.last_seen_at DESC;

-- One-row clustering health dashboard over the last 24 h.
--   clustered_pct      share of enriched articles that got a cluster
--   seed_pct           share of assignments that started a new story
--   gray_join_pct      share of joins rescued by entity / image evidence
--   multi_outlet_*     stories covered by ≥2 outlets (the virality signal)
CREATE OR REPLACE VIEW cluster_health AS
WITH a AS (
  SELECT cluster_assignment
    FROM articles
   WHERE enriched_at > now() - interval '24 hours'
), c AS (
  SELECT outlet_count, article_count
    FROM article_clusters
   WHERE status = 'active'
     AND last_seen_at > now() - interval '24 hours'
)
SELECT
  (SELECT count(*) FROM a)                                                      AS enriched_24h,
  (SELECT round(100.0 * count(*) FILTER (WHERE cluster_assignment IS NOT NULL)
                / nullif(count(*), 0), 1) FROM a)                               AS clustered_pct,
  (SELECT round(100.0 * count(*) FILTER (WHERE cluster_assignment = 'seed')
                / nullif(count(*) FILTER (WHERE cluster_assignment IS NOT NULL), 0), 1) FROM a) AS seed_pct,
  (SELECT round(100.0 * count(*) FILTER (WHERE cluster_assignment = 'gray_join')
                / nullif(count(*) FILTER (WHERE cluster_assignment IN ('join', 'gray_join')), 0), 1) FROM a) AS gray_join_pct,
  (SELECT count(*) FROM c)                                                      AS active_clusters_24h,
  (SELECT count(*) FROM c WHERE outlet_count >= 2)                              AS multi_outlet_clusters_24h,
  (SELECT max(outlet_count) FROM c)                                             AS max_outlet_count_24h,
  (SELECT round(avg(article_count), 2) FROM c)                                  AS avg_cluster_size_24h,
  (SELECT count(*) FROM article_clusters WHERE status = 'merged'
                                          AND updated_at > now() - interval '24 hours') AS merges_24h;


-- ─────────────────────────────────────────────────────────────────────────────
-- 10. Permissions — service role only (see enrichment_queue_migration.sql §7)
-- ─────────────────────────────────────────────────────────────────────────────
DO $$
DECLARE
  fn text;
BEGIN
  FOREACH fn IN ARRAY ARRAY[
    'wizer_assign_cluster(bigint, vector, text, timestamptz, text, text, text, jsonb, text[], bigint, float8, float8, float8, integer, integer, integer, integer, integer)',
    'wizer_merge_clusters(uuid, uuid)',
    'wizer_find_cluster_merge_candidates(text, timestamptz, float8, float8, integer, integer, integer, timestamptz, uuid)',
    'wizer_reconcile_cluster_counts(timestamptz)',
    'wizer_prune_orphan_clusters(integer)',
    'wizer_fetch_unclustered(timestamptz, integer, timestamptz, bigint)',
    'wizer_assign_cluster_batch(jsonb, text, float8, float8, float8, integer, integer, integer, integer, integer)'
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
