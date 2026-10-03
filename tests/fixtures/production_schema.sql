-- Production public-schema snapshot (DDL only, no data), taken read-only with
-- `pg_dump --schema-only --schema=public` on 2026-10-03, BEFORE the 2026-10
-- migrations. Used by tests/test_production_schema.py to catch drift between
-- what the code writes and what production actually has (e.g. articles.language
-- is char(5) here, text in docs/migration.sql). Refresh after schema changes.

SET check_function_bodies = false;   -- functions are dumped before the tables they use

--
-- PostgreSQL database dump
--


-- Dumped from database version 17.6
-- Dumped by pg_dump version 18.6


--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

-- (schema exists)


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS 'standard public schema';


--
-- Name: find_duplicate_clusters(double precision, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.find_duplicate_clusters(similarity_threshold double precision DEFAULT 0.87, max_pairs integer DEFAULT 200) RETURNS TABLE(cluster_a uuid, cluster_b uuid, similarity double precision, a_count integer, b_count integer)
    LANGUAGE sql STABLE
    AS $$
    SELECT
        a.id                                          AS cluster_a,
        near.id                                       AS cluster_b,
        1 - (a.embedding_vec <=> near.embedding_vec)  AS similarity,
        a.article_count,
        near.article_count
    FROM article_clusters a
    CROSS JOIN LATERAL (
        SELECT id, embedding_vec, article_count
        FROM article_clusters c
        WHERE c.id != a.id
          AND c.embedding_vec IS NOT NULL
          -- Use string cast so uuid comparisons are consistent
          AND c.id::text > a.id::text
          AND 1 - (c.embedding_vec <=> a.embedding_vec) >= similarity_threshold
        ORDER BY c.embedding_vec <=> a.embedding_vec
        LIMIT 1
    ) near
    WHERE a.embedding_vec IS NOT NULL
    LIMIT max_pairs;
$$;


--
-- Name: find_nearest_cluster(public.vector, double precision, integer); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.find_nearest_cluster(query_embedding public.vector, similarity_threshold double precision DEFAULT 0.82, window_hours integer DEFAULT 48) RETURNS TABLE(id uuid, canonical_article_id bigint, headline text, outlet_count integer, article_count integer, entity_set jsonb, outlet_set jsonb, top_entities jsonb, canonical_embedding jsonb, first_seen_at timestamp with time zone, last_seen_at timestamp with time zone, similarity double precision)
    LANGUAGE sql STABLE
    AS $$
  SELECT
    id,
    canonical_article_id,
    headline,
    outlet_count,
    article_count,
    entity_set,
    outlet_set,
    top_entities,
    canonical_embedding,
    first_seen_at,
    last_seen_at,
    1 - (embedding_vec <=> query_embedding) AS similarity
  FROM article_clusters
  WHERE embedding_vec IS NOT NULL
    AND (
      window_hours = 0
      OR last_seen_at > now() - make_interval(hours => window_hours)
    )
    AND 1 - (embedding_vec <=> query_embedding) >= similarity_threshold
  ORDER BY embedding_vec <=> query_embedding
  LIMIT 1;
$$;




--
-- Name: article_clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.article_clusters (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    canonical_article_id bigint,
    headline text,
    outlet_count integer DEFAULT 1 NOT NULL,
    article_count integer DEFAULT 1 NOT NULL,
    entity_set jsonb DEFAULT '[]'::jsonb NOT NULL,
    top_entities jsonb DEFAULT '[]'::jsonb NOT NULL,
    canonical_simhash bigint,
    gnews_data jsonb,
    first_seen_at timestamp with time zone,
    last_seen_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    canonical_embedding jsonb,
    embedding_vec public.vector(768),
    outlet_set jsonb DEFAULT '[]'::jsonb NOT NULL
);


--
-- Name: article_entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.article_entities (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    article_id bigint NOT NULL,
    entity_text text NOT NULL,
    entity_type text NOT NULL,
    salience double precision DEFAULT 0.5 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: articles; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.articles (
    id bigint NOT NULL,
    feed_id bigint,
    url text NOT NULL,
    url_hash character(32) NOT NULL,
    title text,
    title_simhash bigint,
    description text,
    full_text text,
    top_image_url text,
    author text,
    published_at timestamp with time zone,
    crawled_at timestamp with time zone,
    language character(5),
    country_code text,
    og_tags jsonb DEFAULT '{}'::jsonb,
    is_crawled boolean DEFAULT false,
    is_duplicate boolean DEFAULT false,
    story_id bigint,
    propensity_score double precision,
    created_at timestamp with time zone DEFAULT now(),
    feed_url text,
    domain text,
    publisher_name text,
    iab_tier1 text,
    iab_tier2 text,
    language_code character varying(10),
    crawl_strategy text,
    cluster_id uuid,
    enriched_at timestamp with time zone,
    word_count integer,
    reading_time_mins double precision,
    language_detected text,
    sentiment text,
    sentiment_score double precision,
    category text,
    keywords jsonb,
    image_phash bigint,
    sentiment_stats jsonb,
    ai_region jsonb,
    ai_org jsonb,
    ai_summary text,
    ai_tag jsonb
);


--
-- Name: article_stats; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.article_stats AS
 SELECT domain,
    publisher_name,
    count(*) AS total,
    count(*) FILTER (WHERE (is_crawled = true)) AS crawled,
    count(*) FILTER (WHERE (is_duplicate = true)) AS near_duplicates,
    count(*) FILTER (WHERE (published_at > (now() - '24:00:00'::interval))) AS last_24h,
    (round((((count(*) FILTER (WHERE (is_crawled = true)))::numeric / (NULLIF(count(*), 0))::numeric) * (100)::numeric)))::integer AS crawl_rate_pct
   FROM public.articles
  GROUP BY domain, publisher_name
  ORDER BY (count(*)) DESC;


--
-- Name: articles_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.articles_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: articles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.articles_id_seq OWNED BY public.articles.id;


--
-- Name: articles_ist; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.articles_ist AS
 SELECT id,
    feed_id,
    url,
    title,
    description,
    full_text,
    (published_at AT TIME ZONE 'Asia/Kolkata'::text) AS published_at_ist,
    (crawled_at AT TIME ZONE 'Asia/Kolkata'::text) AS crawled_at_ist,
    (created_at AT TIME ZONE 'Asia/Kolkata'::text) AS created_at_ist,
    (enriched_at AT TIME ZONE 'Asia/Kolkata'::text) AS enriched_at_ist,
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
   FROM public.articles;


--
-- Name: feeds; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.feeds (
    id bigint NOT NULL,
    feed_url text NOT NULL,
    domain text,
    publisher_name text,
    publisher_type text,
    country_code text,
    language_code character varying(10),
    iab_tier1 text,
    iab_tier2 text,
    feed_format text,
    is_active boolean DEFAULT true,
    last_polled_at timestamp with time zone,
    last_success_at timestamp with time zone,
    poll_interval_mins integer DEFAULT 60,
    fail_count integer DEFAULT 0,
    articles_found integer DEFAULT 0,
    created_at timestamp with time zone DEFAULT now(),
    final_url text,
    feed_type text,
    country_name text,
    language_name text,
    political_lean text,
    audience_type text,
    is_satire boolean,
    has_paywall boolean,
    item_count integer,
    latest_item_date timestamp with time zone,
    days_since_update integer,
    update_cadence text,
    avg_items_per_day double precision,
    freshness_score double precision,
    validation_tier text,
    priority_score double precision,
    http_status integer,
    content_type text,
    metadata_confidence double precision,
    metadata_source text,
    source_file text,
    feed_checked_at timestamp with time zone,
    title text
);


--
-- Name: dormancy_watch; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.dormancy_watch AS
 SELECT id,
    feed_url,
    publisher_name,
    validation_tier,
    fail_count,
    (last_polled_at AT TIME ZONE 'Asia/Kolkata'::text) AS last_polled_at_ist,
    (last_success_at AT TIME ZONE 'Asia/Kolkata'::text) AS last_success_at_ist,
    priority_score
   FROM public.feeds
  WHERE ((is_active = true) AND ((last_polled_at < (now() - '24:00:00'::interval)) OR (last_polled_at IS NULL)))
  ORDER BY last_polled_at NULLS FIRST;


--
-- Name: enrichment_health; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.enrichment_health AS
 SELECT count(*) AS total_articles,
    count(*) FILTER (WHERE (enriched_at IS NOT NULL)) AS enriched,
    count(*) FILTER (WHERE (enriched_at IS NULL)) AS pending,
    (round((((count(*) FILTER (WHERE (enriched_at IS NOT NULL)))::numeric / (NULLIF(count(*), 0))::numeric) * (100)::numeric)))::integer AS enriched_pct,
    count(*) FILTER (WHERE (category IS NOT NULL)) AS has_category,
    count(*) FILTER (WHERE (cluster_id IS NOT NULL)) AS clustered,
    count(*) FILTER (WHERE (language_detected IS NOT NULL)) AS language_detected,
    count(*) FILTER (WHERE (image_phash IS NOT NULL)) AS has_image_hash,
    ( SELECT count(*) AS count
           FROM public.article_clusters) AS total_clusters,
    ( SELECT count(*) AS count
           FROM public.article_entities) AS total_entities
   FROM public.articles;


--
-- Name: feed_health; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.feed_health AS
SELECT
    NULL::bigint AS id,
    NULL::text AS feed_url,
    NULL::text AS publisher_name,
    NULL::text AS validation_tier,
    NULL::boolean AS is_active,
    NULL::integer AS fail_count,
    NULL::integer AS articles_found,
    NULL::timestamp with time zone AS last_polled_at,
    NULL::timestamp with time zone AS last_success_at,
    NULL::double precision AS priority_score,
    NULL::bigint AS articles_last_24h;


--
-- Name: feeds_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.feeds_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: feeds_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.feeds_id_seq OWNED BY public.feeds.id;


--
-- Name: feeds_ist; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.feeds_ist AS
 SELECT id,
    feed_url,
    publisher_name,
    validation_tier,
    is_active,
    (last_polled_at AT TIME ZONE 'Asia/Kolkata'::text) AS last_polled_at_ist,
    (last_success_at AT TIME ZONE 'Asia/Kolkata'::text) AS last_success_at_ist,
    fail_count,
    articles_found,
    (created_at AT TIME ZONE 'Asia/Kolkata'::text) AS created_at_ist
   FROM public.feeds;


--
-- Name: pipeline_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_runs (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    tier text,
    feeds_attempted integer DEFAULT 0 NOT NULL,
    feeds_skipped integer DEFAULT 0 NOT NULL,
    new_articles integer DEFAULT 0 NOT NULL,
    near_duplicates integer DEFAULT 0 NOT NULL,
    exact_duplicates integer DEFAULT 0 NOT NULL,
    errors integer DEFAULT 0 NOT NULL,
    duration_s double precision,
    run_at timestamp with time zone DEFAULT now() NOT NULL,
    dry_run boolean DEFAULT false NOT NULL
);


--
-- Name: recent_runs; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.recent_runs AS
 SELECT id,
    tier,
    feeds_attempted,
    feeds_skipped,
    new_articles,
    near_duplicates,
    exact_duplicates,
    errors,
    duration_s,
    run_at,
    dry_run
   FROM public.pipeline_runs
  ORDER BY run_at DESC
 LIMIT 100;


--
-- Name: stories; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.stories (
    id bigint NOT NULL,
    headline text,
    summary text,
    first_seen_at timestamp with time zone,
    last_updated_at timestamp with time zone,
    article_count integer DEFAULT 0,
    source_count integer DEFAULT 0,
    velocity double precision DEFAULT 0,
    propensity_score double precision DEFAULT 0,
    category text,
    language character(5),
    entities jsonb DEFAULT '[]'::jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: stories_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.stories_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: stories_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.stories_id_seq OWNED BY public.stories.id;


--
-- Name: articles id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles ALTER COLUMN id SET DEFAULT nextval('public.articles_id_seq'::regclass);


--
-- Name: feeds id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.feeds ALTER COLUMN id SET DEFAULT nextval('public.feeds_id_seq'::regclass);


--
-- Name: stories id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stories ALTER COLUMN id SET DEFAULT nextval('public.stories_id_seq'::regclass);


--
-- Name: article_clusters article_clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.article_clusters
    ADD CONSTRAINT article_clusters_pkey PRIMARY KEY (id);


--
-- Name: article_entities article_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.article_entities
    ADD CONSTRAINT article_entities_pkey PRIMARY KEY (id);


--
-- Name: articles articles_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_pkey PRIMARY KEY (id);


--
-- Name: articles articles_url_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_url_key UNIQUE (url);


--
-- Name: feeds feeds_feed_url_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.feeds
    ADD CONSTRAINT feeds_feed_url_key UNIQUE (feed_url);


--
-- Name: feeds feeds_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.feeds
    ADD CONSTRAINT feeds_pkey PRIMARY KEY (id);


--
-- Name: pipeline_runs pipeline_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs
    ADD CONSTRAINT pipeline_runs_pkey PRIMARY KEY (id);


--
-- Name: stories stories_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.stories
    ADD CONSTRAINT stories_pkey PRIMARY KEY (id);


--
-- Name: article_clusters_last_seen_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX article_clusters_last_seen_idx ON public.article_clusters USING btree (last_seen_at DESC NULLS LAST);


--
-- Name: article_clusters_outlet_count_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX article_clusters_outlet_count_idx ON public.article_clusters USING btree (outlet_count DESC, last_seen_at DESC);


--
-- Name: article_clusters_simhash_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX article_clusters_simhash_idx ON public.article_clusters USING btree (canonical_simhash) WHERE (canonical_simhash IS NOT NULL);


--
-- Name: article_entities_article_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX article_entities_article_idx ON public.article_entities USING btree (article_id);


--
-- Name: article_entities_salience_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX article_entities_salience_idx ON public.article_entities USING btree (entity_text, salience DESC) WHERE (salience >= (0.5)::double precision);


--
-- Name: article_entities_type_text_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX article_entities_type_text_idx ON public.article_entities USING btree (entity_type, entity_text);


--
-- Name: articles_category_published_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_category_published_idx ON public.articles USING btree (category, published_at DESC NULLS LAST) WHERE (category IS NOT NULL);


--
-- Name: articles_cluster_id_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_cluster_id_idx ON public.articles USING btree (cluster_id) WHERE (cluster_id IS NOT NULL);


--
-- Name: articles_crawl_strategy_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_crawl_strategy_idx ON public.articles USING btree (crawl_strategy) WHERE (crawl_strategy IS NOT NULL);


--
-- Name: articles_domain_published_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_domain_published_idx ON public.articles USING btree (domain, published_at DESC NULLS LAST);


--
-- Name: articles_enriched_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_enriched_at_idx ON public.articles USING btree (enriched_at NULLS FIRST, published_at DESC NULLS LAST) WHERE (enriched_at IS NULL);


--
-- Name: articles_feed_published_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_feed_published_idx ON public.articles USING btree (feed_id, published_at DESC NULLS LAST);


--
-- Name: articles_iab_published_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_iab_published_idx ON public.articles USING btree (iab_tier1, published_at DESC NULLS LAST);


--
-- Name: articles_image_phash_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_image_phash_idx ON public.articles USING btree (image_phash) WHERE (image_phash IS NOT NULL);


--
-- Name: articles_is_crawled_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_is_crawled_idx ON public.articles USING btree (is_crawled) WHERE (is_crawled = false);


--
-- Name: articles_is_duplicate_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_is_duplicate_idx ON public.articles USING btree (is_duplicate) WHERE (is_duplicate = false);


--
-- Name: articles_lang_published_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_lang_published_idx ON public.articles USING btree (language_code, published_at DESC NULLS LAST);


--
-- Name: articles_sentiment_published_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_sentiment_published_idx ON public.articles USING btree (sentiment, published_at DESC NULLS LAST) WHERE (sentiment IS NOT NULL);


--
-- Name: articles_title_fts_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_title_fts_idx ON public.articles USING gin (to_tsvector('english'::regconfig, COALESCE(title, ''::text)));


--
-- Name: articles_title_simhash_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX articles_title_simhash_idx ON public.articles USING btree (title_simhash) WHERE (title_simhash IS NOT NULL);


--
-- Name: articles_url_hash_uidx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX articles_url_hash_uidx ON public.articles USING btree (url_hash);


--
-- Name: feeds_active_last_polled_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX feeds_active_last_polled_idx ON public.feeds USING btree (is_active, last_polled_at NULLS FIRST) WHERE (is_active = true);


--
-- Name: feeds_active_tier_priority_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX feeds_active_tier_priority_idx ON public.feeds USING btree (is_active, validation_tier, priority_score DESC NULLS LAST) WHERE (is_active = true);


--
-- Name: idx_articles_crawled_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_crawled_at ON public.articles USING btree (crawled_at DESC);


--
-- Name: idx_articles_feed_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_feed_id ON public.articles USING btree (feed_id);


--
-- Name: idx_articles_feed_id_crawled; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_feed_id_crawled ON public.articles USING btree (feed_id, crawled_at DESC);


--
-- Name: idx_articles_is_crawled; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_is_crawled ON public.articles USING btree (is_crawled) WHERE (is_crawled = false);


--
-- Name: idx_articles_published_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_published_at ON public.articles USING btree (published_at DESC);


--
-- Name: idx_articles_story_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_story_id ON public.articles USING btree (story_id);


--
-- Name: idx_articles_url_hash; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_articles_url_hash ON public.articles USING btree (url_hash);


--
-- Name: idx_clusters_embedding_hnsw; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_clusters_embedding_hnsw ON public.article_clusters USING hnsw (embedding_vec public.vector_cosine_ops) WITH (m='16', ef_construction='64');


--
-- Name: idx_clusters_last_seen_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_clusters_last_seen_at ON public.article_clusters USING btree (last_seen_at DESC);


--
-- Name: idx_clusters_outlet_set; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_clusters_outlet_set ON public.article_clusters USING gin (outlet_set);


--
-- Name: idx_feeds_is_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_feeds_is_active ON public.feeds USING btree (is_active) WHERE (is_active = true);


--
-- Name: idx_feeds_last_polled; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_feeds_last_polled ON public.feeds USING btree (last_polled_at);


--
-- Name: pipeline_runs_run_at_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX pipeline_runs_run_at_idx ON public.pipeline_runs USING btree (run_at DESC);


--
-- Name: feed_health _RETURN; Type: RULE; Schema: public; Owner: -
--

CREATE OR REPLACE VIEW public.feed_health AS
 SELECT f.id,
    f.feed_url,
    f.publisher_name,
    f.validation_tier,
    f.is_active,
    f.fail_count,
    f.articles_found,
    f.last_polled_at,
    f.last_success_at,
    f.priority_score,
    count(a.id) AS articles_last_24h
   FROM (public.feeds f
     LEFT JOIN public.articles a ON (((a.feed_id = f.id) AND (a.crawled_at >= (now() - '24:00:00'::interval)))))
  GROUP BY f.id
  ORDER BY f.priority_score DESC NULLS LAST;


--
-- Name: article_entities article_entities_article_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.article_entities
    ADD CONSTRAINT article_entities_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id) ON DELETE CASCADE;


--
-- Name: articles articles_feed_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_feed_id_fkey FOREIGN KEY (feed_id) REFERENCES public.feeds(id);


--
-- Name: articles anon_read_articles; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY anon_read_articles ON public.articles FOR SELECT TO anon USING ((is_duplicate = false));


--
-- Name: pipeline_runs anon_read_runs; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY anon_read_runs ON public.pipeline_runs FOR SELECT TO anon USING (true);


--
-- Name: article_clusters; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.article_clusters ENABLE ROW LEVEL SECURITY;

--
-- Name: article_entities; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.article_entities ENABLE ROW LEVEL SECURITY;

--
-- Name: articles; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.articles ENABLE ROW LEVEL SECURITY;

--
-- Name: feeds; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.feeds ENABLE ROW LEVEL SECURITY;

--
-- Name: pipeline_runs; Type: ROW SECURITY; Schema: public; Owner: -
--

ALTER TABLE public.pipeline_runs ENABLE ROW LEVEL SECURITY;

--
-- Name: article_clusters service_role_all; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY service_role_all ON public.article_clusters TO service_role USING (true) WITH CHECK (true);


--
-- Name: article_entities service_role_all; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY service_role_all ON public.article_entities TO service_role USING (true) WITH CHECK (true);


--
-- Name: articles service_role_full_access; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY service_role_full_access ON public.articles TO service_role USING (true) WITH CHECK (true);


--
-- Name: feeds service_role_full_access; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY service_role_full_access ON public.feeds TO service_role USING (true) WITH CHECK (true);


--
-- Name: pipeline_runs service_role_full_access; Type: POLICY; Schema: public; Owner: -
--

CREATE POLICY service_role_full_access ON public.pipeline_runs TO service_role USING (true) WITH CHECK (true);


--
-- PostgreSQL database dump complete
--


