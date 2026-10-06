"""
config.py
═════════
Central configuration for the entire pipeline.

WHAT IS THIS FILE FOR?
  Every "magic number" or setting lives here so you only ever change
  one file when tuning the pipeline.  No other file hardcodes values.

HOW TO USE IT?
  Every other module does:   from pipeline.config import SOME_SETTING
  For secrets (Supabase keys) it reads from your .env file automatically.
"""

import os
from dotenv import load_dotenv

# Load .env file so os.getenv() can find SUPABASE_URL etc.
load_dotenv()          # tries .env file (standard)
load_dotenv(".env.local", override=False)  # fallback for local dev


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE CONNECTION
# These come from your .env file — never hardcode them here.
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")
# ↑ Use the SERVICE KEY (long JWT), NOT the anon key.
#   Service key bypasses Row Level Security so the pipeline can write freely.


# ─────────────────────────────────────────────────────────────────────────────
# YOUR EXACT DATABASE TABLE NAMES
# ─────────────────────────────────────────────────────────────────────────────
TABLE_FEEDS    = "feeds"
TABLE_ARTICLES = "articles"
TABLE_RUNS     = "pipeline_runs"   # we create this — stores one row per run


# ─────────────────────────────────────────────────────────────────────────────
# YOUR EXACT FEEDS TABLE COLUMN NAMES
# The pipeline reads these columns from every feed row.
# ─────────────────────────────────────────────────────────────────────────────
# Read columns (already exist in your DB)
FEED_COL_ID              = "id"
FEED_COL_URL             = "feed_url"
FEED_COL_FINAL_URL       = "final_url"
FEED_COL_DOMAIN          = "domain"
FEED_COL_PUBLISHER       = "publisher_name"
FEED_COL_TIER            = "validation_tier"   # kept for reference / monitoring views
FEED_COL_CADENCE         = "update_cadence"    # breaking_news / multiple_daily / daily / several_weekly / weekly / monthly / unknown
FEED_COL_LANGUAGE        = "language_code"
FEED_COL_COUNTRY         = "country_code"
FEED_COL_IAB1            = "iab_tier1"
FEED_COL_IAB2            = "iab_tier2"
FEED_COL_HAS_PAYWALL     = "has_paywall"
FEED_COL_IS_ACTIVE       = "is_active"
FEED_COL_POLL_INTERVAL   = "poll_interval_mins"
FEED_COL_PRIORITY        = "priority_score"
FEED_COL_FAIL_COUNT      = "fail_count"        # incremented on each error
FEED_COL_LAST_POLLED     = "last_polled_at"    # timestamp of last poll attempt
FEED_COL_LAST_SUCCESS    = "last_success_at"   # timestamp of last successful poll
FEED_COL_ARTICLES_FOUND  = "articles_found"    # running total of articles found
FEED_COL_CREATED         = "created_at"        # dormancy grace-period fallback (optional column)
FEED_COL_LAST_NEW        = "last_new_article_at"  # advanced ONLY when a poll inserts >0 articles
                                                  # (optional: added by docs/ingestion_fixes_migration.sql)
FEED_COL_DISABLED_REASON = "disabled_reason"   # 'dormant' | 'errors' | NULL (optional column)

# feeds.disabled_reason values — makes the two "is_active=False" states distinguishable
DISABLED_DORMANT = "dormant"   # no new articles for DORMANCY_DAYS; re-polled weekly
DISABLED_ERRORS  = "errors"    # MAX_ERRORS consecutive failures; manual reset only

# Columns that may not exist on older deployments.  db.py degrades gracefully
# (retries without them) if PostgREST says they are missing.
FEED_OPTIONAL_COLS = (FEED_COL_CREATED, FEED_COL_LAST_NEW, FEED_COL_DISABLED_REASON)

# Written columns (pipeline updates these after each poll)
# These are the only columns the pipeline ever writes to feeds:
FEED_WRITE_COLS = [
    FEED_COL_FAIL_COUNT,
    FEED_COL_LAST_POLLED,
    FEED_COL_LAST_SUCCESS,
    FEED_COL_ARTICLES_FOUND,
    FEED_COL_LAST_NEW,
    FEED_COL_DISABLED_REASON,
    "is_active",     # set to False when feed goes dormant or error-disabled
]

# feeds.language_name is NOT guaranteed to exist (docs/migration.sql creates it,
# but older deployments may lack it and selecting a missing column makes
# PostgREST reject the whole query).  So we never select it; we derive the
# human-readable articles.language from feeds.language_code instead.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "or": "Odia",
    "as": "Assamese", "ne": "Nepali", "sa": "Sanskrit", "sd": "Sindhi",
    "fr": "French", "de": "German", "es": "Spanish", "pt": "Portuguese",
    "ar": "Arabic", "zh": "Chinese", "ja": "Japanese", "ru": "Russian",
}


# ─────────────────────────────────────────────────────────────────────────────
# YOUR EXACT ARTICLES TABLE COLUMN NAMES
# ─────────────────────────────────────────────────────────────────────────────
ART_COL_ID            = "id"
ART_COL_FEED_ID       = "feed_id"
ART_COL_URL           = "url"
ART_COL_URL_HASH      = "url_hash"        # MurmurHash3 of normalised URL → bigint
ART_COL_TITLE         = "title"
ART_COL_TITLE_SIMHASH = "title_simhash"   # SimHash of title for near-dedup
ART_COL_DESCRIPTION   = "description"
ART_COL_FULL_TEXT     = "full_text"
ART_COL_TOP_IMAGE     = "top_image_url"
ART_COL_AUTHOR        = "author"
ART_COL_PUBLISHED     = "published_at"
ART_COL_CRAWLED       = "crawled_at"
ART_COL_LANGUAGE      = "language"
ART_COL_COUNTRY       = "country_code"
ART_COL_OG_TAGS       = "og_tags"         # jsonb — full Open Graph tag dump
ART_COL_IS_CRAWLED    = "is_crawled"      # True once full-text crawl succeeded
ART_COL_IS_DUPLICATE  = "is_duplicate"    # True if this is a near-duplicate
ART_COL_STORY_ID      = "story_id"        # future: cluster ID for story grouping
ART_COL_PROPENSITY    = "propensity_score"# future: virality score
ART_COL_CREATED       = "created_at"
ART_COL_FEED_URL      = "feed_url"        # denormalised for easy querying
ART_COL_DOMAIN        = "domain"
ART_COL_PUBLISHER     = "publisher_name"
ART_COL_IAB1          = "iab_tier1"
ART_COL_IAB2          = "iab_tier2"
ART_COL_LANG_CODE     = "language_code"


# ─────────────────────────────────────────────────────────────────────────────
# POLLING INTERVALS PER UPDATE CADENCE
#
# The pipeline groups feeds by their update_cadence column and polls each
# group on a separate GitHub Actions schedule.
# poll_interval_mins in your DB overrides these defaults when set.
# ─────────────────────────────────────────────────────────────────────────────
CADENCE_POLL_INTERVALS: dict[str, int] = {
    "breaking_news":   60,    # 60 minutes  — live news desks, wire agencies
    "multiple_daily":  180,   # 3 hours     — major outlets publishing 5+ times/day
    "daily":           720,   # 12 hours    — once-a-day publishers
    "several_weekly":  1440,  # 24 hours    — a few posts per week
    "weekly":          1440,  # 24 hours    — weekly newsletters / digests
    "monthly":         1440,  # 24 hours    — monthly publications
    "unknown":         720,   # 12 hours    — unclassified; treat conservatively
}

# How many feeds to poll at the same time
# (asyncio semaphore — don't set above 30 or remote servers start blocking you)
# Defaults are 15 feeds / 5 article crawls (the ingest_*.yml workflows override
# them per cadence via env vars).
MAX_CONCURRENT_FEEDS    = int(os.getenv("MAX_CONCURRENT_FEEDS", "15"))
MAX_CONCURRENT_ARTICLES = int(os.getenv("MAX_CONCURRENT_ARTICLES", "5"))

# Size of the thread pool installed as the asyncio loop's default executor.
# The stock default (min(32, cpu+4), ~6 on a 2-vCPU runner) is SMALLER than
# MAX_CONCURRENT_FEEDS, so blocking fetches queued behind each other and one
# hung server could stall the whole run.  We need a slot per concurrent feed
# fetch + a slot per concurrent crawl + headroom for short DB calls.
EXECUTOR_MAX_WORKERS = int(os.getenv(
    "EXECUTOR_MAX_WORKERS",
    str(max(8, MAX_CONCURRENT_FEEDS + MAX_CONCURRENT_ARTICLES + 4)),
))

# ── RSS feed fetching (poller._fetch_rss_blocking) ───────────────────────────
# feedparser.parse(url) has NO timeout, so we download the bytes ourselves.
FEED_FETCH_TIMEOUT_SECONDS  = float(os.getenv("FEED_FETCH_TIMEOUT", "20"))   # per socket op (connect / each recv)
FEED_FETCH_DEADLINE_SECONDS = float(os.getenv("FEED_FETCH_DEADLINE", "45"))  # wall-clock cap per request (slow-drip guard)
FEED_MAX_BYTES              = int(os.getenv("FEED_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB

# ── Feed scheduling ──────────────────────────────────────────────────────────
# A feed is due when elapsed >= interval * FEED_DUE_TOLERANCE.  See
# db._is_feed_due for why exact equality made feeds poll every 2 intervals.
FEED_DUE_TOLERANCE = float(os.getenv("FEED_DUE_TOLERANCE", "0.9"))


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER — DORMANCY DETECTION
#
# What is the circuit breaker?
#   If a feed keeps failing, the pipeline stops trying to fetch it rather than
#   wasting resources.  This is called "opening the circuit".
#
# Two failure modes:
#   1. ERROR STREAK  — fail_count >= MAX_ERRORS  → mark is_active=False immediately
#   2. DORMANCY      — no new articles for DORMANCY_DAYS → mark is_active=False
# ─────────────────────────────────────────────────────────────────────────────
MAX_ERRORS_BEFORE_DISABLE = int(os.getenv("MAX_ERRORS", "5"))
# After 5 consecutive fetch/parse failures the feed is disabled.

DORMANCY_DAYS = int(os.getenv("DORMANCY_DAYS", "30"))
# If a feed hasn't produced any new articles in 30 days, mark it dormant.

DORMANT_RECHECK_INTERVAL_DAYS = int(os.getenv("DORMANT_RECHECK_INTERVAL_DAYS", "7"))
# Dormant feeds (feeds.disabled_reason='dormant') get one retry per interval
# (they might have woken up).  Error-disabled feeds are never re-polled.


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION
#
# Two layers of dedup:
#   Layer 1 (EXACT)   — MurmurHash3 of the normalised URL
#                        Catches 100% of exact-same-URL duplicates
#   Layer 2 (NEAR)    — SimHash of the title (64-bit fingerprint)
#                        Catches rephrased re-posts of the same story
#                        (e.g. PTI wire story republished by 10 outlets)
# ─────────────────────────────────────────────────────────────────────────────
HASH_SEED              = 42       # MurmurHash3 seed — NEVER change after first run

# TRANSITION: dedup.normalise_url() changed (http/https collapse, fewer stripped
# params), which changes future url_hash values.  While this is on, an article
# counts as already-seen if EITHER its new hash OR its legacy hash exists.
# Safe to switch off (and delete dedup._legacy_normalise_url) once every
# pre-change article has aged out of the feeds' RSS windows — ~90 days after
# the deploy that introduced it.
LEGACY_URL_HASH_CHECK = os.getenv("LEGACY_URL_HASH_CHECK", "1") not in ("0", "false", "False")

# PostgREST caps every response at max_rows (1000 on Supabase) regardless of
# .limit(), so all large reads are paginated with .range() in pages this size.
DB_PAGE_SIZE = 1000
SIMHASH_DISTANCE_THRESHOLD = 3    # bit distance ≤ 3 = near-duplicate title


# ─────────────────────────────────────────────────────────────────────────────
# ARTICLE TABLE SIZE CAP
# After each pipeline run, oldest articles are pruned to keep the table lean.
# ─────────────────────────────────────────────────────────────────────────────
# DISABLED BY DEFAULT (0) since 2026-10-03.
#   The old Python pruning never actually worked (PostgREST's 1000-row cap), so
#   production grew to 2.98M rows. The fixed pruning would delete ~1.08M of the
#   oldest articles on its first run. That history is valuable (model training
#   data, point-in-time news for research), so deletion stays off until an
#   archive-then-prune step exists that only removes rows already exported.
#   To re-enable (deletes rows!): ARTICLE_HARD_LIMIT=2000000 ARTICLE_PRUNE_TARGET=1900000
ARTICLE_HARD_LIMIT    = int(os.getenv("ARTICLE_HARD_LIMIT",  "0"))
ARTICLE_PRUNE_TARGET  = int(os.getenv("ARTICLE_PRUNE_TARGET", "0"))
# Prune to ARTICLE_PRUNE_TARGET (95% of the hard limit) so the trigger doesn't
# fire on every single run.


# ─────────────────────────────────────────────────────────────────────────────
# CRAWLING
# ─────────────────────────────────────────────────────────────────────────────
CRAWL_TIMEOUT_SECONDS = int(os.getenv("CRAWL_TIMEOUT", "15"))
MAX_ARTICLE_BODY_CHARS = 80_000   # truncate very long articles before storing


# ─────────────────────────────────────────────────────────────────────────────
# HAND-OFF TO CLUSTERING / ENRICHMENT (pipeline/handoff.py)
#
# WIZER_HANDOFF_PATH: where run_pipeline writes the articles it inserted, WITH
#   their crawled body, for the processing jobs (the ingest workflows upload
#   it as an artifact). Unset = no hand-off file (local runs).
# STORE_FULL_TEXT: also write full_text into Postgres. Default: only when there
#   is no hand-off — with a hand-off the body travels in the file instead,
#   which keeps the Micro database small (full_text was ~60 % of it).
# FEED_POLL_BATCH: record feed poll outcomes with one wizer_record_feed_polls
#   call per this many feeds at the end of the run, instead of up to three
#   calls per feed (docs/bulk_io_migration.sql).
# ─────────────────────────────────────────────────────────────────────────────
HANDOFF_PATH    = os.getenv("WIZER_HANDOFF_PATH", "").strip()
STORE_FULL_TEXT = os.getenv("STORE_FULL_TEXT", "" if HANDOFF_PATH else "1").lower() in ("1", "true", "yes")
FEED_POLL_BATCH = int(os.getenv("FEED_POLL_BATCH", "200"))
# CRAWL_AT_INGEST: fetch each article's page during ingestion (the original
#   design). With a hand-off the crawl is DEFERRED to the processing runners
#   (pipeline.crawler.crawl_record), so ingestion only discovers: on 2026-10-06
#   inline crawling let a 58-min breaking_news run poll 456 of 1,365 due feeds.
# INGEST_TIME_BUDGET_MINUTES: stop starting new feeds after this long (0 = no
#   limit); feeds not reached stay due for the next run. Set below the job's
#   timeout so the run always ends cleanly and records what it did.
CRAWL_AT_INGEST = os.getenv("CRAWL_AT_INGEST", "" if HANDOFF_PATH else "1").lower() in ("1", "true", "yes")
INGEST_TIME_BUDGET_MINUTES = float(os.getenv("INGEST_TIME_BUDGET_MINUTES", "0"))

# Per-article wall-clock budget across ALL fetch strategies/retries/backoff.
# Without it, 4 strategies x 3 retries x 15 s timeouts could pin one worker
# thread for several minutes on a single dead article.
CRAWL_ARTICLE_DEADLINE_SECONDS = float(os.getenv("CRAWL_ARTICLE_DEADLINE", "60"))

# Per-run, per-domain fail-fast: after this many CONSECUTIVE articles from one
# domain on which every fetch strategy failed, stop making HTTP requests to that
# domain for the rest of the run and store RSS-only text instead.
CRAWL_DOMAIN_FAIL_THRESHOLD = int(os.getenv("CRAWL_DOMAIN_FAIL_THRESHOLD", "5"))

# Default bot user agent — identifies us honestly
USER_AGENT = (
    "Mozilla/5.0 (compatible; NewsIngestBot/1.0; "
    "+https://github.com/your-org/news-pipeline)"
)

# Googlebot user agent — many paywalled news sites whitelist Googlebot
# so their content gets indexed by Google.  Used as the primary fallback
# when the default UA fails or hits a paywall.
GOOGLEBOT_UA = (
    "Mozilla/5.0 (compatible; Googlebot/2.1; "
    "+http://www.google.com/bot.html)"
)

# Retry / fallback settings
MAX_FETCH_RETRIES    = int(os.getenv("MAX_FETCH_RETRIES", "3"))
RETRY_DELAY_SECONDS  = float(os.getenv("RETRY_DELAY", "1.0"))
ARCHIVE_ORG_TIMEOUT  = int(os.getenv("ARCHIVE_ORG_TIMEOUT", "8"))

# Known paywalled domains — the crawler STILL attempts every strategy
# (Googlebot, AMP, Wayback) on these; this list is only used for logging
# and to skip the default UA attempt (which would definitely fail).
PAYWALLED_DOMAINS: frozenset = frozenset({
    # Indian paywalled
    "thehindu.com", "hindustantimes.com", "financialexpress.com",
    "livemint.com", "theprint.in", "indianexpress.com",
    "telegraphindia.com", "deccanherald.com", "tribuneindia.com",
    # International paywalled
    "ft.com", "wsj.com", "nytimes.com", "bloomberg.com",
    "economist.com", "thetimes.co.uk", "telegraph.co.uk",
    "washingtonpost.com", "newyorker.com",
})
