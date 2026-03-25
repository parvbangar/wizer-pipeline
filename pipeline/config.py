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
load_dotenv()


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

# Written columns (pipeline updates these after each poll)
# These are the only columns the pipeline ever writes to feeds:
FEED_WRITE_COLS = [
    FEED_COL_FAIL_COUNT,
    FEED_COL_LAST_POLLED,
    FEED_COL_LAST_SUCCESS,
    FEED_COL_ARTICLES_FOUND,
    "is_active",     # set to False when feed goes dormant
]


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
MAX_CONCURRENT_FEEDS    = int(os.getenv("MAX_CONCURRENT_FEEDS", "3"))
MAX_CONCURRENT_ARTICLES = int(os.getenv("MAX_CONCURRENT_ARTICLES", "2"))


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

DORMANT_RECHECK_INTERVAL_DAYS = 7
# Dormant feeds get one retry every 7 days (they might have woken up).


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
SIMHASH_DISTANCE_THRESHOLD = 3    # bit distance ≤ 3 = near-duplicate title


# ─────────────────────────────────────────────────────────────────────────────
# CRAWLING
# ─────────────────────────────────────────────────────────────────────────────
CRAWL_TIMEOUT_SECONDS = int(os.getenv("CRAWL_TIMEOUT", "15"))
MAX_ARTICLE_BODY_CHARS = 80_000   # truncate very long articles before storing

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
