"""
enrichment/config.py
════════════════════
All configuration for the Layer 2 enrichment pipeline.

Same pattern as pipeline/config.py — every setting lives here.
No other file in the enrichment package hardcodes values.

HOW TO USE:
    from enrichment.config import ENRICH_BATCH_SIZE, CLUSTER_WINDOW_HOURS
"""

import os
from dotenv import load_dotenv

load_dotenv()          # tries .env file (standard)
load_dotenv(".env.local", override=False)  # fallback for local dev


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE — same credentials as Layer 1
# ─────────────────────────────────────────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE NAMES
# ─────────────────────────────────────────────────────────────────────────────
TABLE_ARTICLES  = "articles"
TABLE_ENTITIES  = "article_entities"
TABLE_CLUSTERS  = "article_clusters"


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
#
# The enrichment runner processes articles in batches to stay within Supabase
# API limits and avoid memory issues with large NLP models.
#
# ENRICH_BATCH_SIZE:
#   How many articles to fetch and process per run.
#   500 allows ~4,000 articles/day enriched (8 runs × 500). Lower if on free tier.
#   NLP is CPU-bound so concurrency doesn't help here — sequential is fine.
# ─────────────────────────────────────────────────────────────────────────────
ENRICH_BATCH_SIZE = int(os.getenv("ENRICH_BATCH_SIZE", "500"))


# ─────────────────────────────────────────────────────────────────────────────
# STEP: TEXT STATS
# ─────────────────────────────────────────────────────────────────────────────
WORDS_PER_MINUTE = 200   # average adult reading speed used for reading_time_mins


# ─────────────────────────────────────────────────────────────────────────────
# STEP: LANGUAGE DETECTION
#
# langdetect needs a minimum amount of text to be accurate.
# Short snippets (< 20 chars) produce unreliable results — skip them.
# ─────────────────────────────────────────────────────────────────────────────
LANG_DETECT_MIN_CHARS = 20


# ─────────────────────────────────────────────────────────────────────────────
# STEP: NAMED ENTITY RECOGNITION
#
# SPACY_MODEL:
#   'en_core_web_sm'  — 12 MB, fast, good for English news (default)
#   'en_core_web_md'  — 43 MB, better accuracy, worth it on paid infra
#   'en_core_web_lg'  — 741 MB, best accuracy, use only if you have fast storage
#
#   Download command:  python -m spacy download en_core_web_sm
#
# NER_MIN_SALIENCE:
#   Entities below this salience score are not stored.
#   Keeps the article_entities table lean — filters out
#   incidental mentions like "he said", "the company", etc.
# ─────────────────────────────────────────────────────────────────────────────
SPACY_MODEL       = os.getenv("SPACY_MODEL", "en_core_web_sm")
NER_MIN_SALIENCE  = 0.1

# Entity types we care about for Indian news
# spaCy label → stored entity_type
ENTITY_TYPE_MAP = {
    "PERSON":   "PERSON",    # politicians, celebrities, athletes
    "ORG":      "ORG",       # parties, companies, institutions
    "GPE":      "GPE",       # cities, states, countries
    "LOC":      "GPE",       # map LOC → GPE for simplicity
    "EVENT":    "EVENT",     # named events (Budget 2024, IPL, G20)
    "PRODUCT":  "PRODUCT",   # apps, platforms, products
    "LAW":      "LAW",       # named laws and acts (CAA, RTI, Article 370)
    "NORP":     "ORG",       # nationalities/political groups → treat as ORG
    "WORK_OF_ART": "PRODUCT",
}
# Types NOT in the map (DATE, TIME, MONEY, PERCENT, CARDINAL, etc.) are skipped


# ─────────────────────────────────────────────────────────────────────────────
# STEP: KEYWORD EXTRACTION (YAKE)
#
# YAKE (Yet Another Keyword Extractor) is unsupervised and language-agnostic —
# it works on Hindi and Tamil articles as well as English, which matters for
# the Indian market.
#
# MAX_KEYWORDS:    top N keywords to store per article
# KEYWORD_MAX_NGRAM: max words in a keyword phrase (3 = "Narendra Modi budget")
# KEYWORD_DEDUP_THRESHOLD: lower = more diversity between keywords (0.9 = loose)
# ─────────────────────────────────────────────────────────────────────────────
MAX_KEYWORDS             = int(os.getenv("MAX_KEYWORDS", "10"))
KEYWORD_MAX_NGRAM        = 3
KEYWORD_DEDUP_THRESHOLD  = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# STEP: ARTICLE CATEGORY CLASSIFICATION
#
# Rule-based keyword classifier for Indian news categories.
# More granular than the feed-level iab_tier1/iab_tier2.
#
# WHY RULE-BASED (not ML)?
#   - No model download, no GPU, fast
#   - Deterministic — easier to audit and fix
#   - Indian-specific keywords you control
#   - Can upgrade to zero-shot classifier later without changing the interface
#
# Category priority: categories listed first win on ties.
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_RULES: dict[str, list[str]] = {
    "cricket": [
        "ipl", "bcci", "cricket", "test match", "odi", "t20", "virat kohli",
        "rohit sharma", "ms dhoni", "sachin", "batting", "bowling", "wicket",
        "innings", "run chase", "century", "world cup cricket",
    ],
    "politics": [
        "bjp", "congress", "aap", "modi", "rahul gandhi", "parliament",
        "lok sabha", "rajya sabha", "election", "cm ", "chief minister",
        "minister", "mla", "mp ", "governor", "president of india",
        "political party", "vote", "constituency", "bypoll", "cabinet",
        "nda", "upa", "indi alliance",
    ],
    "business": [
        "sensex", "nse", "bse", "nifty", "rbi", "rupee", "gdp", "inflation",
        "stock market", "shares", "ipo", "sebi", "mutual fund", "revenue",
        "profit", "quarterly results", "startup", "unicorn", "funding",
        "acquisition", "merger", "budget", "fiscal", "import", "export",
        "gst", "trade", "economic", "economy", "retail", "fdi",
    ],
    "entertainment": [
        "bollywood", "box office", "film", "movie", "actor", "actress",
        "director", "ott", "netflix", "amazon prime", "hotstar", "web series",
        "music", "album", "song", "celebrity", "award", "filmfare",
        "srk", "shah rukh", "salman", "deepika", "ranveer", "alia",
    ],
    "technology": [
        "artificial intelligence", "ai ", "machine learning", "startup",
        "tech ", "software", "app ", "smartphone", "5g", "cyber", "hack",
        "data breach", "isro", "satellite", "space", "elon musk",
        "google", "meta", "apple", "microsoft", "amazon", "chip",
    ],
    "sports": [
        "football", "fifa", "isl", "hockey", "badminton", "tennis",
        "olympics", "commonwealth games", "asian games", "wrestling",
        "kabaddi", "pro kabaddi", "formula 1", "f1", "chess", "athlete",
        "gold medal", "silver medal",
    ],
    "health": [
        "hospital", "doctor", "medicine", "disease", "covid", "vaccine",
        "health ministry", "aiims", "cancer", "diabetes", "surgery",
        "outbreak", "epidemic", "who ", "drug", "clinical trial", "patient",
    ],
    "education": [
        "school", "college", "university", "jee", "neet", "upsc", "ias",
        "board exam", "cbse", "icse", "ugc", "iit", "iim", "scholarship",
        "syllabus", "admission", "student", "education policy", "nep",
    ],
    "crime": [
        "murder", "rape", "assault", "arrested", "police", "fir", "court",
        "judge", "verdict", "convicted", "bail", "chargesheet", "cbi",
        "ed ", "enforcement directorate", "scam", "fraud", "crime",
        "gangster", "kidnap", "robbery",
    ],
    "environment": [
        "climate", "pollution", "air quality", "aqi", "flood", "drought",
        "cyclone", "earthquake", "wildlife", "forest", "deforestation",
        "carbon", "renewable", "solar", "wind energy", "green",
        "environment ministry",
    ],
    "world": [
        "united states", "usa ", "china", "pakistan", "russia", "ukraine",
        "europe", "nato", "un ", "united nations", "g20", "g7",
        "foreign ministry", "diplomatic", "sanctions", "border dispute",
        "international", "global",
    ],
}
# Fallback when no category matches
CATEGORY_DEFAULT = "general"


# ─────────────────────────────────────────────────────────────────────────────
# STEP: IMAGE DOWNLOAD + PERCEPTUAL HASHING
#
# IMAGE_DOWNLOAD_TIMEOUT: seconds to wait for image HTTP response
# IMAGE_MAX_BYTES:        skip images larger than this (5 MB)
#                         Avoids downloading hi-res photos just to hash them.
# IMAGE_HASH_SIZE:        pHash grid size. 8 = 64-bit hash (standard).
#                         Higher = more detail, larger hash.
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_DOWNLOAD_TIMEOUT = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "10"))
IMAGE_MAX_BYTES        = 5 * 1024 * 1024   # 5 MB
IMAGE_HASH_SIZE        = 8                  # produces 64-bit pHash


# ─────────────────────────────────────────────────────────────────────────────
# STEP: STORY CLUSTERING
#
# Two articles belong to the same cluster if they're about the same event.
# We detect this using two signals:
#
#   1. Entity overlap  — share ≥ CLUSTER_MIN_SHARED_ENTITIES named entities
#   2. Title similarity — title SimHash distance ≤ CLUSTER_SIMHASH_THRESHOLD
#
# CLUSTER_SCORE_THRESHOLD:
#   Combined score (entity Jaccard + title similarity) needed to join a cluster.
#   0.35 is calibrated for Indian news wire syndication patterns.
#   Raise to 0.5 if you're getting too many false clusters.
#
# CLUSTER_WINDOW_HOURS:
#   Only cluster against articles published in the last N hours.
#   48h covers breaking news lifecycles (most Indian stories resolve in 24-36h).
#
# CLUSTER_MAX_CANDIDATES:
#   Max number of recent clusters to compare against per article.
#   Prevents slow O(n²) comparisons when there are thousands of active clusters.
# ─────────────────────────────────────────────────────────────────────────────
CLUSTER_WINDOW_HOURS        = int(os.getenv("CLUSTER_WINDOW_HOURS", "48"))
CLUSTER_MIN_SHARED_ENTITIES = 2
CLUSTER_SIMHASH_THRESHOLD   = 5     # bit distance ≤ 5 = titles are very similar
CLUSTER_SCORE_THRESHOLD     = 0.35
CLUSTER_MAX_CANDIDATES      = 500   # max clusters to compare per article


# ─────────────────────────────────────────────────────────────────────────────
# STEP: SENTIMENT ANALYSIS (VADER)
#
# VADER is English-only. For non-English articles (Hindi, Tamil, etc.)
# sentiment and sentiment_score are stored as NULL.
# Upgrade path: replace with a multilingual model (XLM-R, MuRIL) later.
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_POSITIVE_THRESHOLD =  0.05   # compound score above this = positive
SENTIMENT_NEGATIVE_THRESHOLD = -0.05   # compound score below this = negative
# Between these thresholds = neutral
SENTIMENT_ENGLISH_ONLY_LANGS = {"en", "en-us", "en-gb", "en-in"}
