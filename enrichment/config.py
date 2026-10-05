"""
enrichment/config.py
════════════════════
All configuration for the Layer 2 enrichment pipeline.

Same pattern as pipeline/config.py — every setting lives here.
No other file in the enrichment package hardcodes values.

HOW TO USE:
    from enrichment.config import ENRICH_BATCH_SIZE, ENRICH_MAX_AGE_HOURS
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
TABLE_RUNS      = "enrichment_runs"


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
#
# ENRICH_MIN_WORD_COUNT:
#   Articles below this word count (wire briefs, snippets, failed crawls with
#   only an RSS description) still get every step that works on a headline +
#   description — language, sentiment, NER, category, tags, summary, image —
#   but skip keyword extraction, which only yields noise on a few words.
#   Typical Indian news snippet / wire brief = 30–60 words.
# ─────────────────────────────────────────────────────────────────────────────
# Raised from 500 → 1000 after Supabase Pro upgrade.
ENRICH_BATCH_SIZE     = int(os.getenv("ENRICH_BATCH_SIZE",     "1000"))
ENRICH_MIN_WORD_COUNT = int(os.getenv("ENRICH_MIN_WORD_COUNT", "50"))

# Only enrich articles INGESTED within this many hours (0 = no limit).
# Default 0: every ingested article is enriched. With a gate, any shortfall in
# enrichment throughput silently drops the oldest articles for good.
ENRICH_MAX_AGE_HOURS  = int(os.getenv("ENRICH_MAX_AGE_HOURS",  "0"))

# Languages to fully enrich (NER, sentiment, keywords, classification).
# Articles in other languages get text_stats + language_detected only, then marked done.
# Set to empty set to enrich all languages.
# "en" covers en, en-in, en-us etc — the gate checks startswith in runner.
_supported = os.getenv("ENRICH_SUPPORTED_LANGUAGES", "en,hi")
ENRICH_SUPPORTED_LANGUAGES: set[str] = (
    {lang.strip() for lang in _supported.split(",") if lang.strip()}
    if _supported.strip() else set()
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP: TEXT STATS
# ─────────────────────────────────────────────────────────────────────────────
WORDS_PER_MINUTE = 200   # average adult reading speed used for reading_time_mins


# ─────────────────────────────────────────────────────────────────────────────
# STEP: LANGUAGE DETECTION
#
# lingua needs a minimum amount of text to be accurate.
# Short snippets (< 20 chars) produce unreliable results — skip them.
# ─────────────────────────────────────────────────────────────────────────────
LANG_DETECT_MIN_CHARS = 20


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY CLASSIFICATION — mDeBERTa zero-shot NLI
#
# Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
#   - State-of-the-art multilingual NLI model
#   - Understands all Indian regional languages natively
#   - Zero-shot: no training data needed, just define category labels
#   - ~560 MB download, cached in ~/.cache/huggingface/hub after first run
#
# CLASSIFY_CONFIDENCE_THRESHOLD:
#   Minimum score from the model to accept a category.
#   Below this → return 'general'. Prevents low-confidence mislabelling.
# ─────────────────────────────────────────────────────────────────────────────
DEBERTA_MODEL               = os.getenv("DEBERTA_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
# Minimum independent entailment probability for the best category label;
# below it the article is "general". 0.30 is the measured sweet spot on the
# labelled set (68.1 % accuracy with the shipped input + rules; 0.50 → lower).
# Previous values (0.15 / 0.20 / 0.35) applied to a softmax over 12 long
# labels, a different scale — see enrichment/steps/classifier.py.
CLASSIFY_CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFY_CONFIDENCE_THRESHOLD", "0.30"))
# Minimum per-tag sigmoid score to include a tag in ai_tag.
# multi_label=True uses sigmoid (not softmax), so scores are independent.
# Raised 0.25 → 0.35: at 0.25 "conflict" and "crime" fired on sports/general
# articles (66 observed false-positive co-occurrences in production).
CLASSIFY_TAG_THRESHOLD        = float(os.getenv("CLASSIFY_TAG_THRESHOLD",        "0.35"))


# ─────────────────────────────────────────────────────────────────────────────
# STEP: NAMED ENTITY RECOGNITION (spaCy)
#
# Two models based on detected language:
#   SPACY_MODEL              — English articles
#   SPACY_MULTILINGUAL_MODEL — all other languages (Hindi, Tamil, etc.)
#
# NER_MIN_SALIENCE:
#   Entities below this salience score are not stored. Keeps article_entities
#   lean — filters incidental mentions.
# ─────────────────────────────────────────────────────────────────────────────
# Upgraded from en_core_web_sm (12 MB) → en_core_web_md (43 MB) after Supabase Pro upgrade.
# Requires: python -m spacy download en_core_web_md
#           python -m spacy download xx_ent_wiki_sm
SPACY_MODEL              = os.getenv("SPACY_MODEL",             "en_core_web_md")
SPACY_MULTILINGUAL_MODEL = os.getenv("SPACY_MULTILINGUAL_MODEL", "xx_ent_wiki_sm")
NER_MIN_SALIENCE         = 0.1

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
# STEP: SENTIMENT ANALYSIS (multilingual transformer)
#
# Model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
#   - Multilingual: EN, Hindi, Tamil, Telugu, Bengali, and 100+ others
#   - ~268 MB, CPU-fast (distilled from mDeBERTa)
#   - Upgrade from VADER: Hindi/Tamil/Telugu articles now get sentiment
#     instead of NULL — covers the 40-50% of our corpus that was skipped
#
# sentiment_score = positive_prob - negative_prob → −1.0 to +1.0
# (matches VADER compound score semantics for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_MULTILINGUAL_MODEL = os.getenv(
    "SENTIMENT_MULTILINGUAL_MODEL",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
)
# Class-probability thresholds for the 3-way sentiment classifier.
# If neither positive nor negative class reaches its threshold, label = "neutral".
# 0.40 means the model must be at least moderately sure before committing to
# positive or negative — prevents "max wins by 1%" cases from suppressing neutral.
# Previously these were compound-score thresholds (±0.05) but were unused.
SENTIMENT_POSITIVE_THRESHOLD = 0.40
SENTIMENT_NEGATIVE_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# WORK QUEUE (docs/enrichment_queue_migration.sql)
#
# ENRICH_CLAIM_LEASE_MINUTES:
#   A claimed article is reserved for this long. If the runner dies, the claim
#   expires and another run picks the article up. MUST exceed the longest run
#   (enrichment.yml timeout-minutes = 120), or a slow-but-alive run could have its
#   articles re-claimed underneath it.
#
# ENRICH_MAX_ATTEMPTS:
#   Claims per article before it becomes a dead letter. An article whose
#   processing kills the process (OOM, segfault in a native lib) would
#   otherwise crash every run forever.
#
# ENRICH_RETRY_HOURS / ENRICH_MAX_RETRIES:
#   A dead letter is not abandoned: it is retried once every
#   ENRICH_RETRY_HOURS, up to ENRICH_MAX_RETRIES more times (a transient
#   outage — DB, a model download — must not cost an article its enrichment).
#   Its enriched_at stays NULL the whole time. After the last retry it shows
#   up as enrichment_queue_health.given_up.
#
# ENRICH_TIME_BUDGET_MINUTES:
#   Stop taking new articles after this long and release the unprocessed rest
#   of the batch, so a run never gets killed by the job timeout mid-article.
#   0 = no budget (local runs).
# ─────────────────────────────────────────────────────────────────────────────
ENRICH_CLAIM_LEASE_MINUTES = int(os.getenv("ENRICH_CLAIM_LEASE_MINUTES", "150"))
ENRICH_MAX_ATTEMPTS        = int(os.getenv("ENRICH_MAX_ATTEMPTS",        "3"))
ENRICH_RETRY_HOURS         = int(os.getenv("ENRICH_RETRY_HOURS",         "24"))
ENRICH_MAX_RETRIES         = int(os.getenv("ENRICH_MAX_RETRIES",         "7"))
ENRICH_TIME_BUDGET_MINUTES = float(os.getenv("ENRICH_TIME_BUDGET_MINUTES", "0"))


# ─────────────────────────────────────────────────────────────────────────────
# STORY CLUSTERING (docs/clustering_v2_migration.sql, docs/CLUSTERING.md)
#
# Every number below was calibrated on hand-labelled pairs of live Indian
# headlines — see docs/CLUSTERING.md §Calibration and tools/cluster_eval/.
# Re-run the calibration before changing the model or a threshold.
#
# CLUSTER_EMBEDDING_MODEL      sentence-embedding model (must be 768-dim and
#                              listed in enrichment/steps/embedding.py)
# CLUSTER_JOIN_THRESHOLD       cosine(article, cluster centroid) to join outright
# CLUSTER_GRAY_THRESHOLD       lower bound of the "gray zone": joins only with
#                              corroborating entity or image evidence
# CLUSTER_ANCHOR_THRESHOLD     cosine(article, cluster SEED) floor — stops a
#                              centroid drifting from one story into the next
# CLUSTER_MIN_SHARED_ENTITIES  salient entities in common that count as evidence
# CLUSTER_IMAGE_MAX_DISTANCE   pHash Hamming distance that counts as "same photo"
# CLUSTER_MAX_GAP_HOURS        an article may join a story whose latest article
#                              is at most this far away in time
# CLUSTER_MAX_SPAN_HOURS       a story never spans more than this end-to-end
# CLUSTER_MERGE_THRESHOLD      centroid↔centroid cosine for the maintenance merge
# ─────────────────────────────────────────────────────────────────────────────
CLUSTERING_ENABLED          = os.getenv("CLUSTERING_ENABLED", "true").lower() not in ("0", "false", "no")
CLUSTER_EMBEDDING_MODEL     = os.getenv("CLUSTER_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CLUSTER_JOIN_THRESHOLD      = float(os.getenv("CLUSTER_JOIN_THRESHOLD",      "0.85"))
CLUSTER_GRAY_THRESHOLD      = float(os.getenv("CLUSTER_GRAY_THRESHOLD",      "0.85"))
CLUSTER_ANCHOR_THRESHOLD    = float(os.getenv("CLUSTER_ANCHOR_THRESHOLD",    "0.75"))
CLUSTER_MIN_SHARED_ENTITIES = int(os.getenv("CLUSTER_MIN_SHARED_ENTITIES",   "2"))
CLUSTER_IMAGE_MAX_DISTANCE  = int(os.getenv("CLUSTER_IMAGE_MAX_DISTANCE",    "6"))
CLUSTER_MAX_GAP_HOURS       = int(os.getenv("CLUSTER_MAX_GAP_HOURS",         "18"))
CLUSTER_MAX_SPAN_HOURS      = int(os.getenv("CLUSTER_MAX_SPAN_HOURS",        "120"))
CLUSTER_CANDIDATES          = int(os.getenv("CLUSTER_CANDIDATES",            "5"))
CLUSTER_MERGE_THRESHOLD     = float(os.getenv("CLUSTER_MERGE_THRESHOLD",     "0.845"))
# In-memory clustering (enrichment/memory_clustering.py): clusters last seen
# longer ago than this are dropped from the runner's state. A fresh article
# can only join a cluster seen within CLUSTER_MAX_GAP_HOURS of its time, so
# gap + a margin for publish-to-crawl delay is enough.
CLUSTER_STATE_RETAIN_HOURS  = int(os.getenv("CLUSTER_STATE_RETAIN_HOURS",    "30"))

# Embedding input: headline + description/lead, capped (see steps/embedding.py)
EMBED_MAX_CHARS  = int(os.getenv("EMBED_MAX_CHARS",  "400"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Entities used as clustering evidence: salient, specific types only.
CLUSTER_ENTITY_MIN_SALIENCE = 0.3
CLUSTER_ENTITY_TYPES        = frozenset({"PERSON", "ORG", "GPE", "EVENT", "LAW", "PRODUCT"})
# Entities that co-occur with almost every Indian story (or are the wire
# agency's own byline) — shared mentions of these are NOT evidence of
# being the same story.
CLUSTER_ENTITY_STOPLIST = frozenset({
    "india", "indian", "indians", "bharat", "new delhi", "delhi",
    "centre", "government", "govt", "the government", "union government",
    "pti", "ani", "ians", "reuters", "afp", "ap", "uni", "bloomberg",
    "x", "twitter", "facebook", "instagram", "youtube", "whatsapp",
    "today", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "भारत", "दिल्ली", "सरकार", "केंद्र",
})
# Perceptual hashes of blank / solid-colour images (all bits equal) carry no
# information — never use them as evidence.
CLUSTER_IMAGE_HASH_DENYLIST = frozenset({0, -1})
