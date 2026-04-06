# WIZER — News Intelligence Pipeline: Full System Guide for Claude

This document explains the entire WIZER codebase so Claude can understand the exact
architecture, data flow, design decisions, and known issues before suggesting improvements.

---

## What WIZER Is

WIZER is a **two-layer news intelligence pipeline** that:
1. **Layer 1 — Ingestion**: Polls 25,000+ RSS/Atom feeds, crawls full article text, deduplicates, and stores articles in Supabase.
2. **Layer 2 — Enrichment**: Runs NLP on stored articles (language detection, NER, sentiment, keywords, category classification, story clustering, propensity scoring).

Everything runs on **GitHub Actions free tier** (2 vCPU, 7 GB RAM, 6h timeout). No paid infrastructure.

The product is **Indian news** — the corpus is primarily English and Hindi but includes Tamil, Telugu, Bengali, Marathi, Gujarati, Punjabi, Urdu. All model choices must work multilingual.

---

## Repository Layout

```
WIZER/
├── CLAUDE.md                        ← this file
├── Ingestion/
│   ├── main.py                      ← Layer 1 CLI entry point
│   ├── enrich.py                    ← Layer 2 CLI entry point
│   ├── push_feeds.py                ← import feeds from CSV → Supabase
│   ├── recrawl.py                   ← retry failed article crawls (nightly)
│   ├── cleanup_clusters.py          ← prune stale singleton clusters
│   ├── train_propensity.py          ← train LightGBM propensity model
│   ├── debug_classifier.py          ← interactive classifier testing
│   ├── requirements.txt
│   ├── pipeline/                    ← Layer 1 modules
│   │   ├── config.py                ← all Layer 1 settings
│   │   ├── poller.py                ← async feed-polling engine
│   │   ├── crawler.py               ← HTML fetch + content extraction
│   │   ├── db.py                    ← all Supabase reads/writes for Layer 1
│   │   ├── dedup.py                 ← URL normalisation, hashing, SimHash
│   │   └── circuit_breaker.py       ← feed health / dormancy detection
│   ├── enrichment/                  ← Layer 2 modules
│   │   ├── config.py                ← all Layer 2 settings
│   │   ├── db.py                    ← all Supabase reads/writes for Layer 2
│   │   ├── runner.py                ← enrichment orchestrator
│   │   └── steps/
│   │       ├── text_stats.py        ← word count, reading time
│   │       ├── language.py          ← lingua language detection
│   │       ├── sentiment.py         ← multilingual distilbert sentiment
│   │       ├── ner.py               ← spaCy NER (en + multilingual)
│   │       ├── keywords.py          ← YAKE keyword extraction
│   │       ├── classifier.py        ← mDeBERTa zero-shot category classification
│   │       ├── images.py            ← download top image + perceptual hash
│   │       ├── clustering.py        ← LaBSE embedding + pgvector story clustering
│   │       └── propensity.py        ← virality scoring (formula + LightGBM)
│   └── tests/
│       └── test_pipeline.py         ← unit tests for dedup logic
├── Feed_Validator/
│   └── validate.py                  ← validate RSS feeds at scale
└── Enrich/                          ← (legacy / alternate enrichment folder)
```

---

## Database (Supabase / PostgreSQL)

### Key Tables

**`feeds`** — one row per RSS feed
```
feed_url, final_url, domain, publisher_name
update_cadence          — breaking_news | multiple_daily | daily | weekly | monthly
language_code           — feed-level declared language (not always accurate)
country_code            — IN, US, GB, etc.
iab_tier1, iab_tier2    — IAB category labels
has_paywall             — bool (static list in config.py)
poll_interval_mins      — how often to poll (60 for breaking_news, 1440 for daily)
priority_score          — float, higher = more important
is_active               — false = disabled (error streak or dormancy)
fail_count              — consecutive failures
last_polled_at          — timestamp of last poll
last_success_at         — timestamp of last successful poll
articles_found          — total articles ever stored from this feed
```

**`articles`** — one row per article
```
feed_id                 — FK to feeds
url                     — canonical URL (after redirect resolution)
url_hash                — MurmurHash3 of normalised URL (UNIQUE index)
title
title_simhash           — 64-bit SimHash of title for near-dedup
description             — RSS summary
full_text               — crawled full article body (up to 80,000 chars)
author
top_image_url
published_at
crawled_at
language_code           — feed-declared language
language_detected       — enrichment-detected actual language
country_code
og_tags                 — JSONB: all Open Graph metadata
is_crawled              — bool: full_text was successfully fetched
is_duplicate            — bool: near-duplicate of another article
crawl_strategy          — which of the 4 crawl strategies succeeded
feed_url, domain, publisher_name   — denormalised from feeds
iab_tier1, iab_tier2    — denormalised from feeds
enriched_at             — null = not yet enriched; set after Layer 2 runs
word_count
reading_time_mins
sentiment               — positive | neutral | negative
sentiment_score         — float -1.0 to 1.0
keywords                — JSONB array of strings
category                — cricket | politics | business | entertainment |
                          technology | sports | health | education | crime |
                          environment | world | crypto | general
image_phash             — perceptual hash of top image (for image dedup)
cluster_id              — FK to article_clusters (null if unclustered)
propensity_score        — float 0.0–1.0 (virality estimate)
```

**`article_entities`** — NER results, many-to-one with articles
```
article_id
entity_text             — "Narendra Modi", "BJP", "Mumbai"
entity_type             — PERSON | ORG | GPE | EVENT | PRODUCT | LAW
salience                — float 0.0–1.0
```

**`article_clusters`** — story clusters (one cluster = one real-world event)
```
id
headline                — best headline seen so far (refreshed to longest title)
canonical_article_id    — first article that created the cluster
article_count           — total articles in this cluster
outlet_count            — number of DISTINCT domains in this cluster
outlet_set              — JSONB array of domain strings (for dedup)
entity_set              — JSONB array of entity text strings
top_entities            — JSONB array of {text, type, count} sorted by frequency
canonical_embedding     — JSONB float list: running-mean LaBSE centroid
embedding_vec           — pgvector vector(768): same data, used for HNSW search
first_seen_at
last_seen_at
updated_at
```

**`pipeline_runs`** — one row per Layer 1 run
```
cadence, feeds_attempted, feeds_skipped, new_articles,
near_duplicates, exact_duplicates, errors, duration_s, dry_run
```

---

## Layer 1: Ingestion Pipeline — Exact Data Flow

### Entry Point
```bash
python main.py                            # poll all cadences
python main.py --cadence breaking_news    # poll specific cadence
python main.py --dry-run --verbose        # no DB writes
```

### Step-by-step flow

```
1. db.get_due_feeds(cadence)
   → SELECT feeds WHERE is_active=true AND last_polled_at < NOW() - poll_interval_mins
   → Returns list of feed dicts

2. For each feed (async, MAX_CONCURRENT_FEEDS=3 semaphore):

   a. circuit_breaker.should_skip_feed(feed)
      → Skip if fail_count >= MAX_ERRORS_BEFORE_DISABLE (5)
      → Skip dormant feeds except 1x/week retry window

   b. db.load_recent_hashes(feed_id, limit=2000)
      → Load last 2000 url_hash values for this feed into Python set
      → In-memory O(1) dedup layer 1

   c. db.load_recent_simhashes(limit=10000)
      → Load last 10000 title_simhash values across ALL feeds
      → Near-dedup layer 3

   d. _fetch_rss_blocking(feed_url)
      → feedparser.parse() in thread pool (blocking, must not block event loop)
      → Returns parsed feed with entries list

   e. For each RSS entry (async, MAX_CONCURRENT_ARTICLES=2 semaphore):

      i.  dedup.normalise_url(entry.link)
          → lowercase scheme+host, strip www., remove tracking params
            (?utm_*, ?fbclid=, ?ref=, ?source=, etc.)
          → Canonical URL

      ii. dedup.url_hash(normalised_url)
          → MurmurHash3 64-bit signed int (mmh3 library, hashlib fallback)

      iii. Layer 1 dedup: if url_hash in seen_hashes → skip (exact, in-memory)

      iv.  Layer 2 dedup: db.url_hash_exists(url_hash)
           → Query UNIQUE index on articles.url_hash
           → fail-open: if DB error, returns False (UNIQUE constraint is last resort)

      v.   dedup.simhash(title)
           → 64-bit fingerprint of article title words
           → Layer 3 dedup: hamming_distance(sh, seen_sh) <= 3 → is_near_duplicate=True
           → Near-duplicates are STORED but flagged is_duplicate=True

      vi.  crawler.crawl_article(entry, feed, norm_url, url_hash, title_sh)
           → Returns CrawledArticle dataclass

3. crawler.crawl_article() — HTML fetch strategies (tried in order):
   Strategy 1: Default UA "NewsIngestBot/1.0" (skip if has_paywall=True)
   Strategy 2: Googlebot UA (many paywalls whitelist Googlebot)
   Strategy 3: AMP variants — try /amp, ?amp=1, amp.{domain}
   Strategy 4: Wayback Machine — https://web.archive.org/web/2/{url}
   Each strategy: 3 retries, exponential backoff (1s → 2s → 4s), 15s timeout

4. crawler: full-text extraction (tried in order, returns first result ≥200 chars):
   1. RSS content:encoded (no HTTP needed, free)
   2. trafilatura (primary — best accuracy for news)
   3. readability-lxml (Mozilla Readability algorithm)
   4. newspaper3k (re-fetches URL internally)
   5. BeautifulSoup raw <p> extraction (last resort)

5. crawler: metadata extraction from HTML:
   - Open Graph tags (og:title, og:description, og:image, og:video,
                       article:published_time, article:author)
   - Twitter Card tags (twitter:image, twitter:player)
   - JSON-LD structured data (NewsArticle, Article, BlogPosting)
   - Images: og:image → JSON-LD image → RSS media:content → first <img>
   - Videos: og:video → twitter:player → YouTube/Vimeo iframes → <source> MP4/WebM
   - Publication date: RSS date → JSON-LD datePublished → og:article:published_time
   - Author: JSON-LD author → og:article:author → RSS entry author

6. db.upsert_articles(rows)
   → Batch INSERT with ON CONFLICT(url_hash) DO NOTHING
   → Falls back to row-by-row if batch fails (partial batches not lost)

7. db.update_feed_after_poll(feed_id, ...)
   → Update last_polled_at, fail_count, articles_found

8. circuit_breaker.check_dormancy(feed)
   → If no new articles in 30 days → db.mark_feed_dormant()

9. db.log_run_finish(run_id, summary)
   → Update pipeline_runs row with final stats
```

---

## Layer 2: Enrichment Pipeline — Exact Data Flow

### Entry Point
```bash
python enrich.py                          # enrich next 500 unenriched articles
python enrich.py --batch-size 200         # smaller batch
python enrich.py --dry-run                # compute but don't write
python enrich.py --force                  # re-enrich already-enriched articles
```

### Fetch criteria
```sql
SELECT ... FROM articles
WHERE enriched_at IS NULL
  AND is_crawled = true
  AND title IS NOT NULL
ORDER BY published_at DESC
LIMIT :batch_size
```
Newest-first so the app layer gets enriched data for fresh articles before stale ones.

### Processing model
**Sequential, not concurrent.** NLP models are CPU-bound. `asyncio` doesn't help.
Processing time: 1–5 seconds/article → 100–500 seconds/100 articles.
All models loaded once per process and cached globally (lazy init on first use).

### Step-by-step enrichment (per article)

```
enrich_one(article) → (article_update dict, entities list, cluster_payload, cluster_action)

Step 1: text_stats.compute_text_stats(full_text, description)
  → word_count (whitespace split)
  → reading_time_mins (word_count / 200 wpm)
  → NO external libraries, pure Python

  GATE: if word_count < ENRICH_MIN_WORD_COUNT (50):
    → return early, mark enriched_at, skip all expensive steps
    → stub articles / wire briefs don't have enough signal

Step 2: language.detect_language(full_text, description, title)
  → lingua library (deterministic, not probabilistic like langdetect)
  → Returns ISO code: "en", "hi", "ta", "te", "bn", "mr", etc.
  → Stored as language_detected (separate from language_code which is feed-declared)

  GATE: lang_base in ENRICH_SUPPORTED_LANGUAGES (currently {"en", "hi"})
  → If NOT in supported languages: skip steps 3–7 (no NER, sentiment, etc.)
  → ISSUE: Tamil/Telugu/Bengali articles don't get NER or clustering
    despite the clustering step being designed to be language-agnostic

Step 3: sentiment.analyse_sentiment(title, description, language_detected)
  → Model: lxyuan/distilbert-base-multilingual-cased-sentiments-student
  → 268 MB, CPU ~50ms/article
  → Input: title + description (NOT full_text — short text has strongest sentiment signal)
  → Returns: sentiment ("positive"|"neutral"|"negative"), sentiment_score (pos_prob - neg_prob)
  → language_detected param is accepted but UNUSED (model is natively multilingual)

Step 4: ner.extract_entities(title, full_text, description, language_detected)
  → English → spaCy en_core_web_sm
  → Others  → spaCy xx_ent_wiki_sm (fallback: en_core_web_sm if missing)
  → Combined text: title + body[:5000]
  → Entity types: PERSON, ORG, GPE, EVENT, PRODUCT, LAW
    (LOC→GPE, NORP→ORG, WORK_OF_ART→PRODUCT; DATE/MONEY/CARDINAL etc. skipped)
  → Salience scoring per entity:
      title mention:         +0.5
      first 200 chars:       +0.3
      each body mention:     +0.1 (capped at 0.3)
  → Filter: salience < NER_MIN_SALIENCE (0.1) → dropped
  → Returns top 30 entities sorted by salience desc

Step 5: keywords.extract_keywords(title, full_text, description)
  → YAKE (Yet Another Keyword Extractor): unsupervised, no model, language-agnostic
  → Input: title + body[:3000] (title at top for position weighting)
  → Returns top 10 keyword strings (max 3-gram, dedup threshold 0.9)
  → Stored as JSONB array in articles.keywords

Step 6: classifier.classify_article(title, description, iab_tier1, full_text)
  → Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
  → 560 MB, CPU ~150-200ms/article
  → Zero-shot NLI: frames classification as "does article ENTAIL 'this is about [category]'?"
  → 12 candidate labels (descriptive English phrases regardless of article language)
  → Returns winning category if score >= CLASSIFY_CONFIDENCE_THRESHOLD (0.15 — TOO LOW)
  → Returns "general" on low confidence or error
  → ISSUE: iab_tier1 param is accepted but NEVER used in the function body
  → ISSUE: CLASSIFY_CONFIDENCE_THRESHOLD=0.15 is barely above random for 12 labels

Step 7: images.download_and_hash_image(top_image_url)
  → Download image (10s timeout, skip if >5 MB)
  → Compute 64-bit perceptual hash (pHash, 8x8 grid)
  → Stored in articles.image_phash
  → Used later for image deduplication across outlets

Step 8: clustering.find_or_create_cluster(article, entities)
  → Compute LaBSE embedding of title + description[:512]
     Model: sentence-transformers/LaBSE (471 MB, Google, 109 languages)
     Output: 768-dim L2-normalised float vector
  → db.find_nearest_cluster(embedding, threshold=0.82, window_hours=48)
     One SQL call → pgvector HNSW nearest-neighbour search on article_clusters
     Returns best cluster if cosine_similarity >= 0.82
  → If match found (action="join"):
     - Merge entity_set (union of entity texts)
     - Merge top_entities (increment mention counts, keep top 15)
     - Update outlet_set and outlet_count (domain dedup via outlet_set JSONB array)
     - Update canonical_embedding via running mean: new = ((N-1)/N)*old + (1/N)*new
       Re-normalise to unit vector after blending
     - Refresh headline if new title is longer (wire flash → full story)
  → If no match (action="create"):
     - Create new cluster row with this article as seed
  → Embedding failures (zero vector) → action="skip", article not clustered

Step 9: propensity.compute_propensity(article, update, entities, cluster_payload, cluster_action)
  → Two phases, same 6-element feature vector for both:
     [word_count, category_encoded, is_english, sentiment_num, sentiment_score, avg_salience]
     avg_salience = mean of top-5 entity saliences

  Phase 2 (if models/propensity_model.lgb exists):
    → LightGBM model trained on log(1 + outlet_count) as self-supervised label
    → Raw prediction / log(1+50) → normalised to [0,1]
    → Blended: score * 0.85 + recency * 0.15 (recency = exp(-hours_old/24))

  Phase 1 (formula fallback):
    → score = 0.40 * outlet_virality   (log-normalised outlet_count)
            + 0.25 * category_weight   (cricket=0.90, politics=0.85, ..., general=0.30)
            + 0.20 * entity_signal     (avg salience of top-5 entities)
            + 0.15 * recency_factor    (exp decay over 24h)

  outlet_count from clustering:
    - action="join"  → cluster_payload["outlet_count"]
    - action="create" or "skip" → 1

Persist results:
  db.save_entities(article_id, entities)         → INSERT into article_entities
  db.create_cluster / db.update_cluster(...)     → INSERT/UPDATE article_clusters
  db.save_article_enrichment(article_id, update) → UPDATE articles SET ..., enriched_at=NOW()
  → enriched_at is set ONLY if save succeeds → article retried on next run if DB fails
```

---

## Deduplication System (Layer 1)

Three independent layers, each catching different duplicate types:

```
Layer 1 — In-memory URL hash set (per feed, per run)
  - 2,000 recent url_hash values loaded at start of each feed poll
  - Python set, O(1) lookup
  - Catches: exact same URL seen twice within one run

Layer 2 — Database UNIQUE index on articles.url_hash
  - db.url_hash_exists(url_hash) queries this index
  - Catches: exact same URL stored anytime in the past, across all feeds
  - Fail-open: if DB unreachable, returns False; UNIQUE constraint is last resort

Layer 3 — SimHash near-dedup on article title
  - 64-bit fingerprint of title words (mmh3 or hashlib)
  - 10,000 recent title_simhash values loaded per run
  - hamming_distance(sh1, sh2) <= 3 bits → near-duplicate
  - Near-duplicates STORED but flagged is_duplicate=True
  - Catches: same PTI wire story published by 10 different outlets with slightly
    different headlines ("Modi announces budget" vs "PM Modi announces Union Budget")
```

---

## Circuit Breaker (Layer 1)

Feed health is tracked via two independent failure modes:

```
Mode 1: Error streak
  - fail_count incremented on each consecutive failure
  - fail_count >= 5 (MAX_ERRORS_BEFORE_DISABLE) → feed.is_active = False
  - Reset: manual update in Supabase (fail_count=0, is_active=True)

Mode 2: Dormancy (no new articles in 30 days)
  - Checked after each poll via circuit_breaker.check_dormancy()
  - get_last_new_article_date(feed) → most recent article from this feed
  - If > DORMANCY_DAYS (30) days ago → mark_feed_dormant()
  - Dormant feeds get 1x/week retry (may have reactivated)
```

---

## Concurrency Model (Layer 1)

```
Async event loop with semaphores:
  MAX_CONCURRENT_FEEDS = 3     → at most 3 feeds being polled simultaneously
  MAX_CONCURRENT_ARTICLES = 2  → at most 2 articles being crawled per feed

Blocking operations run in thread pool via loop.run_in_executor():
  - feedparser.parse()         → blocks event loop without executor
  - crawler.crawl_article()    → HTTP + heavy parsing, must not block loop

Why low concurrency limits?
  - GitHub Actions IPs get rate-limited by large publishers
  - Avoids thundering-herd on Supabase connection pool
  - Tunable via config.py if running on dedicated infrastructure
```

---

## DB Connection Handling (Layer 2)

```python
# enrichment/db.py
_DB_TIMEOUT_SECONDS = 20  # was 120s default — caused 2-min hangs on dead connections

def _run_with_retry(operation):
    # NLP steps take 1–5 min per article
    # During NLP, Supabase TCP connection sits idle and gets dropped (OS/network ~2 min)
    # First write attempt → ReadTimeout after 20s
    # _reset_client() clears singleton, sleep(1), retry with fresh connection
    # Second attempt almost always succeeds (fresh TCP)
    try:
        return operation()
    except retriable_error:
        _reset_client()
        time.sleep(1)
        return operation()
```

---

## Known Issues and Technical Debt

### Bugs (incorrect behaviour)

1. **`CLASSIFY_CONFIDENCE_THRESHOLD = 0.15` is too low**
   With 12 candidate NLI labels, random-baseline per label ≈ 0.083.
   Threshold of 0.15 accepts very low-confidence classifications.
   Articles with no clear category get mislabelled instead of returning "general".
   **Fix**: raise to 0.30–0.35.

2. **Propensity feature vector count mismatch in documentation**
   `_build_features()` returns 6 elements: `[word_count, category_enc, is_english,
   sentiment_num, sent_score, avg_salience]`.
   The module docstring says 5 features (omits `avg_salience`).
   If `train_propensity.py` was run before `avg_salience` was added, the saved
   `.lgb` model expects 5 features → silent wrong predictions at inference time.
   **Fix**: verify feature count in `train_propensity.py` matches `_build_features()`.

3. **`iab_tier1` param in `classify_article()` is silently ignored**
   Accepted as "fallback tiebreaker" in docstring but body never uses it.
   **Fix**: either use it as a prior/tiebreaker or remove the parameter.

4. **`SENTIMENT_POSITIVE_THRESHOLD` / `SENTIMENT_NEGATIVE_THRESHOLD` are defined but unused**
   `config.py:504-505` defines thresholds; `sentiment.py` uses `max(score_map)` directly.
   **Fix**: use the thresholds to gate label assignment or delete them.

### Dead Code

5. **`CATEGORY_RULES` dict (~250 lines) in `config.py` is never read**
   The rule-based keyword classifier was replaced by mDeBERTa zero-shot NLI.
   `classifier.py` never imports or uses `CATEGORY_RULES`.
   The comment above it still says "WHY RULE-BASED (not ML)?" — wrong.
   **Fix**: delete `CATEGORY_RULES` and the stale comment.

6. **`language_detected` param in `analyse_sentiment()` is unused**
   The multilingual distilbert model doesn't need a language hint.
   The param is kept "for interface compatibility" but adds confusion.

### Design Gaps

7. **Non-English languages get almost no enrichment**
   `ENRICH_SUPPORTED_LANGUAGES = {"en", "hi"}` by default.
   Tamil (ta), Telugu (te), Bengali (bn), Marathi (mr) articles skip NER,
   sentiment, keywords, classification, and — critically — **clustering**.
   The clustering step comment in `runner.py:164` says "ALL languages — every
   article contributes" but the `rich_enrich` gate contradicts this.
   **Fix**: move clustering outside the `rich_enrich` gate, or expand supported languages.

8. **URL normalisation is duplicated**
   `push_feeds.py` has its own `_normalise()`.
   `pipeline/dedup.py` has `normalise_url()`.
   Two implementations → risk of drift if one is updated but not the other.
   **Fix**: extract to a shared `pipeline/utils.py`.

9. **`recrawl.py` doesn't re-evaluate SimHash**
   If an article was initially marked `is_duplicate=True`, recrawl won't re-check.
   The duplicate flag is permanent once set.

10. **Throughput ceiling**
    Sequential enrichment: 1–5s/article.
    500-article batch = up to 42 minutes.
    At 8 enrichment runs/day = ~4,000 articles/day max enriched.
    If ingestion pulls 20K+ articles/day, the enrichment queue grows indefinitely.
    No queue depth metric is recorded anywhere.

11. **Clustering only embeds title + description (not full_text)**
    `embed_text = f"{title}. {description}"[:512]` in `clustering.py:211`.
    Two articles about the same event with different headlines and thin
    descriptions may score below the 0.82 threshold and create duplicate clusters.

12. **No source credibility scoring**
    All feeds are polled with equal weight regardless of publisher reputation.
    PTI wire republished by 100 clickbait sites contributes equally to
    outlet_count as 100 independent editorial stories.

13. **Static paywall domain list**
    `has_paywall` is set at feed import time and never updated automatically.
    Sites that add/remove paywalls silently degrade crawl quality.

---

## Model Inventory

| Model | Size | Used for | Where loaded |
|---|---|---|---|
| `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | 268 MB | Sentiment | `sentiment.py` |
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | 560 MB | Category classification | `classifier.py` |
| `sentence-transformers/LaBSE` | 471 MB | Story clustering embeddings | `clustering.py` |
| `en_core_web_sm` (spaCy) | 12 MB | English NER | `ner.py` |
| `xx_ent_wiki_sm` (spaCy) | ~50 MB | Multilingual NER | `ner.py` |
| `models/propensity_model.lgb` | small | Propensity scoring Phase 2 | `propensity.py` |

Total RAM at peak (all models loaded): ~1.5 GB.
GitHub Actions free tier: 7 GB RAM — comfortable.

All models are lazy-loaded (first call) and cached in module-level globals.
Models are NOT reloaded between articles — only once per process.

---

## Configuration Reference

### Layer 1 (`pipeline/config.py`)
```
MAX_CONCURRENT_FEEDS       = 3
MAX_CONCURRENT_ARTICLES    = 2
MAX_ERRORS_BEFORE_DISABLE  = 5        # consecutive failures before disabling feed
DORMANCY_DAYS              = 30       # days without new articles before marking dormant
CRAWL_TIMEOUT_SECONDS      = 15
MAX_ARTICLE_BODY_CHARS     = 80_000
ARTICLE_HARD_LIMIT         = 500_000  # max articles table size
ARTICLE_PRUNE_TARGET       = 490_000  # prune to this level
```

### Layer 2 (`enrichment/config.py`)
```
ENRICH_BATCH_SIZE               = 500
ENRICH_MIN_WORD_COUNT           = 50      # articles below this skip NER/clustering
ENRICH_SUPPORTED_LANGUAGES      = {"en", "hi"}   # others get text_stats only
CLUSTER_EMBEDDING_THRESHOLD     = 0.82    # cosine similarity to join a cluster
CLUSTER_WINDOW_HOURS            = 48      # search window for nearest cluster
CLUSTER_SINGLETON_TTL_HOURS     = 72      # cleanup_clusters.py prunes these
CLASSIFY_CONFIDENCE_THRESHOLD   = 0.15    # BUG: too low, should be 0.30+
NER_MIN_SALIENCE                = 0.1
MAX_KEYWORDS                    = 10
PROPENSITY_MAX_OUTLET_COUNT     = 50      # log normalisation cap
```

---

## Entry Points Summary

```bash
# Layer 1
python Ingestion/main.py --cadence breaking_news   # poll high-frequency feeds
python Ingestion/main.py                           # poll all cadences
python Ingestion/main.py --dry-run --verbose

# Layer 2
python Ingestion/enrich.py                         # enrich next 500 articles
python Ingestion/enrich.py --batch-size 200 --dry-run
python Ingestion/enrich.py --force                 # re-enrich already-enriched

# Maintenance
python Ingestion/recrawl.py --limit 500 --max-age-days 7   # retry failed crawls
python Ingestion/cleanup_clusters.py                        # prune stale clusters
python Ingestion/train_propensity.py                        # retrain LightGBM model
python Ingestion/push_feeds.py --csv feeds.csv --dry-run    # import new feeds
```

---

## What "Good" Looks Like for This Pipeline

For context when suggesting improvements:

- **Ingestion throughput target**: process all due feeds within their cadence window.
  A breaking_news feed at 60-min interval must be polled within 60 minutes.
- **Enrichment throughput target**: enrich all articles within ~2 hours of ingestion.
  This means the enrichment queue must not grow faster than it's consumed.
- **Cluster quality**: a PTI story reported by 50 outlets should form ONE cluster
  with outlet_count=50, not 50 singleton clusters.
- **Category accuracy**: <5% misclassification on unambiguous articles.
- **Dedup precision**: no article stored twice; near-duplicates flagged but preserved.
- **Zero data loss**: if GitHub Actions job is killed mid-run, partial progress is saved.
