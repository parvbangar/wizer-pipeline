# Wizer Ingestion Pipeline

A two-layer production pipeline that ingests 25,000+ RSS feeds, crawls full article text, and enriches each article with NLP metadata — all running on GitHub Actions at zero cost.

---

## Architecture Overview

```
Layer 1 — Ingestion          Layer 2 — Enrichment
─────────────────────        ──────────────────────────────────────
RSS Feeds (25K+)             Unenriched articles (is_crawled=true)
       ↓                                    ↓
  Fetch RSS                         Language detection
       ↓                                    ↓
  Crawl HTML                        Named entity recognition
       ↓                                    ↓
  Deduplicate                       Keyword extraction
       ↓                                    ↓
  Store articles                    Category classification
                                           ↓
                                    Sentiment analysis
                                           ↓
                                    Image pHash
                                           ↓
                                    Story clustering
                                           ↓
                                  Enriched articles in DB
```

**Database**: Supabase (PostgreSQL)
**Runtime**: GitHub Actions (free tier, public repo)
**Cost**: $0

---

## Layer 1 — Ingestion

### What It Does

Polls RSS feeds on a schedule, crawls the full article HTML, deduplicates, and inserts into Supabase.

### Entry Point

```bash
python main.py --cadence breaking_news
python main.py --cadence daily --dry-run
```

### Feed Cadences

| Cadence | Workflow | Schedule | Poll Interval |
|---|---|---|---|
| `breaking_news` | `ingest_breaking_news.yml` | Every 1 hour | 60 min |
| `multiple_daily` | `ingest_multiple_daily.yml` | Every 3 hours | 3 hours |
| `daily` | `ingest_daily.yml` | Every 12 hours | 12 hours |
| `several_weekly` | `ingest_several_weekly.yml` | Every 24 hours | 24 hours |
| `weekly` | `ingest_weekly.yml` | Every 7 days | 7 days |
| `monthly` | `ingest_monthly.yml` | Every 30 days | 30 days |

~2,150 feeds total. Breaking news feeds are polled every hour.

### How a Single Feed Gets Processed

```
1. circuit_breaker.should_skip_feed()
   → Skip if is_active=False or fail_count ≥ 5

2. db.load_recent_hashes(feed_id)
   → Warm in-memory set of last 2,000 URL hashes for fast dedup

3. _fetch_rss_blocking(feed_url)
   → feedparser parses RSS/Atom; handles malformed feeds

4. For each RSS entry:
   a. dedup.normalise_url()    → strip utm_*, sort params, remove www.
   b. dedup.url_hash()         → MurmurHash3 64-bit
   c. Check in-memory hash set → skip if seen
   d. db.url_hash_exists()     → authoritative DB check on UNIQUE index
   e. crawler.crawl_article()  → fetch full HTML + extract text
   f. dedup.simhash(title)     → Hamming distance ≤ 3 = near-duplicate, skip

5. db.upsert_articles(rows)    → bulk insert, conflict ignored on url_hash
6. db.update_feed_after_poll() → reset fail_count or increment
7. circuit_breaker.check_dormancy()
   → No new articles in 30 days → mark is_active=False
```

### Article Crawler (`pipeline/crawler.py`)

When RSS has no full text, the crawler fetches the article URL.

**Four HTTP strategies (tried in order):**
1. `NewsIngestBot/2.0` user-agent — works for ~70% of feeds
2. Googlebot spoof — bypasses soft paywalls (~20% more)
3. AMP URL (`/amp`, `?amp=1`) — lightweight paywall-free version (~5% more)
4. Wayback Machine — last resort for archived content

Each strategy: up to 3 retries with 1s → 2s → 4s exponential backoff.

**Four text extraction libraries (returns longest result):**
1. `trafilatura` — news-optimised, primary extractor
2. `readability-lxml` — Mozilla Readability algorithm
3. `newspaper3k` — fallback, re-fetches URL internally
4. `BeautifulSoup` raw `<p>` tags — last resort

### Deduplication (`pipeline/dedup.py`)

Three independent layers:

| Layer | Method | Speed | Coverage |
|---|---|---|---|
| In-memory | MurmurHash3 hash set | ~0ms | Last 2,000 URLs per feed |
| DB unique index | `WHERE url_hash = ?` | ~5ms | All articles ever |
| Near-duplicate | SimHash Hamming distance ≤ 3 | ~0ms | Same story, different URL |

URL normalisation strips tracking params (`utm_*`, `fbclid`, `gclid`, `ref`, etc.) so the same article with different tracking URLs is treated as one article.

### Circuit Breaker (`pipeline/circuit_breaker.py`)

Auto-disables feeds that consistently fail:
- **Error streak**: `fail_count ≥ 5` → `is_active = False`
- **Dormancy**: No new articles in 30 days → `is_active = False`

To re-enable manually:
```sql
UPDATE feeds SET is_active = true, fail_count = 0 WHERE id = '...';
```

### Concurrency

- `MAX_CONCURRENT_FEEDS = 20` — 20 feeds polled simultaneously
- `MAX_CONCURRENT_ARTICLES = 2` — 2 articles crawled per feed at once
- Controlled via `asyncio.Semaphore`

### Article Table Pruning

Auto-prunes when the `articles` table exceeds 500,000 rows, targeting 490,000. Oldest articles deleted first. Configurable via `ARTICLE_HARD_LIMIT` in `pipeline/config.py`.

---

## Layer 2 — Enrichment

### What It Does

Runs 8 NLP steps on each crawled article and writes results back to Supabase. Runs every hour, processing up to 4,000 articles per trigger using two parallel shards.

### Entry Point

```bash
python enrich.py
python enrich.py --batch-size 2000 --offset 0
python enrich.py --dry-run --verbose
python enrich.py --force --batch-size 1000   # re-enrich already-enriched articles
```

### GitHub Actions Schedule

Every hour at `:30` (offset from ingestion at `:00`).

| Shard | Offset | Articles | Runs/day | Total/day |
|---|---|---|---|---|
| Shard 0 | 0 | 2,000 newest | 24 | 48,000 |
| Shard 1 | 2,000 | 2,000 next | 24 | 48,000 |
| **Total** | | | | **~96,000** |

Both shards run in parallel with zero article overlap (offset-based pagination).

### The 8 Enrichment Steps

Articles below `ENRICH_MIN_WORD_COUNT = 50` get `text_stats` only — all other steps are skipped (stubs/briefs with no useful NLP signal).

---

#### Step 1: Text Stats
**File**: `enrichment/steps/text_stats.py`

Computes `word_count` and `reading_time_mins` (word_count ÷ 200 WPM). Pure Python, always runs.

---

#### Step 2: Language Detection
**File**: `enrichment/steps/language.py`
**Library**: `lingua-language-detector`

Detects the actual language of the article body. Deterministic (unlike `langdetect`).

Supported Indian languages: Hindi (hi), Bengali (bn), Gujarati (gu), Marathi (mr), Punjabi (pa), Tamil (ta), Telugu (te), Urdu (ur).

---

#### Step 3: Sentiment Analysis
**File**: `enrichment/steps/sentiment.py`
**Library**: VADER (`vaderSentiment`)

English-only. Returns `sentiment` (positive / negative / neutral) and `sentiment_score` (−1.0 to +1.0). Non-English articles get `NULL`.

Thresholds: compound ≥ 0.05 = positive, ≤ −0.05 = negative, between = neutral.

---

#### Step 4: Named Entity Recognition
**File**: `enrichment/steps/ner.py`
**Library**: spaCy (two models)

| Language | Model | Notes |
|---|---|---|
| English | `en_core_web_sm` | Optimised for English news |
| All Indian languages | `xx_ent_wiki_sm` | Multilingual Wikipedia-trained |

Entity types stored: `PERSON`, `ORG`, `GPE`, `EVENT`, `PRODUCT`, `LAW`.

Salience scoring (0.0–1.0): title mention +0.5, first 200 chars +0.3, each body mention +0.1. Entities below `NER_MIN_SALIENCE = 0.1` are discarded. Results go into the `article_entities` table.

---

#### Step 5: Keyword Extraction
**File**: `enrichment/steps/keywords.py`
**Library**: YAKE

Language-agnostic, unsupervised. Top 10 keyword phrases (up to 3 words). Stored as a jsonb array in `articles.keywords`.

---

#### Step 6: Category Classification
**File**: `enrichment/steps/classifier.py`
**Model**: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (~560 MB, cached)

Zero-shot Natural Language Inference. The model computes:
> "Does this article **entail** the hypothesis: *this article is about [category]*?"

The highest-scoring category wins (minimum confidence: 0.40, else `general`). Works correctly on Hindi/Tamil articles with English category labels via cross-lingual transfer.

**Categories**: cricket · politics · business · entertainment · technology · sports · health · education · crime · environment · world · crypto · general

Performance: ~150–200ms/article on CPU.

---

#### Step 7: Image Perceptual Hash
**File**: `enrichment/steps/images.py`
**Library**: `imagehash`, `Pillow`

Downloads `top_image_url` (10s timeout, 5 MB max), computes a 64-bit pHash stored as a signed `bigint`. Used for image-level deduplication and future visual LLM features.

---

#### Step 8: Story Clustering
**File**: `enrichment/steps/clustering.py`
**Model**: `sentence-transformers/LaBSE` (~471 MB, cached)

Groups articles about the same real-world event into `article_clusters`.

**Why LaBSE over entity/keyword matching:**
Entity Jaccard fails for non-English — spaCy extracts zero entities from Tamil or Telugu text, so every regional article would start its own cluster. LaBSE produces cross-lingual embeddings: a Hindi and English article about the same PTI wire story get cosine similarity > 0.82 with zero word overlap.

**Algorithm:**
1. Encode `title + description` as a 768-dim normalised vector
2. Compare against all clusters updated in the last 48 hours (loaded once per batch, not per article)
3. Cosine similarity ≥ 0.82 → join the best matching cluster; update `outlet_count`, `article_count`, `entity_set`
4. Below threshold → create a new cluster, store embedding as `canonical_embedding`

**Why it matters:**
PTI wire stories are distributed to 100+ outlets. Without clustering your feed sees 100 identical signals. With clustering: "1 story, 87 outlets" — the correct virality signal for ranking and deduplication.

---

### Enrichment DB Schema

**`articles` table additions** (from `docs/enrichment_migration.sql`):

| Column | Type | Populated by |
|---|---|---|
| `enriched_at` | timestamptz | Set on completion of any enrichment |
| `word_count` | int | text_stats |
| `reading_time_mins` | float | text_stats |
| `language_detected` | text | language |
| `sentiment` | text | sentiment |
| `sentiment_score` | float | sentiment |
| `keywords` | jsonb | keywords |
| `category` | text | classifier |
| `image_phash` | bigint | images |
| `cluster_id` | uuid | clustering |

**`article_entities`**: One row per entity per article.

**`article_clusters`**: One row per story cluster. Key fields: `headline`, `article_count`, `outlet_count`, `entity_set` (jsonb), `top_entities` (jsonb), `canonical_embedding` (jsonb, 768 floats), `first_seen_at`, `last_seen_at`.

---

## Dependencies

| Library | Purpose |
|---|---|
| `feedparser` | RSS/Atom parsing |
| `trafilatura` | Primary article text extraction |
| `readability-lxml` | Mozilla Readability extraction |
| `newspaper3k` | Fallback text extraction |
| `beautifulsoup4` | Last-resort HTML parsing |
| `supabase` | Database client |
| `mmh3` | MurmurHash3 URL hashing |
| `simhash` | Near-duplicate title detection |
| `lingua-language-detector` | Deterministic multilingual language detection |
| `spacy` + en_core_web_sm + xx_ent_wiki_sm | Named entity recognition |
| `vaderSentiment` | English sentiment analysis |
| `yake` | Unsupervised keyword extraction |
| `transformers` + mDeBERTa | Zero-shot category classification |
| `sentence-transformers` + LaBSE | Cross-lingual story clustering |
| `imagehash` + `Pillow` | Perceptual image hashing |
| `torch` (CPU-only) | Backend for transformers and sentence-transformers |

---

## Running Migrations

Run in Supabase SQL Editor in this order:

```
1. docs/migration.sql                → Layer 1 schema (articles, feeds, pipeline_runs)
2. docs/enrichment_migration.sql     → Layer 2 columns (enriched_at, category, etc.)
3. docs/embedding_migration.sql      → canonical_embedding jsonb on article_clusters
4. docs/cluster_index_migration.sql  → Index on article_clusters.last_seen_at
```

---

## Environment Variables

Set as GitHub Actions secrets and locally in `.env.local`:

```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

Optional overrides (all have defaults):

```
ENRICH_BATCH_SIZE=2000
ENRICH_OFFSET=0
ENRICH_MIN_WORD_COUNT=50
CLUSTER_WINDOW_HOURS=48
CLUSTER_EMBEDDING_THRESHOLD=0.82
CLASSIFY_CONFIDENCE_THRESHOLD=0.40
MAX_CONCURRENT_FEEDS=20
```

---

## Quality Assessment

### Classification (mDeBERTa)

**Strengths:**
- Context-aware: "Modi launches app" → `politics`, not `technology`
- Cross-lingual: Hindi and Tamil articles classified correctly without regional keywords
- No false positives from substring collisions (old "odi" in "commodity" → cricket bug is gone)
- Confidence threshold prevents uncertain labels from polluting the data

**Limitations:**
- Articles below the 0.40 confidence threshold fall back to `general` — typically 15–25% of articles. These are often legitimately borderline (political healthcare article, tech-focused business story).
- The 12 fixed categories miss Indian-specific verticals: agriculture, state politics, real estate, religion/spirituality.
- ~150ms per article; with 2,000-article batches, classification takes ~5 min per shard.

**How to audit:**
```sql
SELECT category, COUNT(*) FROM articles
WHERE enriched_at IS NOT NULL
GROUP BY category ORDER BY COUNT(*) DESC;
```
If `general` exceeds 25%, lower `CLASSIFY_CONFIDENCE_THRESHOLD` to 0.30, or add more specific candidate labels.

### Clustering (LaBSE)

**Strengths:**
- Language-agnostic: Hindi + English articles about the same PTI story correctly grouped
- Threshold 0.82 is calibrated for Indian wire syndication (same story, different outlet: ~0.90–0.98)
- `outlet_count` and `article_count` correctly track story spread

**Limitations:**
- Before `docs/cluster_index_migration.sql` was applied, `fetch_recent_clusters` was timing out — articles during that window were all singletons and won't be retroactively merged.
- Cosine similarity is computed in Python (NumPy dot product per cluster). Fine at 500 active clusters; degrades at 10,000+.
- 48-hour window misses slow-burning stories spanning multiple days.

**How to audit:**
```sql
-- Most viral stories (highest outlet coverage)
SELECT headline, article_count, outlet_count, first_seen_at
FROM article_clusters
ORDER BY outlet_count DESC LIMIT 20;

-- Singleton rate (articles that didn't match any cluster)
SELECT COUNT(*) FROM article_clusters WHERE article_count = 1;

-- Overall clustering rate
SELECT
  COUNT(*) FILTER (WHERE cluster_id IS NOT NULL) AS clustered,
  COUNT(*) FILTER (WHERE cluster_id IS NULL)     AS unclustered
FROM articles WHERE enriched_at IS NOT NULL;
```

A healthy clustering rate for Indian wire news is 40–60%.

---

## Suggested Next Steps

### High Impact / Low Effort

**1. pgvector for fast similarity search**
Currently cosine similarity loads all `canonical_embedding` jsonb arrays into Python memory and uses NumPy. pgvector moves this into PostgreSQL with an HNSW index — millisecond queries regardless of cluster count, and no 48-hour window required.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE article_clusters ADD COLUMN embedding_vec vector(768);
CREATE INDEX ON article_clusters USING hnsw (embedding_vec vector_cosine_ops);
```

This is the single highest-leverage infrastructure improvement.

**2. Expand categories**
Zero-shot classification needs no retraining — just add labels:
- `agriculture` (MSP, crop prices, farmer protests — major Indian vertical)
- `state politics` (separate from national)
- `real estate` (property market, housing)
- `religion` (temple news, festivals, religious politics)

Add to `_CANDIDATE_LABELS` and `_LABEL_TO_CATEGORY` in `enrichment/steps/classifier.py`.

**3. Enrichment quality dashboard**
Add a Supabase view to monitor:
- Category distribution per day
- Clustering rate (clustered vs singleton %)
- Enrichment queue depth (`enriched_at IS NULL` count)
- Average `word_count` (are stubs being skipped correctly?)

### High Impact / Medium Effort

**4. Multilingual sentiment**
VADER is English-only — ~40–50% of articles get `NULL` sentiment. Replacement options:
- **MuRIL** (Google) — trained on 17 Indian languages
- **XLM-R** fine-tuned on multilingual sentiment datasets

Both are HuggingFace models and slot into the existing sentiment step pattern.

**5. Better NER for Indian languages**
`xx_ent_wiki_sm` is generic. **IndicNER** (AI4Bharat) is trained specifically on Hindi, Tamil, Telugu, Bengali, Marathi, Kannada, Malayalam, Gujarati, Punjabi, Urdu — significantly better entity extraction from regional articles, which improves clustering quality (richer `entity_set` on clusters).

**6. Propensity / virality scoring**
All signals are already in the DB:
- `outlet_count` (how many publishers covered the story)
- `article_count` (total articles in cluster)
- `sentiment_score`
- `category`
- `published_at` (recency decay)
- `word_count` (article depth)

A scoring formula (e.g. `virality = log(outlet_count + 1) × recency_decay × category_weight`) computed as a Supabase function or post-enrichment step gives you the core signal for feed ranking and personalisation.

### Medium Impact / Low Effort

**7. Breaking news polling (after Supabase Pro)**
When Pro is purchased:
- Change `ingest_breaking_news.yml` cron to `0 */2 * * *`
- Raise `MAX_CONCURRENT_FEEDS` from 20 → 30
- Covers all 2,150 feeds per run without timeout

**8. Near-real-time enrichment**
Current latency: article published → ingested → enriched → visible in app: ~30–90 min.

To reduce to < 5 min: replace the hourly enrichment cron with a Supabase database webhook that triggers enrichment on `articles.is_crawled` update. Runs instantly when new crawled articles appear rather than waiting for the next `:30`.

**9. Image pHash deduplication**
`image_phash` is computed but unused. Articles with Hamming distance ≤ 10 between pHash values, published within the same time window, are very likely the same story. This adds a fourth dedup layer on top of the existing three.
