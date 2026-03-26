# Layer 2 — Metadata Enrichment

Processes raw articles collected by Layer 1 (RSS ingestion) and adds
structured metadata needed for Layer 3 (propensity scoring).

---

## What it does

For each article, it runs 8 steps in sequence:

| Step | File | Output columns | Notes |
|---|---|---|---|
| 1 | `steps/text_stats.py` | `word_count`, `reading_time_mins` | No dependencies |
| 2 | `steps/language.py` | `language_detected` | Uses langdetect |
| 3 | `steps/sentiment.py` | `sentiment`, `sentiment_score` | English only (VADER) |
| 4 | `steps/ner.py` | `article_entities` table | Requires spaCy model |
| 5 | `steps/keywords.py` | `keywords` | Uses YAKE, language-agnostic |
| 6 | `steps/classifier.py` | `category` | Rule-based, Indian-specific |
| 7 | `steps/images.py` | `image_phash` | Downloads top image |
| 8 | `steps/clustering.py` | `cluster_id`, `article_clusters` table | Depends on step 4 |

---

## Setup

**1. Run the DB migration** (once, in Supabase SQL Editor):
```
docs/enrichment_migration.sql
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Test locally:**
```bash
python enrich.py --dry-run --verbose --batch-size 10
```

**4. Run for real:**
```bash
python enrich.py --batch-size 100
```

---

## How to run

```bash
# Enrich next 100 articles (default)
python enrich.py

# Enrich a larger batch
python enrich.py --batch-size 500

# Dry run — no DB writes
python enrich.py --dry-run --verbose

# Re-process already-enriched articles (after upgrading a model)
python enrich.py --force --batch-size 200
```

---

## Database tables

### New columns on `articles`
| Column | Type | Set by |
|---|---|---|
| `enriched_at` | timestamptz | All runs (always set) |
| `word_count` | integer | text_stats |
| `reading_time_mins` | float | text_stats |
| `language_detected` | text | language |
| `sentiment` | text | sentiment |
| `sentiment_score` | float | sentiment |
| `category` | text | classifier |
| `keywords` | jsonb | keywords |
| `cluster_id` | uuid | clustering |
| `image_phash` | bigint | images |

### `article_entities`
One row per named entity per article. Query examples:
```sql
-- All articles mentioning Narendra Modi this week
SELECT a.title, a.published_at
FROM articles a
JOIN article_entities e ON e.article_id = a.id
WHERE e.entity_text = 'Narendra Modi'
  AND a.published_at > now() - interval '7 days'
ORDER BY a.published_at DESC;
```

### `article_clusters`
One row per story cluster. Key columns:
- `outlet_count` — how many distinct outlets covered this story (virality signal)
- `article_count` — total articles in cluster
- `top_entities` — jsonb [{text, type, count}] for the cluster
- `gnews_data` — reserved for GNews API enrichment (future)

---

## Monitoring

Check enrichment progress in Supabase SQL Editor:
```sql
SELECT * FROM enrichment_health;
```

---

## Architecture notes

- **Steps are pure functions** — text in, structured data out. No DB calls inside steps.
- **`enrichment/db.py`** is the only file that touches Supabase.
- **`enrichment/runner.py`** orchestrates everything: fetches articles, calls steps, persists results.
- **`enrichment/config.py`** is where all settings live. Nothing is hardcoded in step files.
- **Fault tolerant**: each step has its own try/except. A step failure doesn't block the others.
- **Resumable**: `enriched_at IS NULL` is the cursor. Re-start anytime, no duplicates.

---

## Upgrade paths

| What to upgrade | Where to change |
|---|---|
| Better NER model | `SPACY_MODEL` in `config.py` |
| Add Hindi sentiment | Replace VADER in `steps/sentiment.py` with MuRIL |
| Add ML classifier | Swap `steps/classifier.py` — same function signature |
| Add GNews enrichment | Add `steps/gnews.py`, call it in `runner.py` after clustering |
| Bigger batches | `ENRICH_BATCH_SIZE` env var or `--batch-size` CLI flag |
