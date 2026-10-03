# Layer 2 — Metadata Enrichment

This package turns crawled articles (Layer 1) into enriched articles grouped into story
clusters.

| module | role |
|---|---|
| `runner.py` | claims a batch from the work queue, runs the steps, persists results (`enriched_at` last), and handles crashes, the time budget and signals |
| `db.py` | every Supabase call: queue claim/release, enrichment writes, cluster RPCs, run log |
| `clustering.py` | builds the `wizer_assign_cluster` call: evidence entities, image hash, thresholds |
| `cluster_maintenance.py` | merge planning and the merge → reconcile → prune pass |
| `config.py` | every setting, each documented where it is defined |
| `steps/` | pure functions, one per step: no DB calls, models lazy-loaded once per process |

Steps, in order (see `runner.py` for the gates):

| # | file | output |
|---|---|---|
| 1 | `steps/text_stats.py` | `word_count`, `reading_time_mins` |
| 2 | `steps/language.py` | `language_detected` (lingua) |
| 3 | `steps/sentiment.py` | `sentiment`, `sentiment_score`, `sentiment_stats` (multilingual DistilBERT) |
| 4 | `steps/ner.py` | `article_entities`, `ai_region`, `ai_org` (spaCy) |
| 5 | `steps/keywords.py` | `keywords` (YAKE) |
| 6 | `steps/classifier.py` | `category` (mDeBERTa zero-shot; calibrated, see `tools/classifier_eval/`) |
| 7 | `steps/classifier.py` | `ai_tag` |
| 8 | `steps/summarizer.py` | `ai_summary` |
| 9 | `steps/images.py` | `image_phash` |
| 10 | `steps/embedding.py` + `clustering.py` | `cluster_id` (see `docs/CLUSTERING.md`) |

For setup, operations and the schema, see the top-level `README.md` and `docs/MIGRATIONS.md`.
