"""
enrichment/runner.py
════════════════════
Orchestration engine for Layer 2 enrichment.

WHAT THIS FILE DOES:
  This is the engine that runs all enrichment steps on each article,
  handles failures gracefully, and persists results to Supabase.

  It is the enrichment equivalent of pipeline/poller.py.

PROCESSING MODEL:
  Sequential, not concurrent.
  NLP steps (spaCy, YAKE, VADER) are CPU-bound — async concurrency
  doesn't help. Processing 100 articles sequentially with cached models
  is fast enough: ~1-5 seconds per article → 100-500s per 100-article batch.

  GitHub Actions free tier: 2 vCPU, 7 GB RAM, 6h timeout.
  Paid tier allows longer runs. Adjust ENRICH_BATCH_SIZE accordingly.

FAULT TOLERANCE:
  Each step is wrapped in a try/except independently.
  If NER crashes, the article still gets text_stats, language, keywords etc.
  Partial enrichment is saved. enriched_at is always set so the article
  isn't re-processed endlessly.

  The only exception: if saving to DB fails, enriched_at is NOT set —
  the article will be retried on the next run.

STEP EXECUTION ORDER:
  The order matters because some steps depend on earlier ones:

  1. text_stats   — independent, no deps
  2. language     — independent, no deps
  3. sentiment    — depends on language_detected (for English-only gate)
  4. ner          — independent, no deps
  5. keywords     — independent, no deps
  6. classifier   — independent, no deps
  7. images       — independent, no deps
  8. clustering   — depends on ner (needs entities) + title_simhash from article

CLUSTERING NOTE:
  Uses pgvector HNSW — one SQL nearest-neighbour query per article.
  No in-memory cluster list. Sub-millisecond regardless of cluster count.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from enrichment import db
from enrichment.config import ENRICH_BATCH_SIZE, ENRICH_MIN_WORD_COUNT
from enrichment.steps.text_stats  import compute_text_stats
from enrichment.steps.language    import detect_language
from enrichment.steps.sentiment   import analyse_sentiment
from enrichment.steps.ner         import extract_entities
from enrichment.steps.keywords    import extract_keywords
from enrichment.steps.classifier  import classify_article
from enrichment.steps.images      import download_and_hash_image
from enrichment.steps.clustering  import find_or_create_cluster

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE ARTICLE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one(
    article: dict,
) -> tuple[dict, list[dict], dict | None, str]:
    """
    Run all enrichment steps on a single article.

    Args:
      article: Article dict from the DB

    Returns:
      (article_update, entities, cluster_payload, cluster_action)

      article_update:  dict of columns to write to articles table
      entities:        list of entity dicts for article_entities table
      cluster_payload: dict for create_cluster / update_cluster (or None)
      cluster_action:  'join' | 'create' | 'skip'
    """
    article_id  = article.get("id", "")
    title       = article.get("title") or ""
    description = article.get("description") or ""
    full_text   = article.get("full_text") or ""
    image_url   = article.get("top_image_url") or ""
    language_code = article.get("language_code") or ""

    update: dict = {}

    # ── Step 1: Text stats ────────────────────────────────────────────────────
    try:
        stats = compute_text_stats(full_text, description)
        update.update(stats)
    except Exception as e:
        log.warning("[%s] text_stats failed: %s", article_id[:8], e)

    # ── Word count gate ───────────────────────────────────────────────────────
    # Articles below ENRICH_MIN_WORD_COUNT are stubs/briefs with no enrichment
    # signal. Return early with just text_stats so they don't clog NER/keywords.
    # enriched_at is still set so they don't re-enter the queue.
    word_count = update.get("word_count") or 0
    if word_count < ENRICH_MIN_WORD_COUNT:
        log.debug("[%s] Skipping enrichment — too short (%d words)", article_id, word_count)
        return update, [], None, "skip"

    # ── Step 2: Language detection ────────────────────────────────────────────
    language_detected = None
    try:
        language_detected = detect_language(full_text, description, title)
        update["language_detected"] = language_detected
    except Exception as e:
        log.warning("[%s] language detection failed: %s", article_id[:8], e)

    # ── Step 3: Sentiment ─────────────────────────────────────────────────────
    try:
        sentiment_result = analyse_sentiment(title, description, language_detected)
        update.update(sentiment_result)
    except Exception as e:
        log.warning("[%s] sentiment failed: %s", article_id[:8], e)

    # ── Step 4: NER ───────────────────────────────────────────────────────────
    entities: list[dict] = []
    try:
        entities = extract_entities(title, full_text, description, language_detected)
    except Exception as e:
        log.warning("[%s] NER failed: %s", article_id[:8], e)

    # ── Step 5: Keywords ──────────────────────────────────────────────────────
    try:
        keywords = extract_keywords(title, full_text, description)
        update["keywords"] = keywords if keywords else None
    except Exception as e:
        log.warning("[%s] keyword extraction failed: %s", article_id[:8], e)

    # ── Step 6: Category classification ──────────────────────────────────────
    try:
        category = classify_article(title, description, article.get("iab_tier1") or "", full_text)
        update["category"] = category
    except Exception as e:
        log.warning("[%s] classifier failed: %s", article_id[:8], e)

    # ── Step 7: Image pHash ───────────────────────────────────────────────────
    try:
        image_phash = download_and_hash_image(image_url)
        update["image_phash"] = image_phash
    except Exception as e:
        log.warning("[%s] image hashing failed: %s", article_id[:8], e)

    # ── Step 8: Story clustering ──────────────────────────────────────────────
    # Uses pgvector HNSW: one SQL query per article, no in-memory cluster list.
    cluster_id      = None
    cluster_payload = None
    cluster_action  = "skip"
    try:
        cluster_id, cluster_payload, cluster_action = find_or_create_cluster(
            article  = article,
            entities = entities,
        )
        if cluster_id:
            update["cluster_id"] = cluster_id
    except Exception as e:
        log.warning("[%s] clustering failed: %s", article_id[:8], e)

    return update, entities, cluster_payload, cluster_action


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_enrichment(
    batch_size: int = ENRICH_BATCH_SIZE,
    dry_run: bool   = False,
    force: bool     = False,
    offset: int     = 0,
) -> dict:
    """
    Main entry point. Fetches a batch of unenriched articles and processes them.

    Args:
      batch_size: How many articles to process in this run
      dry_run:    If True, compute enrichment but do NOT write to DB
      force:      If True, re-process already-enriched articles

    Returns:
      Summary dict with counts of processed/failed/clustered articles.
    """
    log.info("═══ Enrichment run start | batch=%d | offset=%d | dry_run=%s | force=%s ═══",
             batch_size, offset, dry_run, force)
    t_start = time.perf_counter()

    summary = {
        "processed":   0,
        "failed":      0,
        "clustered":   0,
        "new_clusters": 0,
        "skipped_cluster": 0,
        "duration_s":  0.0,
    }

    # ── Fetch articles ────────────────────────────────────────────────────────
    if force:
        articles = db.fetch_unenriched_batch_forced(batch_size, offset=offset)
    else:
        articles = db.fetch_unenriched_batch(batch_size, offset=offset)

    if not articles:
        log.info("No unenriched articles found — nothing to do")
        return summary

    log.info("Fetched %d articles to enrich", len(articles))

    # ── Process each article ──────────────────────────────────────────────────
    for i, article in enumerate(articles, 1):
        article_id = article.get("id", "")
        title      = (article.get("title") or "")[:60]

        log.debug("[%d/%d] Enriching: %s…", i, len(articles), title)

        try:
            article_update, entities, cluster_payload, cluster_action = enrich_one(article)
        except Exception as e:
            log.error("[%s] enrich_one crashed: %s", article_id, e)
            summary["failed"] += 1
            # Mark as enriched anyway so it doesn't block the queue forever
            if not dry_run:
                db.save_article_enrichment(article_id, {})
            continue

        if dry_run:
            log.info(
                "DRY RUN [%s] category=%s lang=%s words=%s entities=%d action=%s",
                article_id,
                article_update.get("category"),
                article_update.get("language_detected"),
                article_update.get("word_count"),
                len(entities),
                cluster_action,
            )
            summary["processed"] += 1
            continue

        # ── Persist cluster ───────────────────────────────────────────────────
        if cluster_action == "create" and cluster_payload:
            new_cluster_id = db.create_cluster(cluster_payload)
            if new_cluster_id:
                article_update["cluster_id"] = new_cluster_id
                summary["new_clusters"] += 1

        elif cluster_action == "join" and cluster_payload:
            cluster_id = article_update.get("cluster_id")
            if cluster_id:
                db.update_cluster(cluster_id, cluster_payload)
                summary["clustered"] += 1

        else:
            summary["skipped_cluster"] += 1

        # ── Persist entities ──────────────────────────────────────────────────
        if entities:
            db.save_entities(article_id, entities)

        # ── Persist article enrichment ────────────────────────────────────────
        saved = db.save_article_enrichment(article_id, article_update)
        if saved:
            summary["processed"] += 1
        else:
            summary["failed"] += 1

    # ── Final summary ─────────────────────────────────────────────────────────
    duration = round(time.perf_counter() - t_start, 2)
    summary["duration_s"] = duration

    log.info(
        "═══ Enrichment complete | processed=%d | failed=%d | "
        "clustered=%d | new_clusters=%d | %.1fs ═══",
        summary["processed"], summary["failed"],
        summary["clustered"], summary["new_clusters"],
        duration,
    )
    return summary
