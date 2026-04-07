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
                    also produces: sentiment_stats (full score breakdown)
  4. ner          — independent, no deps
                    also produces: ai_region (top GPE entities), ai_org (top ORG entities)
  5. keywords     — independent, no deps
  6. classifier   — independent, no deps
  7. tags         — reuses mDeBERTa (same model as step 6, already in memory)
  8. summary      — independent, no model (extractive, regex-based)
  9. images       — independent, no deps
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from enrichment import db
from enrichment.config import ENRICH_BATCH_SIZE, ENRICH_MIN_WORD_COUNT, ENRICH_SUPPORTED_LANGUAGES
from enrichment.steps.text_stats  import compute_text_stats
from enrichment.steps.language    import detect_language
from enrichment.steps.sentiment   import analyse_sentiment
from enrichment.steps.ner         import extract_entities
from enrichment.steps.keywords    import extract_keywords
from enrichment.steps.classifier  import classify_article, classify_tags
from enrichment.steps.summarizer  import summarize_article
from enrichment.steps.images      import download_and_hash_image

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE ARTICLE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich_one(
    article: dict,
) -> tuple[dict, list[dict]]:
    """
    Run all enrichment steps on a single article.

    Args:
      article: Article dict from the DB

    Returns:
      (article_update, entities)

      article_update:  dict of columns to write to articles table
      entities:        list of entity dicts for article_entities table
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
        return update, []

    # ── Step 2: Language detection ────────────────────────────────────────────
    language_detected = None
    try:
        language_detected = detect_language(full_text, description, title)
        update["language_detected"] = language_detected
    except Exception as e:
        log.warning("[%s] language detection failed: %s", article_id[:8], e)

    # If language detection failed (None), default to "en" so NLP steps still run.
    # Silently blocking all enrichment on detection failure was causing 1,165 articles
    # to get enriched_at set but no category, sentiment, NER, or keywords.
    lang_base = (language_detected or "en").split("-")[0].lower()
    rich_enrich = not ENRICH_SUPPORTED_LANGUAGES or lang_base in ENRICH_SUPPORTED_LANGUAGES

    # ── Step 3: Sentiment (English + Hindi only) ──────────────────────────────
    if rich_enrich:
        try:
            sentiment_result = analyse_sentiment(title, description, language_detected)
            update.update(sentiment_result)
        except Exception as e:
            log.warning("[%s] sentiment failed: %s", article_id[:8], e)

    # ── Step 4: NER (English + Hindi only) ───────────────────────────────────
    entities: list[dict] = []
    if rich_enrich:
        try:
            entities = extract_entities(title, full_text, description, language_detected)
        except Exception as e:
            log.warning("[%s] NER failed: %s", article_id[:8], e)

    # Derive ai_region and ai_org from NER output — free, no extra compute.
    # ai_region: top GPE (geopolitical) entities by salience → geographic focus
    # ai_org:    top ORG entities by salience → organisations mentioned
    if entities:
        ai_region = [
            e["entity_text"].lower()
            for e in entities if e["entity_type"] == "GPE"
        ][:5]
        ai_org = [
            e["entity_text"].lower()
            for e in entities if e["entity_type"] == "ORG"
        ][:5]
        if ai_region:
            update["ai_region"] = ai_region
        if ai_org:
            update["ai_org"] = ai_org

    # ── Step 5: Keywords (English + Hindi only) ──────────────────────────────
    if rich_enrich:
        try:
            keywords = extract_keywords(title, full_text, description, language_detected)
            update["keywords"] = keywords if keywords else None
        except Exception as e:
            log.warning("[%s] keyword extraction failed: %s", article_id[:8], e)

    # ── Step 6: Category classification (all languages — mDeBERTa is multilingual) ──
    # mDeBERTa handles 100+ languages natively via cross-lingual embeddings.
    # English labels work for Tamil/Kannada/Telugu/Malayalam input — the model
    # maps all languages into the same semantic space. No rich_enrich gate needed.
    try:
        category = classify_article(title, description, full_text)
        update["category"] = category
    except Exception as e:
        log.warning("[%s] classifier failed: %s", article_id[:8], e)

    # ── Step 7: AI topic tags (all languages — same mDeBERTa pipeline) ───────
    # Reuses the model already loaded by step 6 — no extra RAM.
    try:
        tags = classify_tags(title, description, full_text)
        update["ai_tag"] = tags if tags else None
    except Exception as e:
        log.warning("[%s] tag classification failed: %s", article_id[:8], e)

    # ── Step 8: Extractive summary (all languages) ───────────────────────────
    # No model — first 3 sentences of full_text or description if rich.
    try:
        summary = summarize_article(full_text, description)
        update["ai_summary"] = summary
    except Exception as e:
        log.warning("[%s] summarization failed: %s", article_id[:8], e)

    # ── Step 9: Image pHash ───────────────────────────────────────────────────
    try:
        image_phash = download_and_hash_image(image_url)
        update["image_phash"] = image_phash
    except Exception as e:
        log.warning("[%s] image hashing failed: %s", article_id[:8], e)

    return update, entities


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
        "processed":  0,
        "failed":     0,
        "duration_s": 0.0,
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
            article_update, entities = enrich_one(article)
        except Exception as e:
            log.error("[%s] enrich_one crashed: %s", article_id, e)
            summary["failed"] += 1
            # Mark as enriched anyway so it doesn't block the queue forever
            if not dry_run:
                db.save_article_enrichment(article_id, {})
            continue

        if dry_run:
            log.info(
                "DRY RUN [%s] category=%s lang=%s words=%s entities=%d",
                article_id,
                article_update.get("category"),
                article_update.get("language_detected"),
                article_update.get("word_count"),
                len(entities),
            )
            summary["processed"] += 1
            continue

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
        "═══ Enrichment complete | processed=%d | failed=%d | %.1fs ═══",
        summary["processed"], summary["failed"],
        duration,
    )
    return summary
