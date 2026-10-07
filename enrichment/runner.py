"""
enrichment/runner.py
════════════════════
Orchestration engine for Layer 2 enrichment.

WHAT THIS FILE DOES:
  Claims a batch of articles from the work queue, runs every enrichment step
  on each one, puts each article into its story cluster, and persists the
  results to Supabase. It is the enrichment equivalent of pipeline/poller.py.

WORK QUEUE (docs/enrichment_queue_v2_migration.sql):
  Articles are CLAIMED, not paged by offset. wizer_claim_enrichment_batch()
  leases rows with FOR UPDATE SKIP LOCKED, so any number of concurrent runs
  (the two matrix shards, overlapping workflow_run triggers, a manual run)
  always work on disjoint articles. A run that stops early (time budget,
  SIGTERM from a cancelled job, Ctrl+C) releases its unprocessed claims; a
  run that dies outright leaves leases that expire after
  ENRICH_CLAIM_LEASE_MINUTES. Either way nothing is lost or stuck.
  The queue is oldest-first by ingestion time with no age limit, and includes
  failed crawls: every ingested article is enriched.

PROCESSING MODEL:
  Sequential, not concurrent — the NLP steps are CPU-bound, so async buys
  nothing. Models load once per process and are reused for the whole batch.
  The one batched step is the clustering embedding: the whole claimed batch
  is embedded up front in mini-batches (4-6× faster on CPU than one call per
  article).

FAULT TOLERANCE:
  Each step is wrapped in its own try/except: if NER crashes the article
  still gets text_stats, language, keywords etc.
  enriched_at is written LAST, only after entities and the cluster
  assignment are saved — if anything before it fails, the article stays in
  the queue and is retried (the cluster assignment is idempotent, so a retry
  cannot double-count it).
  If enrich_one() itself raises, the article is retried on a later run. On
  its final regular attempt (ENRICH_MAX_ATTEMPTS) enrich_error records why;
  it stays unenriched and the queue retries it once a day
  (ENRICH_RETRY_HOURS, ENRICH_MAX_RETRIES), so one bad article can never
  block the queue and a transient failure never costs an article.

STEP EXECUTION ORDER:
   1. text_stats   — word count, reading time
      < ENRICH_MIN_WORD_COUNT words (briefs, failed crawls): every step runs
      on headline + description except keywords
   2. language     — detect actual language of the body
      GATE: ENRICH_SUPPORTED_LANGUAGES → sentiment / NER / keywords only for these
   3. sentiment    — multilingual distilbert (+ sentiment_stats)
   4. ner          — spaCy entities (+ ai_region, ai_org)
   5. keywords     — YAKE (not on short text)
   6. classifier   — mDeBERTa zero-shot category (all languages)
   7. tags         — mDeBERTa multi-label topic tags (same model)
   8. summary      — extractive, no model
   9. images       — top-image perceptual hash
  10. clustering   — ALL articles, ALL languages, short ones included: the
                     story a 40-word wire brief belongs to is exactly what
                     outlet_count measures. Uses the batch embedding, the NER
                     entities (step 4) and the image hash (step 9) as evidence.
"""

from __future__ import annotations

import logging
import os
import signal
import time

from enrichment import db
from enrichment.clustering import assign_cluster
from enrichment.config import (
    CATEGORY_HEAD,
    TAG_HEAD,
    CLUSTERING_ENABLED,
    ENRICH_BATCH_SIZE,
    ENRICH_MAX_ATTEMPTS,
    ENRICH_MIN_WORD_COUNT,
    ENRICH_SUPPORTED_LANGUAGES,
    ENRICH_TIME_BUDGET_MINUTES,
)
from enrichment.steps.text_stats  import compute_text_stats
from enrichment.steps.language    import detect_language
from enrichment.steps.sentiment   import analyse_sentiment
from enrichment.steps.ner         import extract_entities
from enrichment.steps.keywords    import extract_keywords
from enrichment.steps.classifier  import classify_article, classify_tags
from enrichment.steps.summarizer  import summarize_article
from enrichment.steps.images      import download_and_hash_image
from enrichment.steps.embedding   import build_embedding_text, embed_texts, CLUSTER_EMBEDDING_MODEL
from enrichment.steps             import category_head, tag_head

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE ARTICLE ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def _vector(article: dict, title: str, description: str, full_text: str):
    """The article's E5 vector: the caller's ("_category_vector"), else embedded once and kept."""
    if "_category_vector" not in article or article["_category_vector"] is None:
        text = build_embedding_text(title, description, full_text)
        article["_category_vector"] = embed_texts([text])[0] if text else None
    return article["_category_vector"]


def _category(article: dict, title: str, description: str, full_text: str) -> str:
    """Category from the E5 head when possible, else the mDeBERTa zero-shot classifier."""
    if CATEGORY_HEAD and category_head.available(CLUSTER_EMBEDDING_MODEL):
        result = category_head.predict(_vector(article, title, description, full_text), CLUSTER_EMBEDDING_MODEL)
        if result is not None:
            return result[0]
    return classify_article(title, description, full_text)


def _tags(article: dict, title: str, description: str, full_text: str) -> list[str]:
    """Topic tags from the E5 tag head when possible, else the mDeBERTa zero-shot tagger."""
    if TAG_HEAD and tag_head.available(CLUSTER_EMBEDDING_MODEL):
        result = tag_head.predict(_vector(article, title, description, full_text), CLUSTER_EMBEDDING_MODEL)
        if result is not None:
            return result
    return classify_tags(title, description, full_text)


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
    article_id  = article.get("id")
    title       = article.get("title") or ""
    description = article.get("description") or ""
    full_text   = article.get("full_text") or ""
    image_url   = article.get("top_image_url") or ""

    update: dict = {}

    # ── Step 1: Text stats ────────────────────────────────────────────────────
    try:
        stats = compute_text_stats(full_text, description)
        update.update(stats)
    except Exception as e:
        log.warning("[%s] text_stats failed: %s", article_id, e)

    # ── Short text (briefs, failed crawls with only an RSS description) ──────
    # Every step below works on headline + description, so short articles get
    # the full enrichment except keyword extraction (noise on a few words).
    # They used to return here with text stats only — "enriched" with nothing.
    word_count = update.get("word_count") or 0
    short_text = word_count < ENRICH_MIN_WORD_COUNT

    # ── Step 2: Language detection ────────────────────────────────────────────
    language_detected = None
    try:
        language_detected = detect_language(full_text, description, title)
        update["language_detected"] = language_detected
    except Exception as e:
        log.warning("[%s] language detection failed: %s", article_id, e)

    # If language detection failed (None), default to "en" so NLP steps still run.
    # Silently blocking all enrichment on detection failure was causing 1,165 articles
    # to get enriched_at set but no category, sentiment, NER, or keywords.
    lang_base = (language_detected or "en").split("-")[0].lower()
    rich_enrich = not ENRICH_SUPPORTED_LANGUAGES or lang_base in ENRICH_SUPPORTED_LANGUAGES
    if (language_detected or "").endswith("-latn"):
        # Romanised Hindi: the en/hi NER, sentiment and keyword models expect
        # English or Devanagari; classifier, summary and clustering still run.
        rich_enrich = False

    # ── Step 3: Sentiment (English + Hindi only) ──────────────────────────────
    if rich_enrich:
        try:
            sentiment_result = analyse_sentiment(title, description, language_detected)
            update.update(sentiment_result)
        except Exception as e:
            log.warning("[%s] sentiment failed: %s", article_id, e)

    # ── Step 4: NER (English + Hindi only) ───────────────────────────────────
    entities: list[dict] = []
    if rich_enrich:
        try:
            entities = extract_entities(title, full_text, description, language_detected)
        except Exception as e:
            log.warning("[%s] NER failed: %s", article_id, e)

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

    # ── Step 5: Keywords (English + Hindi only, not on short text) ───────────
    if rich_enrich and not short_text:
        try:
            keywords = extract_keywords(title, full_text, description, language_detected)
            update["keywords"] = keywords if keywords else None
        except Exception as e:
            log.warning("[%s] keyword extraction failed: %s", article_id, e)

    # ── Step 6: Category (all languages) ─────────────────────────────────────
    # The linear head on the multilingual-E5 vector (category_head.py), measured
    # on the 13-language gold set; mDeBERTa zero-shot is the fallback when the
    # head is off or its model file is missing. The vector is the clustering
    # embedding when the caller has one ("_category_vector"), else computed here.
    try:
        update["category"] = _category(article, title, description, full_text)
    except Exception as e:
        log.warning("[%s] classifier failed: %s", article_id, e)

    # ── Step 7: AI topic tags ────────────────────────────────────────────────
    # The per-tag heads on the same E5 vector (tag_head.py); mDeBERTa zero-shot
    # only as the fallback. With both heads present mDeBERTa is never loaded.
    try:
        tags = _tags(article, title, description, full_text)
        update["ai_tag"] = tags if tags else None
    except Exception as e:
        log.warning("[%s] tag classification failed: %s", article_id, e)

    # ── Step 8: Extractive summary (all languages) ───────────────────────────
    # No model — first 3 sentences of full_text or description if rich.
    try:
        summary = summarize_article(full_text, description)
        update["ai_summary"] = summary
    except Exception as e:
        log.warning("[%s] summarization failed: %s", article_id, e)

    # ── Step 9: Image pHash ───────────────────────────────────────────────────
    # The hand-off runner downloads images in a thread pool ahead of the CPU
    # work and passes the hash in as "_image_phash" (enrichment/handoff_runner.py).
    if "_image_phash" in article:
        update["image_phash"] = article["_image_phash"]
    else:
        try:
            image_phash = download_and_hash_image(image_url)
            update["image_phash"] = image_phash
        except Exception as e:
            log.warning("[%s] image hashing failed: %s", article_id, e)

    return update, entities


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

class _StopFlag:
    """
    Set by SIGTERM / SIGINT. GitHub Actions sends SIGINT then SIGTERM when a
    job is cancelled or times out; finishing the current article and releasing
    the rest is far better than dying mid-write.
    """

    def __init__(self) -> None:
        self.reason: str | None = None

    def __call__(self, signum, _frame) -> None:
        if self.reason is None:
            log.warning("Received signal %s — finishing current article, then stopping", signum)
        self.reason = "signal"


def _install_signal_handlers(flag: _StopFlag) -> dict:
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, flag)
        except (ValueError, OSError):      # not in main thread / unsupported
            pass
    return previous


def _restore_signal_handlers(previous: dict) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def _embed_batch(articles: list[dict]) -> list[list[float] | None]:
    """
    Embed the batch up front (see module docstring) — but only articles that
    the clustering job (cluster.py run) has not already placed. Those get None
    and skip the clustering RPC: re-assigning them would only return 'existing'.
    """
    todo = [i for i, a in enumerate(articles) if not a.get("cluster_id")]
    vectors: list[list[float] | None] = [None] * len(articles)
    if not todo:
        log.info("All %d articles already clustered by the clustering job", len(articles))
        return vectors
    t0 = time.perf_counter()
    embedded = embed_texts([
        build_embedding_text(articles[i].get("title"), articles[i].get("description"),
                             articles[i].get("full_text"))
        for i in todo
    ])
    for i, v in zip(todo, embedded):
        vectors[i] = v
    log.info("Embedded %d/%d unclustered articles in %.1fs (%d already clustered)",
             sum(v is not None for v in embedded), len(todo), time.perf_counter() - t0,
             len(articles) - len(todo))
    return vectors


def _new_summary() -> dict:
    return {
        "claimed":             0,
        "processed":           0,
        "failed":              0,
        "released":            0,
        "cluster_joined":      0,
        "cluster_gray_joined": 0,
        "cluster_created":     0,
        "cluster_skipped":     0,
        "queue_depth_start":   None,
        "stop_reason":         "drained",
        "duration_s":          0.0,
    }


def run_enrichment(
    batch_size: int = ENRICH_BATCH_SIZE,
    dry_run: bool   = False,
    force: bool     = False,
    offset: int     = 0,
    time_budget_minutes: float = ENRICH_TIME_BUDGET_MINUTES,
    min_age_hours: float = 0,
) -> dict:
    """
    Main entry point: claim a batch, enrich + cluster each article, persist.

    min_age_hours > 0 is the SWEEPER: only articles ingested at least that long
    ago are claimed (fresher ones belong to the hand-off runners).

    Args:
      batch_size:          articles to process in this run
      dry_run:             compute everything, write nothing (reads the queue
                           without claiming it, and skips cluster assignment —
                           that step IS a write)
      force:               re-process already-enriched articles (no claims;
                           paged with `offset`). Clustering is idempotent, so
                           already-clustered articles keep their cluster.
      offset:              only used with --force / --dry-run paging
      time_budget_minutes: stop taking new articles after this long and
                           release the rest (0 = unlimited)

    Returns a summary dict (see _new_summary) — also written to enrichment_runs.
    """
    log.info("═══ Enrichment run start | batch=%d | dry_run=%s | force=%s | clustering=%s ═══",
             batch_size, dry_run, force, CLUSTERING_ENABLED)
    t_start = time.perf_counter()
    summary = _new_summary()
    use_queue = not dry_run and not force

    run_id = None
    if not dry_run:
        summary["queue_depth_start"] = db.queue_depth()
        run_id = db.log_run_start({
            "trigger":           os.getenv("GITHUB_EVENT_NAME", "local"),
            "runner":            os.getenv("ENRICH_RUNNER_NAME") or None,
            "dry_run":           False,
            "queue_depth_start": summary["queue_depth_start"],
        })

    stop = _StopFlag()
    previous_handlers = _install_signal_handlers(stop)
    pending_ids: list = []
    try:
        # ── Fetch / claim ────────────────────────────────────────────────────
        if use_queue:
            articles = db.claim_batch(batch_size, min_age_hours=min_age_hours) if min_age_hours                 else db.claim_batch(batch_size)
        elif force:
            articles = db.fetch_unenriched_batch_forced(batch_size, offset=offset)
        else:
            articles = db.fetch_unenriched_batch(batch_size, offset=offset)
        summary["claimed"] = len(articles)
        if not articles:
            log.info("Queue empty — nothing to do (queue depth: %s)", summary["queue_depth_start"])
            return summary
        log.info("%s %d articles (queue depth at start: %s)",
                 "Claimed" if use_queue else "Fetched", len(articles), summary["queue_depth_start"])
        # The queue hands out the oldest INGESTED articles; within the batch we
        # process in publication order: online clustering groups a story better
        # when its articles arrive in time order (strict pair F1 0.675 vs 0.646
        # newest-first on the calibration set, docs/CLUSTERING.md).
        articles.sort(key=lambda a: a.get("published_at") or "")
        pending_ids = [a["id"] for a in articles]

        embeddings = _embed_batch(articles) if CLUSTERING_ENABLED else [None] * len(articles)
        deadline = (t_start + time_budget_minutes * 60) if time_budget_minutes > 0 else None

        for i, (article, embedding) in enumerate(zip(articles, embeddings), 1):
            if stop.reason:
                summary["stop_reason"] = stop.reason
                break
            if deadline is not None and time.perf_counter() >= deadline:
                summary["stop_reason"] = "time_budget"
                log.warning("Time budget of %.0f min reached after %d/%d articles",
                            time_budget_minutes, i - 1, len(articles))
                break

            log.debug("[%d/%d] Enriching: %s…", i, len(articles), (article.get("title") or "")[:60])
            _process_article(article, embedding, dry_run, summary)
            pending_ids.pop(0)

        if use_queue and pending_ids:
            summary["released"] = db.release_claims(pending_ids)
            log.info("Released %d unprocessed claims back to the queue", summary["released"])
            pending_ids = []
        return summary

    except KeyboardInterrupt:
        summary["stop_reason"] = "signal"
        raise
    except Exception as e:
        summary["stop_reason"] = "error"
        summary["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        if use_queue and pending_ids:          # crashed mid-batch
            summary["released"] = db.release_claims(pending_ids)
        _restore_signal_handlers(previous_handlers)
        summary["duration_s"] = round(time.perf_counter() - t_start, 2)
        db.log_run_finish(run_id, {
            k: summary.get(k) for k in (
                "claimed", "processed", "failed", "released",
                "cluster_joined", "cluster_gray_joined", "cluster_created",
                "cluster_skipped", "duration_s", "stop_reason", "error",
            )
        })
        log.info(
            "═══ Enrichment complete | processed=%d failed=%d released=%d | clusters: "
            "joined=%d gray=%d created=%d skipped=%d | %s | %.1fs ═══",
            summary["processed"], summary["failed"], summary["released"],
            summary["cluster_joined"], summary["cluster_gray_joined"],
            summary["cluster_created"], summary["cluster_skipped"],
            summary["stop_reason"], summary["duration_s"],
        )


def _process_article(article: dict, embedding, dry_run: bool, summary: dict) -> None:
    """Enrich, cluster and persist ONE article, updating the run summary."""
    article_id = article.get("id")
    try:
        article_update, entities = enrich_one({**article, "_category_vector": embedding})
    except Exception as e:
        log.error("[%s] enrich_one crashed: %s", article_id, e)
        summary["failed"] += 1
        attempts = article.get("enrich_attempts") or 0
        if not dry_run and attempts >= ENRICH_MAX_ATTEMPTS:
            # Final attempt: park it so it stops coming back. Earlier attempts
            # keep their claim; the lease expires and a later run retries.
            db.mark_enrichment_failed(article_id, f"{type(e).__name__}: {e}")
        return

    language = article_update.get("language_detected") or article.get("language_code")

    if dry_run:
        log.info(
            "DRY RUN [%s] category=%s lang=%s words=%s entities=%d embedded=%s",
            article_id, article_update.get("category"), language,
            article_update.get("word_count"), len(entities), embedding is not None,
        )
        summary["processed"] += 1
        return

    # ── Persist entities ──────────────────────────────────────────────────────
    if entities:
        db.save_entities(article_id, entities)

    # ── Story clustering (writes articles.cluster_id itself, atomically) ─────
    result = None
    if article.get("cluster_id"):
        summary["cluster_existing"] = summary.get("cluster_existing", 0) + 1
    elif CLUSTERING_ENABLED:
        result = assign_cluster(
            article, embedding, entities,
            image_phash=article_update.get("image_phash"),
            language=language,
        )
    if result is None:
        if not article.get("cluster_id"):
            summary["cluster_skipped"] += 1
    elif result.action == "seed":
        summary["cluster_created"] += 1
    elif result.action == "join":
        summary["cluster_joined"] += 1
    elif result.action == "gray_join":
        summary["cluster_gray_joined"] += 1
    if result is not None:
        log.debug("[%s] cluster %s (%s, sim=%s, outlets=%s)", article_id, result.cluster_id,
                  result.action, result.similarity, result.outlet_count)

    # ── Persist article enrichment — enriched_at is written LAST ─────────────
    if db.save_article_enrichment(article_id, article_update):
        summary["processed"] += 1
    else:
        summary["failed"] += 1
