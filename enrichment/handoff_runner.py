"""
enrichment/handoff_runner.py
════════════════════════════
Crawl + enrich a set of stored articles on a processing runner, in parallel,
and save the results in bulk.

  hand-off mode  `enrich.py --handoff DIR --shard I --shards N` (process.yml):
                 one shard of an ingest run's new articles.
  sweeper mode   `enrich.py --sweep` (enrichment.yml): articles claimed from the
                 queue that are still unenriched hours after ingestion.

PER ARTICLE:
  1. crawl the page if the feed did not carry the body
     (pipeline.crawler.crawl_record — the crawl is deferred from ingestion so
     discovery never times out), in a thread pool: CRAWL_WORKERS pages in
     flight while the CPU does NLP on finished ones;
  2. hash the top image (same pool);
  3. enrich_one() — CPU;
  4. queue the result; every SAVE_EVERY articles one wizer_save_enrichment_batch
     call writes enrichment + crawl columns + entities for all of them;
  5. the body goes to the body store (Supabase Storage, enrichment/body_store.py),
     never to Postgres.

NOTHING IS LOST: an article that crashes, a failed save, or articles left
when the time budget runs out keep enriched_at NULL. In sweeper mode they are
released back to the queue (or, on the final attempt, recorded as a dead
letter that is retried daily — docs/enrichment_queue_v2_migration.sql).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from enrichment import db
from enrichment.body_store import BodyStore
from enrichment.clustering import entity_keys_for_clustering, top_entities_payload
from enrichment.config import ENRICH_MAX_ATTEMPTS
from enrichment.runner import download_and_hash_image, enrich_one
from pipeline.crawler import crawl_record

log = logging.getLogger(__name__)

SAVE_EVERY = int(os.getenv("ENRICH_SAVE_EVERY", "100"))
CRAWL_WORKERS = int(os.getenv("ENRICH_CRAWL_WORKERS", "32"))
_CRAWL_WAIT_S = 240          # crawl_record has its own deadlines; this only guards a wedged thread

# Crawl columns that are saved with the enrichment (full_text goes to Storage).
_CRAWL_COLUMNS = ("title", "title_simhash", "description", "top_image_url", "author",
                  "published_at", "og_tags", "is_crawled", "crawl_strategy")


def _hash_image(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return download_and_hash_image(url)
    except Exception as e:                      # a broken image never costs the article
        log.debug("image hash failed for %s: %s", url, e)
        return None


def _prepare(rec: dict) -> tuple[dict, dict, int | None]:
    """Network part, run in the pool: crawl if needed, then hash the image."""
    crawl: dict = {}
    if not (rec.get("full_text") or "").strip():
        crawl = crawl_record(rec)
    merged = {**rec, **{k: v for k, v in crawl.items() if v not in (None, "")}}
    return merged, crawl, _hash_image(merged.get("top_image_url"))


def _item(article_id, update: dict, crawl: dict, entities: list[dict]) -> dict:
    return {
        "id": article_id,
        "update": update,
        "crawl": {k: crawl[k] for k in _CRAWL_COLUMNS if k in crawl},
        "entities": entities,
        "cluster_entities": top_entities_payload(entities),
        "entity_keys": entity_keys_for_clustering(entities),
    }


def run_handoff_enrichment(records: list[dict], time_budget_minutes: float = 0,
                           dry_run: bool = False, tag: str | None = None,
                           log_run: bool = True) -> dict:
    """
    Crawl + enrich `records`. Returns a summary including `saved_ids` and
    `unprocessed_ids` (articles not started before the budget ran out).
    """
    t0 = time.perf_counter()
    deadline = t0 + time_budget_minutes * 60 if time_budget_minutes > 0 else None
    runner_name = os.getenv("ENRICH_RUNNER_NAME") or tag or "local"
    summary = {"articles": len(records), "processed": 0, "failed": 0, "saved": 0,
               "save_failed": 0, "crawled": 0, "crawl_failed": 0, "left_for_sweeper": 0,
               "stop_reason": "drained", "saved_ids": [], "unprocessed_ids": [],
               "crashed": []}
    run_id = None if (dry_run or not log_run) else db.log_run_start({
        "trigger": os.getenv("GITHUB_EVENT_NAME", "local") + (":" + tag if tag else ""),
        "runner": runner_name,
        "dry_run": False,
        "queue_depth_start": len(records),
    })
    bodies = BodyStore(f"{runner_name}-{os.getenv('GITHUB_RUN_ID', 'local')}", enabled=not dry_run)

    pool = ThreadPoolExecutor(max_workers=max(CRAWL_WORKERS, 1), thread_name_prefix="crawl")
    futures = [pool.submit(_prepare, r) for r in records]
    batch: list[dict] = []
    batch_ids: list = []

    def save() -> None:
        nonlocal batch, batch_ids
        if batch and not dry_run:
            try:
                summary["saved"] += db.save_enrichment_batch(batch)
                summary["saved_ids"].extend(batch_ids)
            except Exception as e:
                log.error("save_enrichment_batch failed for %d articles (left unenriched): %s",
                          len(batch), e)
                summary["save_failed"] += len(batch)
        batch, batch_ids = [], []

    try:
        for i, (rec, fut) in enumerate(zip(records, futures)):
            if deadline is not None and time.perf_counter() >= deadline:
                summary["stop_reason"] = "time_budget"
                summary["unprocessed_ids"] = [r["id"] for r in records[i:]]
                summary["left_for_sweeper"] = len(records) - i
                log.warning("Time budget reached after %d/%d articles", i, len(records))
                break
            try:
                merged, crawl, phash = fut.result(timeout=_CRAWL_WAIT_S)
            except Exception as e:
                log.warning("[%s] crawl/image step failed: %s", rec.get("id"), e)
                merged, crawl, phash = dict(rec), {}, None
            if crawl:
                summary["crawled" if crawl.get("is_crawled") else "crawl_failed"] += 1
            try:
                update, entities = enrich_one({**merged, "_image_phash": phash})
            except Exception as e:
                log.error("[%s] enrich_one crashed: %s", rec.get("id"), e)
                summary["failed"] += 1
                summary["crashed"].append((rec["id"], int(rec.get("enrich_attempts") or 0),
                                           f"{type(e).__name__}: {e}"))
                continue
            summary["processed"] += 1
            bodies.add(merged)
            batch.append(_item(rec["id"], update, crawl, entities))
            batch_ids.append(rec["id"])
            if len(batch) >= SAVE_EVERY:
                save()
        save()
        bodies.flush()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        summary["bodies_stored"] = bodies.uploaded
        summary["duration_s"] = round(time.perf_counter() - t0, 1)
        if run_id is not None:
            db.log_run_finish(run_id, {
                "claimed": summary["articles"], "processed": summary["saved"],
                "failed": summary["failed"] + summary["save_failed"],
                "released": summary["left_for_sweeper"],
                "cluster_joined": 0, "cluster_gray_joined": 0, "cluster_created": 0,
                "cluster_skipped": 0, "duration_s": summary["duration_s"],
                "stop_reason": summary["stop_reason"], "error": None,
            })
        log.info("Crawl + enrich done: %s",
                 {k: v for k, v in summary.items() if k not in ("saved_ids", "unprocessed_ids", "crashed")})
    return summary


def run_sweeper(batch_size: int, min_age_hours: float, time_budget_minutes: float = 0) -> dict:
    """
    Claim articles that are still unenriched `min_age_hours` after ingestion
    (oldest first), crawl + enrich them like a hand-off shard, then hand back
    what was not saved: released claims go straight back to the queue; a crash
    on the final attempt records enrich_error (the article stays unenriched and
    is retried daily as a dead letter).
    """
    claimed = db.claim_batch(batch_size, min_age_hours=min_age_hours)
    log.info("Sweeper: claimed %d articles ingested more than %.1f h ago", len(claimed), min_age_hours)
    if not claimed:
        return {"claimed": 0}
    summary = run_handoff_enrichment(claimed, time_budget_minutes, tag="sweep")
    saved = set(summary["saved_ids"])
    crashed = {aid: (attempts, err) for aid, attempts, err in summary["crashed"]}
    for aid, (attempts, err) in crashed.items():
        if attempts >= ENRICH_MAX_ATTEMPTS:
            db.mark_enrichment_failed(aid, err)
    release = [r["id"] for r in claimed
               if r["id"] not in saved and r["id"] not in crashed]
    summary["released"] = db.release_claims(release) if release else 0
    summary["claimed"] = len(claimed)
    log.info("Sweeper: saved %d, crashed %d, released %d", len(saved), len(crashed), summary["released"])
    return summary
