"""
enrichment/handoff_runner.py
════════════════════════════
Enrich one shard of an ingest run's hand-off file
(`python enrich.py --handoff DIR --shard I --shards N`, process.yml).

The articles arrive WITH their crawled body (pipeline/handoff.py), so nothing
is read from the database. Results are written a batch at a time with
wizer_save_enrichment_batch — one round trip per SAVE_EVERY articles instead
of three per article — and top images are downloaded in a thread pool while
the CPU works on the NLP steps (the download was ~27 % of the time per
article, almost all of it network wait).

NOTHING IS LOST: an article that crashes enrich_one(), a batch whose save
fails, or articles left over when the time budget runs out all keep
enriched_at NULL. The sweeper (enrich.py without --handoff: the claim queue)
enriches them later from title + description.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from enrichment import db
from enrichment.clustering import entity_keys_for_clustering, top_entities_payload
from enrichment.config import IMAGE_DOWNLOAD_TIMEOUT
from enrichment.runner import download_and_hash_image, enrich_one

log = logging.getLogger(__name__)

SAVE_EVERY = int(os.getenv("ENRICH_SAVE_EVERY", "100"))
IMAGE_WORKERS = int(os.getenv("ENRICH_IMAGE_WORKERS", "8"))


def _hash_image(url: str | None) -> int | None:
    if not url:
        return None
    try:
        return download_and_hash_image(url)
    except Exception as e:                      # a broken image never costs the article
        log.debug("image hash failed for %s: %s", url, e)
        return None


def _item(article_id, update: dict, entities: list[dict]) -> dict:
    return {
        "id": article_id,
        "update": update,
        "entities": entities,
        "cluster_entities": top_entities_payload(entities),
        "entity_keys": entity_keys_for_clustering(entities),
    }


def run_handoff_enrichment(records: list[dict], time_budget_minutes: float = 0,
                           dry_run: bool = False) -> dict:
    t0 = time.perf_counter()
    deadline = t0 + time_budget_minutes * 60 if time_budget_minutes > 0 else None
    summary = {"articles": len(records), "processed": 0, "failed": 0, "saved": 0,
               "save_failed": 0, "left_for_sweeper": 0, "stop_reason": "drained"}
    run_id = None if dry_run else db.log_run_start({
        "trigger": os.getenv("GITHUB_EVENT_NAME", "local") + ":handoff",
        "runner": os.getenv("ENRICH_RUNNER_NAME") or None,
        "dry_run": False,
        "queue_depth_start": len(records),
    })

    pool = ThreadPoolExecutor(max_workers=max(IMAGE_WORKERS, 1), thread_name_prefix="img")
    images = {r["id"]: pool.submit(_hash_image, r.get("top_image_url")) for r in records}
    batch: list[dict] = []

    def save() -> None:
        nonlocal batch
        if not batch or dry_run:
            batch = []
            return
        try:
            summary["saved"] += db.save_enrichment_batch(batch)
        except Exception as e:
            log.error("save_enrichment_batch failed for %d articles (left for the sweeper): %s",
                      len(batch), e)
            summary["save_failed"] += len(batch)
        batch = []

    try:
        for i, rec in enumerate(records):
            if deadline is not None and time.perf_counter() >= deadline:
                summary["stop_reason"] = "time_budget"
                summary["left_for_sweeper"] = len(records) - i
                log.warning("Time budget reached after %d/%d articles", i, len(records))
                break
            try:
                phash = images[rec["id"]].result(timeout=IMAGE_DOWNLOAD_TIMEOUT + 10)
            except Exception:
                phash = None
            try:
                update, entities = enrich_one({**rec, "_image_phash": phash})
            except Exception as e:
                log.error("[%s] enrich_one crashed (left for the sweeper): %s", rec.get("id"), e)
                summary["failed"] += 1
                continue
            summary["processed"] += 1
            batch.append(_item(rec["id"], update, entities))
            if len(batch) >= SAVE_EVERY:
                save()
        save()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        summary["duration_s"] = round(time.perf_counter() - t0, 1)
        if not dry_run:
            db.log_run_finish(run_id, {
                "claimed": summary["articles"], "processed": summary["saved"],
                "failed": summary["failed"] + summary["save_failed"],
                "released": summary["left_for_sweeper"],
                "cluster_joined": 0, "cluster_gray_joined": 0, "cluster_created": 0,
                "cluster_skipped": 0, "duration_s": summary["duration_s"],
                "stop_reason": summary["stop_reason"], "error": None,
            })
        log.info("Hand-off enrichment done: %s", summary)
    return summary
