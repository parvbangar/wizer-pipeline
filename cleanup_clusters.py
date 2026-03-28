"""
cleanup_clusters.py
════════════════════
Weekly maintenance job: removes stale singleton clusters and merges
near-duplicate clusters detected by the pgvector find_duplicate_clusters()
function.

SINGLETON CLEANUP:
  A cluster is a singleton if outlet_count=1 AND article_count=1 AND it
  has not been updated in CLUSTER_SINGLETON_TTL_HOURS (default: 72h).

  These arise from:
    - Genuinely unique local stories that never spread (correct to keep
      for 72h in case more coverage arrives, then delete)
    - Paywalled stubs or crawl failures with partial enrichment
    - Articles where LaBSE produced an off embedding

  Steps:
    1. Find singleton clusters older than TTL
    2. NULL out articles.cluster_id for those articles
    3. DELETE the clusters

DUPLICATE CLUSTER MERGE:
  The pgvector find_duplicate_clusters() function finds pairs of clusters
  whose canonical embeddings are above MERGE_SIMILARITY_THRESHOLD (0.87).
  For each pair, we merge the smaller cluster into the larger:
    - Transfer all articles (UPDATE articles SET cluster_id = winner)
    - Merge entity_set, outlet_set, top_entities
    - Update article_count, outlet_count on the winner
    - Delete the loser cluster

  Threshold 0.87 > 0.82 (article-to-cluster) because cluster-to-cluster
  similarity should be higher — we only merge if they're about the same
  specific event, not just the same topic.

USAGE:
  python cleanup_clusters.py [--dry-run] [--ttl-hours 72] [--merge-threshold 0.87]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cleanup_clusters")


def get_client():
    from supabase import create_client, ClientOptions
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    return create_client(url, key, options=ClientOptions(postgrest_client_timeout=30))


# ── Singleton cleanup ─────────────────────────────────────────────────────────

def cleanup_singletons(client, ttl_hours: int, dry_run: bool) -> int:
    """
    Delete singleton clusters older than ttl_hours.
    Returns number of clusters deleted.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()

    # Find stale singletons
    try:
        resp = (
            client.table("article_clusters")
            .select("id")
            .eq("outlet_count", 1)
            .eq("article_count", 1)
            .lt("last_seen_at", cutoff)
            .execute()
        )
    except Exception as e:
        log.error("Failed to fetch singleton clusters: %s", e)
        return 0

    clusters = resp.data or []
    if not clusters:
        log.info("No stale singleton clusters found (TTL: %dh)", ttl_hours)
        return 0

    cluster_ids = [c["id"] for c in clusters]
    log.info("Found %d stale singleton clusters (TTL: %dh)", len(cluster_ids), ttl_hours)

    if dry_run:
        log.info("DRY RUN — would delete %d singleton clusters", len(cluster_ids))
        return len(cluster_ids)

    # NULL out articles.cluster_id that point to these clusters
    deleted_total = 0
    batch_size = 100
    for i in range(0, len(cluster_ids), batch_size):
        batch = cluster_ids[i:i + batch_size]
        try:
            client.table("articles").update({"cluster_id": None}).in_("cluster_id", batch).execute()
        except Exception as e:
            log.warning("Failed to null cluster_id for batch: %s", e)

        try:
            client.table("article_clusters").delete().in_("id", batch).execute()
            deleted_total += len(batch)
            log.info("Deleted batch of %d singleton clusters", len(batch))
        except Exception as e:
            log.error("Failed to delete cluster batch: %s", e)

    log.info("Singleton cleanup complete — deleted %d clusters", deleted_total)
    return deleted_total


# ── Duplicate cluster merge ────────────────────────────────────────────────────

def merge_duplicate_clusters(client, threshold: float, dry_run: bool) -> int:
    """
    Find and merge near-duplicate clusters using pgvector HNSW.
    Returns number of merges performed.
    """
    log.info("Finding duplicate clusters (threshold=%.2f)...", threshold)
    try:
        resp = client.rpc("find_duplicate_clusters", {
            "similarity_threshold": threshold,
            "max_pairs":            200,
        }).execute()
    except Exception as e:
        log.error("find_duplicate_clusters RPC failed: %s", e)
        return 0

    pairs = resp.data or []
    if not pairs:
        log.info("No duplicate clusters found above threshold %.2f", threshold)
        return 0

    log.info("Found %d candidate duplicate pairs", len(pairs))
    merged = 0

    for pair in pairs:
        cluster_a = pair["cluster_a"]
        cluster_b = pair["cluster_b"]
        sim       = pair.get("similarity", 0.0)
        a_count   = pair.get("a_count", 0)
        b_count   = pair.get("b_count", 0)

        # Winner = larger cluster; loser = smaller cluster
        if a_count >= b_count:
            winner_id, loser_id = cluster_a, cluster_b
        else:
            winner_id, loser_id = cluster_b, cluster_a

        log.info(
            "Merging cluster %s (n=%d) ← %s (n=%d) [sim=%.3f]",
            winner_id, max(a_count, b_count),
            loser_id,  min(a_count, b_count),
            sim,
        )

        if dry_run:
            merged += 1
            continue

        # Fetch both clusters for stat merging
        try:
            winner_resp = client.table("article_clusters").select("*").eq("id", winner_id).execute()
            loser_resp  = client.table("article_clusters").select("*").eq("id", loser_id).execute()
            winner = (winner_resp.data or [{}])[0]
            loser  = (loser_resp.data  or [{}])[0]
        except Exception as e:
            log.warning("Failed to fetch cluster pair for merge: %s", e)
            continue

        # Transfer all loser articles to winner
        try:
            client.table("articles").update({"cluster_id": winner_id}).eq("cluster_id", loser_id).execute()
        except Exception as e:
            log.warning("Failed to transfer articles from %s to %s: %s", loser_id, winner_id, e)
            continue

        # Merge outlet_set, entity_set, top_entities
        merged_outlet_set = list(
            set(winner.get("outlet_set") or []) | set(loser.get("outlet_set") or [])
        )
        merged_entity_set = list(
            set(winner.get("entity_set") or []) | set(loser.get("entity_set") or [])
        )

        winner_update = {
            "article_count": (winner.get("article_count") or 0) + (loser.get("article_count") or 0),
            "outlet_count":  len(merged_outlet_set),
            "outlet_set":    merged_outlet_set,
            "entity_set":    merged_entity_set,
            "last_seen_at":  max(
                winner.get("last_seen_at") or "",
                loser.get("last_seen_at")  or "",
            ) or None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Keep the longer headline
        winner_headline = winner.get("headline") or ""
        loser_headline  = loser.get("headline") or ""
        if len(loser_headline) > len(winner_headline):
            winner_update["headline"] = loser_headline

        try:
            client.table("article_clusters").update(winner_update).eq("id", winner_id).execute()
            client.table("article_clusters").delete().eq("id", loser_id).execute()
            merged += 1
        except Exception as e:
            log.error("Failed to finalise merge of %s ← %s: %s", winner_id, loser_id, e)

    log.info("Merge complete — %d duplicate pairs merged", merged)
    return merged


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cluster maintenance: cleanup + merge")
    parser.add_argument("--dry-run",          action="store_true", help="Preview changes, don't write")
    parser.add_argument("--ttl-hours",        type=int,   default=72,   help="Singleton TTL in hours (default: 72)")
    parser.add_argument("--merge-threshold",  type=float, default=0.87, help="Cluster merge similarity threshold (default: 0.87)")
    parser.add_argument("--skip-cleanup",     action="store_true", help="Skip singleton cleanup")
    parser.add_argument("--skip-merge",       action="store_true", help="Skip duplicate cluster merge")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN — no changes will be written")

    client = get_client()

    deleted = 0
    if not args.skip_cleanup:
        deleted = cleanup_singletons(client, args.ttl_hours, args.dry_run)

    merged = 0
    if not args.skip_merge:
        merged = merge_duplicate_clusters(client, args.merge_threshold, args.dry_run)

    log.info("=== Maintenance complete | deleted_singletons=%d | merged_duplicates=%d ===",
             deleted, merged)
    sys.exit(0)


if __name__ == "__main__":
    main()
