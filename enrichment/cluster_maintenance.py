"""
enrichment/cluster_maintenance.py
═════════════════════════════════
Periodic upkeep of story clusters (run by `python cluster.py maintain`, on a
schedule from .github/workflows/cluster_maintenance.yml).

WHY CLUSTERS NEED UPKEEP:
  Assignment is online and greedy — each article is placed once, in arrival
  order. If the first two outlets' versions of a story arrive with unusually
  different headlines, they seed TWO clusters, and later articles split
  between them; neither ever reaches the outlet_count the story deserves.
  Merging repairs exactly that, and only that: two clusters are merged when
  their articles are, on average, as similar to each other as articles inside
  one story are (average-link similarity ≥ CLUSTER_MERGE_THRESHOLD), their
  seeds agree (anchor guard) and their time spans are compatible.

WHAT ONE RUN DOES:
  1. merge     find twin clusters among those touched in the lookback window,
               fold each smaller one into its larger twin (repeat until stable)
  2. reconcile recount article_count / outlet_count from member articles for
               recently touched clusters — non-zero means something outside the
               assignment path changed membership (e.g. Layer 1 table pruning)
  3. prune     delete clusters nobody points at any more (pruned articles,
               aged-out merge tombstones, v1 legacy rows)

BACKEND:
  Every DB call goes through a `backend` object with five methods
  (find_merge_candidates, merge_clusters, reconcile_cluster_counts,
  prune_orphan_clusters) — enrichment.db in production, a psycopg adapter in
  the simulator and the SQL integration tests. The merge PLANNING is a pure
  function so it is unit-tested without a database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from enrichment.config import (
    CLUSTER_ANCHOR_THRESHOLD,
    CLUSTER_EMBEDDING_MODEL,
    CLUSTER_MAX_GAP_HOURS,
    CLUSTER_MAX_SPAN_HOURS,
    CLUSTER_MERGE_THRESHOLD,
)

log = logging.getLogger(__name__)

_PROBE_PAGE = 200      # probes per RPC call — keeps each statement short
_MAX_ROUNDS = 5        # merge → re-probe rounds; real data converges in 1-2


@dataclass
class MaintenanceReport:
    merges: int = 0
    merge_rounds: int = 0
    candidates_seen: int = 0
    reconciled: int = 0
    pruned: int = 0
    dry_run: bool = False
    planned: list[tuple[str, str, float]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# PURE: decide which clusters to merge
# ─────────────────────────────────────────────────────────────────────────────

def plan_merges(candidates: list[dict], threshold: float) -> list[tuple[str, str, float]]:
    """
    Turn candidate pairs into a conflict-free list of (winner, loser, sim).

      - pairs below `threshold` or without a partner are ignored
      - (a, b) and (b, a) are the same pair
      - strongest pairs first
      - the LARGER cluster wins (it is the better-established story; ties go
        to the lexicographically smaller id so the plan is deterministic)
      - every cluster takes part in at most one merge per round: after a
        merge the winner's centroid has moved, so its other candidate pairs
        must be re-measured in the next round rather than trusted stale
    """
    best: dict[frozenset, tuple[float, str, int, str, int]] = {}
    for c in candidates:
        other, sim = c.get("other_id"), c.get("similarity")
        if other is None or sim is None or sim < threshold:
            continue
        a, na = str(c["probe_id"]), int(c.get("probe_count") or 0)
        b, nb = str(other), int(c.get("other_count") or 0)
        if a == b:
            continue
        key = frozenset((a, b))
        if key not in best or sim > best[key][0]:
            best[key] = (float(sim), a, na, b, nb)

    plan: list[tuple[str, str, float]] = []
    used: set[str] = set()
    for sim, a, na, b, nb in sorted(best.values(), key=lambda t: (-t[0], min(t[1], t[3]))):
        if a in used or b in used:
            continue
        if na > nb or (na == nb and a < b):
            winner, loser = a, b
        else:
            winner, loser = b, a
        plan.append((winner, loser, sim))
        used.update((a, b))
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def _collect_candidates(backend, since: datetime, threshold: float, model: str) -> list[dict]:
    rows: list[dict] = []
    after_ts, after_id = "-infinity", "00000000-0000-0000-0000-000000000000"
    while True:
        page = backend.find_merge_candidates({
            "p_model":            model,
            "p_since":            since.isoformat(),
            "p_threshold":        threshold,
            "p_anchor_threshold": CLUSTER_ANCHOR_THRESHOLD,
            "p_max_gap_hours":    CLUSTER_MAX_GAP_HOURS,
            "p_max_span_hours":   CLUSTER_MAX_SPAN_HOURS,
            "p_probe_limit":      _PROBE_PAGE,
            "p_after_ts":         after_ts,
            "p_after_id":         after_id,
        })
        rows.extend(page)
        if len(page) < _PROBE_PAGE:
            return rows
        last = page[-1]
        after_ts, after_id = str(last["probe_updated_at"]), str(last["probe_id"])


def merge_duplicates(
    backend,
    since: datetime,
    threshold: float = CLUSTER_MERGE_THRESHOLD,
    model: str = CLUSTER_EMBEDDING_MODEL,
    dry_run: bool = False,
    report: MaintenanceReport | None = None,
) -> MaintenanceReport:
    """Merge twin clusters touched since `since`, repeating until stable."""
    report = report or MaintenanceReport(dry_run=dry_run)
    for round_no in range(1, _MAX_ROUNDS + 1):
        candidates = _collect_candidates(backend, since, threshold, model)
        report.candidates_seen += len(candidates)
        plan = plan_merges(candidates, threshold)
        report.merge_rounds = round_no
        if not plan:
            break
        if dry_run:
            report.planned.extend(plan)
            log.info("DRY RUN — round %d would merge %d cluster pairs", round_no, len(plan))
            break
        done = 0
        for winner, loser, sim in plan:
            try:
                if backend.merge_clusters(winner, loser):
                    done += 1
                    log.debug("merged %s → %s (avg-link %.3f)", loser, winner, sim)
            except Exception as e:
                log.warning("merge %s → %s failed: %s", loser, winner, e)
        report.merges += done
        log.info("Merge round %d: %d/%d merges applied", round_no, done, len(plan))
        if done == 0:
            break
    return report


def run_maintenance(
    backend,
    lookback_hours: int = 6,
    prune_after_hours: int = 168,
    dry_run: bool = False,
    merge: bool = True,
) -> MaintenanceReport:
    """
    Maintenance pass: merge → reconcile → prune. merge=False skips the SQL twin
    sweep — the in-memory clustering job finds twins itself
    (cluster_job.merge_twins), without an ANN index in Postgres.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    report = merge_duplicates(backend, since, dry_run=dry_run) if merge else MaintenanceReport(dry_run=dry_run)
    if not dry_run:
        report.reconciled = backend.reconcile_cluster_counts(since.isoformat())
        if report.reconciled:
            log.warning("Reconciled %d clusters whose counts had drifted from their members",
                        report.reconciled)
        report.pruned = backend.prune_orphan_clusters(prune_after_hours)
    log.info("Maintenance done: merges=%d (rounds=%d) reconciled=%d pruned=%d%s",
             report.merges, report.merge_rounds, report.reconciled, report.pruned,
             " [dry run]" if dry_run else "")
    return report
