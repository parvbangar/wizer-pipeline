"""
enrichment/memory_clustering.py
═══════════════════════════════
Story clustering in memory — the same decision as wizer_assign_cluster
(docs/clustering_v2_migration.sql §4), computed on the runner instead of
inside Postgres.

WHY:
  On the Supabase Micro instance the per-article SQL path (an exact cosine
  scan over every live cluster, then a TOASTed row update) exhausted the CPU
  and disk-I/O credits. Here the database only receives the RESULT, in bulk
  (wizer_apply_cluster_changes, docs/bulk_io_migration.sql).

SAFETY — ONE WRITER:
  The SQL function serialises on an advisory lock. This module has no lock;
  instead exactly one job at a time owns cluster state: the "story-clustering"
  concurrency group (.github/workflows/process.yml), which also runs
  maintenance. Between runs the state lives in a file (GitHub Actions cache);
  at the start of a run it is brought up to date with the clusters the
  database changed since the last sync (maintenance merges).

PARITY WITH SQL (tests/test_memory_clustering.py checks decision-for-decision):
  - candidates: active clusters of the same embedding model whose time span is
    compatible with the article's time t (gap / span rules), the
    CLUSTER_CANDIDATES nearest by HALF-precision centroid cosine
    (centroid/anchor/representative are halfvec in Postgres);
  - score: exact average-link  q · Σm / n  from the float32 sum;
  - in score order: stop below the threshold, skip if anchor similarity is
    below CLUSTER_ANCHOR_THRESHOLD, otherwise join; no candidate → seed;
  - on join: sum += q, centroid = normalise(sum), representative / headline
    move to the article if it is closer to the new centroid.
  The gray zone (entity / image corroboration) is NOT implemented: at ingest
  time an article has no entities or image hash yet, and the shipped config
  disables it (CLUSTER_GRAY_THRESHOLD == CLUSTER_JOIN_THRESHOLD).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from enrichment.config import (
    CLUSTER_ANCHOR_THRESHOLD,
    CLUSTER_CANDIDATES,
    CLUSTER_EMBEDDING_MODEL,
    CLUSTER_GRAY_THRESHOLD,
    CLUSTER_JOIN_THRESHOLD,
    CLUSTER_MAX_GAP_HOURS,
    CLUSTER_MAX_SPAN_HOURS,
)

log = logging.getLogger(__name__)

DIM = 768
_OUTLET_CAP = None        # outlet_set is uncapped in SQL
_LANGUAGE_CAP = 50        # wizer_jsonb_text_union(..., 50)
_HEADLINE_CHARS = 500


@dataclass(frozen=True)
class Assignment:
    article_id: int
    cluster_id: str
    action: str                 # seed | join
    similarity: float
    article_count: int
    outlet_count: int


@dataclass
class _Params:
    join: float = CLUSTER_JOIN_THRESHOLD
    anchor: float = CLUSTER_ANCHOR_THRESHOLD
    gap_s: float = CLUSTER_MAX_GAP_HOURS * 3600.0
    span_s: float = CLUSTER_MAX_SPAN_HOURS * 3600.0
    candidates: int = CLUSTER_CANDIDATES


def _norm_domain(domain: str | None) -> str:
    d = (domain or "").strip().lower()
    return d or "(unknown)"


def _norm_lang(lang: str | None) -> str | None:
    return (lang or "").strip().lower() or None


def _cos16(m16: np.ndarray, q16: np.ndarray) -> np.ndarray:
    """Cosine between half-precision vectors, as pgvector computes it (float32 maths)."""
    m = m16.astype(np.float32)
    q = q16.astype(np.float32)
    denom = np.linalg.norm(m, axis=-1) * np.linalg.norm(q)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (m @ q) / denom, 0.0)


def _epoch(ts) -> float:
    if ts is None:
        return datetime.now(timezone.utc).timestamp()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@dataclass
class ClusterState:
    """Live clusters of one embedding model, plus what changed since the last flush."""

    model: str = CLUSTER_EMBEDDING_MODEL
    synced_at: str | None = None          # DB clock at the last sync (ISO)
    ids: list[str] = field(default_factory=list)
    sums: np.ndarray = field(default_factory=lambda: np.zeros((0, DIM), np.float32))
    cent16: np.ndarray = field(default_factory=lambda: np.zeros((0, DIM), np.float16))
    anchor16: np.ndarray = field(default_factory=lambda: np.zeros((0, DIM), np.float16))
    rep16: np.ndarray = field(default_factory=lambda: np.zeros((0, DIM), np.float16))
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    first: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    last: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    rep_ids: list[int | None] = field(default_factory=list)
    seed_ids: list[int | None] = field(default_factory=list)   # canonical_article_id
    headlines: list[str | None] = field(default_factory=list)
    outlets: list[list[str]] = field(default_factory=list)
    languages: list[list[str]] = field(default_factory=list)
    # Bookkeeping for the next flush (not persisted).
    dirty: set[int] = field(default_factory=set)
    created: set[int] = field(default_factory=set)
    rep_moved: set[int] = field(default_factory=set)
    assignments: list[Assignment] = field(default_factory=list)
    _n: int = 0
    _index: dict[str, int] = field(default_factory=dict)

    # ── size management ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self._n

    def _grow(self, need: int) -> None:
        cap = self.sums.shape[0]
        if need <= cap:
            return
        new_cap = max(need, cap * 2, 256)
        for name, dtype in (("sums", np.float32), ("cent16", np.float16),
                            ("anchor16", np.float16), ("rep16", np.float16)):
            arr = np.zeros((new_cap, DIM), dtype)
            arr[:cap] = getattr(self, name)
            setattr(self, name, arr)
        for name, dtype in (("counts", np.int32), ("first", np.float64), ("last", np.float64)):
            arr = np.zeros(new_cap, dtype)
            arr[:cap] = getattr(self, name)
            setattr(self, name, arr)

    def _append(self, cid: str) -> int:
        i = self._n
        self._grow(i + 1)
        self._n += 1
        self.ids.append(cid)
        self.rep_ids.append(None)
        self.seed_ids.append(None)
        self.headlines.append(None)
        self.outlets.append([])
        self.languages.append([])
        self._index[cid] = i
        return i

    # ── loading (DB rows / file) ─────────────────────────────────────────────
    def upsert_row(self, row: dict) -> None:
        """
        Apply one cluster row from the database (wizer_cluster_state).
        Non-active or other-model clusters are removed from the live set.
        """
        cid = str(row["id"])
        alive = row.get("status", "active") == "active" and row.get("embedding_model", self.model) == self.model
        if not alive:
            self.remove(cid)
            return
        i = self._index.get(cid)
        if i is None:
            i = self._append(cid)
        s = np.asarray(_parse_vec(row["centroid_sum"]), np.float32)
        self.sums[i] = s
        self.cent16[i] = (s / np.linalg.norm(s)).astype(np.float16)
        self.anchor16[i] = np.asarray(_parse_vec(row["anchor"]), np.float16)
        rep = row.get("representative")
        self.rep16[i] = np.asarray(_parse_vec(rep), np.float16) if rep else 0
        self.counts[i] = int(row["article_count"])
        self.first[i] = _epoch(row["first_seen_at"])
        self.last[i] = _epoch(row["last_seen_at"])
        self.rep_ids[i] = row.get("representative_article_id")
        self.headlines[i] = row.get("headline")
        self.outlets[i] = list(row.get("outlet_set") or [])
        self.languages[i] = list(row.get("language_set") or [])

    def remove(self, cid: str) -> None:
        i = self._index.pop(cid, None)
        if i is None:
            return
        last = self._n - 1
        if i != last:                                   # move the last row into the hole
            moved = self.ids[last]
            for arr in (self.sums, self.cent16, self.anchor16, self.rep16,
                        self.counts, self.first, self.last):
                arr[i] = arr[last]
            for lst in (self.ids, self.rep_ids, self.seed_ids, self.headlines, self.outlets, self.languages):
                lst[i] = lst[last]
            self._index[moved] = i
        for lst in (self.ids, self.rep_ids, self.seed_ids, self.headlines, self.outlets, self.languages):
            lst.pop()
        self._n = last
        for s in (self.dirty, self.created, self.rep_moved):
            s.discard(i)
            if last in s and i != last:
                s.discard(last)
                s.add(i)

    def prune(self, keep_after_epoch: float) -> int:
        """Drop clusters last seen before `keep_after_epoch` (they can no longer be joined by fresh articles)."""
        stale = [self.ids[i] for i in range(self._n) if self.last[i] < keep_after_epoch and i not in self.dirty]
        for cid in stale:
            self.remove(cid)
        return len(stale)

    # ── the decision ─────────────────────────────────────────────────────────
    def assign(self, article_id: int, embedding, published_at, domain: str | None,
               title: str | None, language: str | None, params: _Params | None = None) -> Assignment:
        p = params or _Params()
        q = np.asarray(embedding, np.float32)
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n == 0:
            raise ValueError(f"zero or NaN embedding (article {article_id})")
        if q.shape != (DIM,):
            raise ValueError(f"embedding has {q.shape} dimensions, expected {DIM} (article {article_id})")
        q = q / n
        q16 = q.astype(np.float16)
        t = _epoch(published_at)
        dom = _norm_domain(domain)
        lang = _norm_lang(language)

        pick, pick_sim = None, None
        k = self._n
        if k:
            first, last = self.first[:k], self.last[:k]
            ok = ((last >= t - p.gap_s) & (first <= t + p.gap_s)
                  & ((np.maximum(last, t) - np.minimum(first, t)) <= p.span_s))
            win = np.flatnonzero(ok)
            if win.size:
                csim = _cos16(self.cent16[win], q16)
                top = win[np.argsort(-csim, kind="stable")[:max(p.candidates, 1)]]
                sims = (self.sums[top] @ q) / self.counts[top]
                for j in np.argsort(-sims, kind="stable"):
                    i, sim = int(top[j]), float(sims[j])
                    if sim < p.join:                    # sorted: nothing better follows
                        break
                    if float(_cos16(self.anchor16[i], q16)) < p.anchor:
                        continue
                    pick, pick_sim = i, sim
                    break

        if pick is None:
            i = self._append(str(uuid.uuid4()))
            self.sums[i] = q
            self.cent16[i] = self.anchor16[i] = self.rep16[i] = q16
            self.counts[i] = 1
            self.first[i] = self.last[i] = t
            self.rep_ids[i] = self.seed_ids[i] = article_id
            self.headlines[i] = (title or "")[:_HEADLINE_CHARS] or None
            self.outlets[i] = [dom]
            self.languages[i] = [lang] if lang else []
            self.created.add(i)
            self.dirty.add(i)
            a = Assignment(article_id, self.ids[i], "seed", 1.0, 1, 1)
        else:
            i = pick
            new_sum = self.sums[i] + q
            new_cent16 = (new_sum / np.linalg.norm(new_sum)).astype(np.float16)
            rep_new = float(_cos16(new_cent16, q16))
            rep_old = float(_cos16(new_cent16, self.rep16[i])) if self.rep_ids[i] is not None else -2.0
            self.sums[i] = new_sum
            self.cent16[i] = new_cent16
            self.counts[i] += 1
            if dom not in self.outlets[i]:
                self.outlets[i].append(dom)
            if lang and lang not in self.languages[i] and len(self.languages[i]) < _LANGUAGE_CAP:
                self.languages[i].append(lang)
            self.first[i] = min(self.first[i], t)
            self.last[i] = max(self.last[i], t)
            if rep_new > rep_old:
                self.rep16[i] = q16
                self.rep_ids[i] = article_id
                if title:
                    self.headlines[i] = title[:_HEADLINE_CHARS]
                self.rep_moved.add(i)
            self.dirty.add(i)
            a = Assignment(article_id, self.ids[i], "join", pick_sim,
                           int(self.counts[i]), len(self.outlets[i]))
        self.assignments.append(a)
        return a

    # ── twin detection (wizer_find_cluster_merge_candidates, in memory) ──────
    def merge_candidates(self, probe_ids, threshold: float, params: _Params | None = None) -> list[dict]:
        """
        For each probe cluster, its nearest compatible twin — the same rule as
        SQL wizer_find_cluster_merge_candidates (docs/clustering_v2_migration.sql):
        time-compatible (gap / span), anchors agree (≥ anchor threshold), nearest
        by half-precision centroid cosine, scored by exact average-link
        (Σa · Σb) / (na · nb). Every probe is returned; other_id is None unless
        that score is ≥ threshold. Rows are shaped for
        cluster_maintenance.plan_merges.
        """
        p = params or _Params()
        k = self._n
        rows = []
        if not k:
            return rows
        first, last = self.first[:k], self.last[:k]
        for pid in probe_ids:
            i = self._index.get(str(pid))
            if i is None:
                continue
            ok = ((last >= first[i] - p.gap_s) & (first <= last[i] + p.gap_s)
                  & ((np.maximum(last, last[i]) - np.minimum(first, first[i])) <= p.span_s))
            ok[i] = False
            cand = np.flatnonzero(ok)
            other = sim = None
            if cand.size:
                anchors_agree = _cos16(self.anchor16[cand], self.anchor16[i]) >= p.anchor
                cand = cand[anchors_agree]
            if cand.size:
                j = int(cand[np.argmax(_cos16(self.cent16[cand], self.cent16[i]))])
                s = float(self.sums[i] @ self.sums[j]) / (float(self.counts[i]) * float(self.counts[j]))
                if s >= threshold:
                    other, sim = j, s
            rows.append({"probe_id": self.ids[i], "probe_count": int(self.counts[i]),
                         "other_id": self.ids[other] if other is not None else None,
                         "other_count": int(self.counts[other]) if other is not None else None,
                         "similarity": sim})
        return rows

    # ── flushing ─────────────────────────────────────────────────────────────
    def pending_changes(self) -> tuple[list[dict], list[dict]]:
        """
        (cluster rows, article assignments) to send to wizer_apply_cluster_changes.
        Vectors travel as text; the centroid is derived from the sum in SQL, and
        anchor / representative are only sent when they differ from it.
        """
        clusters = []
        for i in sorted(self.dirty):
            row = {
                "id": self.ids[i],
                "is_new": i in self.created,
                "centroid_sum": _fmt_vec(self.sums[i]),
                "article_count": int(self.counts[i]),
                "outlet_set": self.outlets[i],
                "language_set": self.languages[i],
                "first_seen_at": _iso(self.first[i]),
                "last_seen_at": _iso(self.last[i]),
                "headline": self.headlines[i],
                "representative_article_id": self.rep_ids[i],
                "canonical_article_id": self.seed_ids[i],
            }
            # A singleton's anchor and representative ARE its sum (one unit
            # vector), so SQL derives them; otherwise send them.
            if i in self.created and self.counts[i] > 1:
                row["anchor"] = _fmt_vec(self.anchor16[i].astype(np.float32))
            if (i in self.rep_moved or i in self.created) and self.counts[i] > 1:
                row["representative"] = _fmt_vec(self.rep16[i].astype(np.float32))
            clusters.append(row)
        articles = [{"id": a.article_id, "cluster_id": a.cluster_id, "action": a.action,
                     "similarity": round(a.similarity, 6)} for a in self.assignments]
        return clusters, articles

    def mark_flushed(self, synced_at: str | None) -> None:
        self.dirty.clear()
        self.created.clear()
        self.rep_moved.clear()
        self.assignments.clear()
        if synced_at:
            self.synced_at = synced_at

    # ── persistence (GitHub Actions cache) ───────────────────────────────────
    def save(self, path: str | Path) -> None:
        if self.dirty:
            raise RuntimeError("refusing to save cluster state with unflushed changes")
        k = self._n
        meta = {"model": self.model, "synced_at": self.synced_at, "ids": self.ids,
                "rep_ids": self.rep_ids, "seed_ids": self.seed_ids, "headlines": self.headlines,
                "outlets": self.outlets, "languages": self.languages}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, sums=self.sums[:k], cent16=self.cent16[:k], anchor16=self.anchor16[:k],
                 rep16=self.rep16[:k], counts=self.counts[:k], first=self.first[:k],
                 last=self.last[:k], meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, model: str = CLUSTER_EMBEDDING_MODEL) -> "ClusterState | None":
        """The saved state, or None if missing / for another model / unreadable."""
        try:
            z = np.load(path)
            meta = json.loads(z["meta"].tobytes().decode())
        except (OSError, ValueError, KeyError) as e:
            log.info("No usable cluster state at %s (%s) — will load from the database", path, e)
            return None
        if meta.get("model") != model:
            log.info("Cluster state is for %s, not %s — reloading", meta.get("model"), model)
            return None
        st = cls(model=model, synced_at=meta.get("synced_at"))
        k = len(meta["ids"])
        st._grow(k)
        for name in ("sums", "cent16", "anchor16", "rep16", "counts", "first", "last"):
            getattr(st, name)[:k] = z[name]
        st.ids, st.rep_ids, st.headlines = meta["ids"], meta["rep_ids"], meta["headlines"]
        st.seed_ids = meta.get("seed_ids") or [None] * k
        st.outlets, st.languages = meta["outlets"], meta["languages"]
        st._n = k
        st._index = {cid: i for i, cid in enumerate(st.ids)}
        return st


def _fmt_vec(v: np.ndarray) -> str:
    """pgvector text literal; 9 significant digits round-trips float32 exactly."""
    return "[" + ",".join(f"{x:.9g}" for x in np.asarray(v, np.float32).tolist()) + "]"


def _parse_vec(text) -> list[float]:
    if isinstance(text, (list, tuple, np.ndarray)):
        return list(text)
    return [float(x) for x in str(text).strip("[]").split(",")]


def check_supported_config() -> None:
    """The in-memory path has no gray zone; refuse a config that relies on it."""
    if CLUSTER_GRAY_THRESHOLD < CLUSTER_JOIN_THRESHOLD:
        raise RuntimeError(
            "CLUSTER_GRAY_THRESHOLD < CLUSTER_JOIN_THRESHOLD enables the entity/image gray zone, "
            "which in-memory clustering does not implement (no entities exist at ingest time)."
        )
