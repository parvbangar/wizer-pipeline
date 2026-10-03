"""
tools/pg_backend.py
═══════════════════
A direct-psycopg implementation of the clustering RPC surface that
enrichment.db exposes over Supabase/PostgREST.

Used by the SQL integration tests and the clustering simulator, so they drive
the exact same SQL functions — and the exact same Python orchestration
(enrichment.cluster_maintenance) — as production, against a plain local
Postgres instead of a Supabase project.
"""

from __future__ import annotations

import threading

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# Parameters that need an explicit cast when passed as text/JSON.
_CASTS = {
    "p_embedding": "::vector",
    "p_published_at": "::timestamptz",
    "p_entities": "::jsonb",
    "p_since": "::timestamptz",
    "p_after_ts": "::timestamptz",
    "p_after_id": "::uuid",
}


def _call(conn, fn: str, params: dict, cols: str = "*") -> list[dict]:
    args = dict(params)
    if "p_entities" in args and not isinstance(args["p_entities"], Jsonb):
        args["p_entities"] = Jsonb(args["p_entities"])
    sql = f"SELECT {cols} FROM {fn}(" + ", ".join(
        f"{k} => %({k})s{_CASTS.get(k, '')}" for k in args
    ) + ")"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    conn.commit()
    return rows


class PgBackend:
    """Same method names and return shapes as enrichment.db."""

    def __init__(self, conn):
        self.conn = conn

    def assign_cluster(self, params: dict) -> dict | None:
        rows = _call(self.conn, "wizer_assign_cluster", params)
        return rows[0] if rows else None

    def find_merge_candidates(self, params: dict) -> list[dict]:
        return _call(self.conn, "wizer_find_cluster_merge_candidates", params)

    def merge_clusters(self, winner_id: str, loser_id: str) -> bool:
        rows = _call(self.conn, "wizer_merge_clusters",
                     {"p_winner": winner_id, "p_loser": loser_id}, cols="wizer_merge_clusters AS ok")
        return bool(rows[0]["ok"])

    def reconcile_cluster_counts(self, since_iso: str) -> int:
        rows = _call(self.conn, "wizer_reconcile_cluster_counts", {"p_since": since_iso},
                     cols="wizer_reconcile_cluster_counts AS n")
        return int(rows[0]["n"])

    def prune_orphan_clusters(self, older_than_hours: int) -> int:
        rows = _call(self.conn, "wizer_prune_orphan_clusters", {"p_older_than_hours": older_than_hours},
                     cols="wizer_prune_orphan_clusters AS n")
        return int(rows[0]["n"])


# ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement for the enrichment.db functions the runner uses, so the
# REAL runner (enrichment.runner.run_enrichment) can run end-to-end against a
# local Postgres. Used by tools/e2e_local.py.
# ─────────────────────────────────────────────────────────────────────────────

_JSON_COLUMNS = {"keywords", "sentiment_stats", "ai_tag", "ai_region", "ai_org"}


def install_into_enrichment_db(conn) -> None:
    """Monkey-patch enrichment.db so every call hits `conn` directly."""
    from enrichment import config as cfg
    from enrichment import db

    backend = PgBackend(conn)
    lock = threading.Lock()     # one psycopg connection, possibly several runner threads

    def locked(fn):
        def wrapper(*a, **k):
            with lock:
                return fn(*a, **k)
        return wrapper

    def claim_batch(limit, max_age_hours=cfg.ENRICH_MAX_AGE_HOURS,
                    lease_minutes=cfg.ENRICH_CLAIM_LEASE_MINUTES, max_attempts=cfg.ENRICH_MAX_ATTEMPTS):
        with conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute("SELECT * FROM wizer_claim_enrichment_batch(%s, %s, %s, %s)",
                               (limit, max_age_hours, lease_minutes, max_attempts)).fetchall()
        conn.commit()
        for r in rows:
            if r.get("published_at") is not None:
                r["published_at"] = r["published_at"].isoformat()
        return sorted(rows, key=lambda r: r.get("published_at") or "", reverse=True)

    def release_claims(ids):
        n = conn.execute("SELECT wizer_release_enrichment_claims(%s)", (list(ids),)).fetchone()[0]
        conn.commit()
        return n

    def queue_depth(max_age_hours=cfg.ENRICH_MAX_AGE_HOURS, max_attempts=cfg.ENRICH_MAX_ATTEMPTS):
        n = conn.execute("SELECT wizer_enrichment_queue_depth(%s, %s)",
                         (max_age_hours, max_attempts)).fetchone()[0]
        conn.commit()
        return n

    def save_entities(article_id, entities):
        conn.execute("DELETE FROM article_entities WHERE article_id = %s", (article_id,))
        for e in entities:
            conn.execute("INSERT INTO article_entities (article_id, entity_text, entity_type, salience) "
                         "VALUES (%s, %s, %s, %s)",
                         (article_id, e["entity_text"], e["entity_type"], e["salience"]))
        conn.commit()
        return True

    def save_article_enrichment(article_id, update):
        cols = {k: (Jsonb(v) if k in _JSON_COLUMNS and v is not None else v) for k, v in update.items()}
        sets = "".join(f"{k} = %({k})s, " for k in cols)
        conn.execute(f"UPDATE articles SET {sets}enriched_at = now() WHERE id = %(_id)s",
                     {**cols, "_id": article_id})
        conn.commit()
        return True

    def mark_enrichment_failed(article_id, error):
        return save_article_enrichment(article_id, {"enrich_error": error})

    def log_run_start(row):
        rid = conn.execute(
            "INSERT INTO enrichment_runs (trigger, runner, dry_run, queue_depth_start) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (row.get("trigger"), row.get("runner"), row.get("dry_run", False), row.get("queue_depth_start")),
        ).fetchone()[0]
        conn.commit()
        return rid

    def log_run_finish(run_id, row):
        if not run_id:
            return
        cols = {k: v for k, v in row.items() if v is not None}
        sets = "".join(f"{k} = %({k})s, " for k in cols)
        conn.execute(f"UPDATE enrichment_runs SET {sets}finished_at = now() WHERE id = %(_id)s",
                     {**cols, "_id": run_id})
        conn.commit()

    replacements = {
        "claim_batch": claim_batch, "release_claims": release_claims, "queue_depth": queue_depth,
        "save_entities": save_entities, "save_article_enrichment": save_article_enrichment,
        "mark_enrichment_failed": mark_enrichment_failed, "log_run_start": log_run_start,
        "log_run_finish": log_run_finish, "assign_cluster": backend.assign_cluster,
        "find_merge_candidates": backend.find_merge_candidates,
        "merge_clusters": backend.merge_clusters,
        "reconcile_cluster_counts": backend.reconcile_cluster_counts,
        "prune_orphan_clusters": backend.prune_orphan_clusters,
    }
    for name, fn in replacements.items():
        setattr(db, name, locked(fn))


# ─────────────────────────────────────────────────────────────────────────────
# archiver.db over psycopg — same functions and return shapes, used by the SQL
# integration tests to run archiver.runner.run_archive against real Postgres.
# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(row: dict) -> dict:
    """Make a psycopg row look like PostgREST JSON (ISO timestamps, str uuids)."""
    import datetime as _dt
    import decimal
    import uuid as _uuid
    out = {}
    for k, v in row.items():
        if isinstance(v, (_dt.datetime, _dt.date)):
            v = v.isoformat()
        elif isinstance(v, _uuid.UUID):
            v = str(v)
        elif isinstance(v, decimal.Decimal):
            v = float(v)
        out[k] = v
    return out


class PgArchiveDB:
    def __init__(self, conn):
        self.conn = conn

    def _rows(self, sql, params=()):
        with self.conn.cursor(row_factory=dict_row) as cur:
            rows = [_jsonable(r) for r in cur.execute(sql, params).fetchall()]
        self.conn.commit()
        return rows

    def pending_days(self, hot_days):
        import datetime as _dt
        return [(_dt.date.fromisoformat(r["day"]), r["status"]) for r in
                self._rows("SELECT * FROM wizer_archive_days(%s)", (hot_days,))]

    def day_stats(self, day):
        r = self._rows("SELECT * FROM wizer_archive_day_stats(%s)", (day,))[0]
        return {"article_rows": int(r["article_rows"]), "min_id": r["min_id"], "max_id": r["max_id"]}

    def article_page(self, day, after_id, limit=1000):
        return self._rows("SELECT * FROM wizer_archive_page(%s, %s, %s)", (day, after_id, limit))

    def entity_page(self, day, after_key, limit=1000):
        return self._rows("SELECT * FROM wizer_archive_entities_page(%s, %s, %s)", (day, after_key, limit))

    def clusters(self, day):
        return self._rows("SELECT * FROM wizer_archive_clusters(%s)", (day,))

    def get_log(self, day):
        rows = self._rows("SELECT * FROM article_archive_log WHERE day = %s", (day,))
        return rows[0] if rows else None

    def upsert_log(self, row):
        cols = list(row)
        vals = [Jsonb(v) if k == "objects" else v for k, v in row.items()]
        placeholders = ", ".join(["%s"] * len(cols))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "day")
        self.conn.execute(
            f"INSERT INTO article_archive_log ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT (day) DO UPDATE SET {updates}", vals)
        self.conn.commit()

    def mark_verified(self, day):
        self.conn.execute("UPDATE article_archive_log SET status = 'verified', verified_at = now() "
                          "WHERE day = %s AND status = 'uploaded'", (day,))
        self.conn.commit()

    def prune_day(self, day, min_age_days, max_delete):
        n = self.conn.execute("SELECT wizer_prune_archived_day(%s, %s, %s)",
                              (day, min_age_days, max_delete)).fetchone()[0]
        self.conn.commit()
        return n
