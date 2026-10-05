"""
tools/db_migrations.py
══════════════════════
The canonical order of the SQL migrations in docs/, and a helper that applies
them to a plain PostgreSQL database (used by the SQL integration tests and the
clustering simulator).

On Supabase, run the same files in the same order in the SQL Editor — see
docs/MIGRATIONS.md. Every file is idempotent, so re-running the whole chain
on an existing database is safe.
"""

from __future__ import annotations

from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Order matters: each file assumes the ones above it have run.
MIGRATION_ORDER: list[str] = [
    "migration.sql",                     # Layer 1: feeds, articles, pipeline_runs
    "ingestion_fixes_migration.sql",     # Layer 1 fixes: last_new_article_at, disabled_reason, …
    "enrichment_migration.sql",          # Layer 2: enrichment columns, entities, clusters (v1 table)
    "enrichment_outputs_migration.sql",  # ai_summary / ai_tag / ai_region / ai_org / sentiment_stats
    "enrichment_queue_migration.sql",    # claim queue, enrichment_runs, queue health
    "embedding_migration.sql",           # v1: canonical_embedding jsonb        (history)
    "pgvector_migration.sql",            # v1: embedding_vec + find_nearest_cluster (history)
    "tier2_clustering_migration.sql",    # v1: outlet_set + find_duplicate_clusters (history)
    "cluster_index_migration.sql",       # v1: last_seen_at index               (history)
    "propensity_migration.sql",          # propensity_score column
    "clustering_v2_migration.sql",       # story clustering v2 (current)
    "archive_migration.sql",             # Parquet archive log + verified-only pruning
    "enrichment_queue_v2_migration.sql", # queue v2: oldest-first, no age gate, crawl failures, retries
]

# Roles that Supabase provides and some migrations GRANT to / create policies
# for. Plain Postgres doesn't have them.
SUPABASE_ROLES = ("anon", "authenticated", "service_role")


def migration_paths() -> list[Path]:
    return [DOCS / name for name in MIGRATION_ORDER]


def apply_all(conn) -> None:
    """
    Apply every migration, in order, on a psycopg (v3) connection.
    Creates the Supabase roles first if they don't exist.
    """
    with conn.cursor() as cur:
        for role in SUPABASE_ROLES:
            cur.execute(
                "DO $$ BEGIN "
                f"IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') "
                f"THEN CREATE ROLE {role}; END IF; END $$;"
            )
    conn.commit()
    for path in migration_paths():
        sql = path.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            try:
                cur.execute(sql)
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"{path.name}: {e}") from e
        conn.commit()
