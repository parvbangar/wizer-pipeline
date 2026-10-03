"""
tests/test_production_schema.py
═══════════════════════════════
Contract tests against a snapshot of PRODUCTION's schema
(tests/fixtures/production_schema.sql), not the repo's migrations.

Why: production drifted from docs/migration.sql (bigint feed ids, char(32)
url_hash, char(5) language, extra views). On 2026-10-03 the first production
run failed every insert with 22001 "value too long" because the code wrote
"Malayalam" into articles.language, a char(5) column there. These tests run the
code's real output against the real column types.

Requires WIZER_TEST_DSN (Postgres + pgvector ≥ 0.7), like test_clustering_sql.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

DSN = os.getenv("WIZER_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="WIZER_TEST_DSN not set")

if DSN:
    import psycopg
    from psycopg.types.json import Jsonb

FIXTURE = Path(__file__).parent / "fixtures" / "production_schema.sql"
DB = "wizer_prodschema"
NEW_MIGRATIONS = ["ingestion_fixes_migration.sql", "enrichment_outputs_migration.sql",
                  "enrichment_queue_migration.sql", "clustering_v2_migration.sql",
                  "archive_migration.sql"]


@pytest.fixture(scope="module")
def prod():
    admin = psycopg.connect(f"{DSN} dbname=postgres", autocommit=True)
    admin.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
    admin.execute(f"CREATE DATABASE {DB}")
    conn = psycopg.connect(f"{DSN} dbname={DB}", autocommit=True)
    for role in ("anon", "authenticated", "service_role"):
        conn.execute(f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='{role}') "
                     f"THEN CREATE ROLE {role}; END IF; END $$")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    sql = FIXTURE.read_text(encoding="utf-8").replace("CREATE SCHEMA public;", "")
    conn.execute(sql)
    yield conn
    conn.close()
    admin.execute(f"DROP DATABASE IF EXISTS {DB} WITH (FORCE)")
    admin.close()


def test_new_migrations_apply_to_production_schema_twice(prod):
    docs = Path(__file__).resolve().parents[1] / "docs"
    for _ in range(2):
        for name in NEW_MIGRATIONS:
            prod.execute((docs / name).read_text(encoding="utf-8"))


def test_crawler_rows_fit_production_columns(prod):
    """Every value the ingestion code writes must fit production's column types."""
    from pipeline import db as pdb
    from pipeline.crawler import CrawledArticle

    feed = {"id": 1, "language_code": "ml", "feed_url": "https://f.example/rss"}
    pdb._derive_language_name(feed)
    prod.execute("INSERT INTO feeds (id, feed_url) VALUES (1, 'https://f.example/rss') ON CONFLICT DO NOTHING")
    art = CrawledArticle(
        feed_id=1, url="https://example.com/a", url_hash=-9_123_456_789_012_345_678 // 1000,
        title="शीर्षक " * 30, title_simhash=-5, description="d " * 500, full_text="x" * 80_000,
        language=feed["language_name"], language_code="ml", country_code="IN",
        domain="example.com", publisher_name="Example", feed_url=feed["feed_url"],
        published_at=datetime(2026, 10, 3, tzinfo=timezone.utc),
    )
    row = art.to_db_row()
    cols = [c for (c,) in prod.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='articles'").fetchall()]
    unknown = set(row) - set(cols)
    assert not unknown, f"code writes columns production lacks: {unknown}"
    vals = [Jsonb(v) if isinstance(v, (dict, list)) else v for v in row.values()]
    prod.execute(f"INSERT INTO articles ({', '.join(row)}) VALUES ({', '.join(['%s'] * len(row))})", vals)
    assert prod.execute("SELECT language FROM articles WHERE url = %s", (row["url"],)).fetchone()[0].strip() == "ml"
