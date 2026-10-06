"""
tests/test_ingestion_misc.py
════════════════════════════
Regression tests for push_feeds.py (item 10), requirements-ingest.txt and the
ingest workflows (item 12), and the migration SQL (items 4, 2).
"""

import re
from pathlib import Path

import pytest

import push_feeds

ROOT = Path(__file__).resolve().parent.parent
TABLE_COLS = {"feed_url", "update_cadence", "domain", "is_active", "fail_count", "articles_found"}


# ── item 10: cadence default ─────────────────────────────────────────────────

class TestCadenceDefault:

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_empty_cadence_defaults_to_unknown(self, raw):
        feed = push_feeds._csv_row_to_feed({"feed_url": "https://x.com/rss", "update_cadence": raw}, TABLE_COLS)
        assert feed["update_cadence"] == "unknown"

    def test_missing_csv_column_defaults_to_unknown(self):
        feed = push_feeds._csv_row_to_feed({"feed_url": "https://x.com/rss"}, TABLE_COLS)
        assert feed["update_cadence"] == "unknown"

    def test_valid_cadence_kept_and_normalised(self):
        feed = push_feeds._csv_row_to_feed({"feed_url": "u", "update_cadence": " Daily "}, TABLE_COLS)
        assert feed["update_cadence"] == "daily"

    def test_garbage_cadence_becomes_unknown(self):
        feed = push_feeds._csv_row_to_feed({"feed_url": "u", "update_cadence": "hourly-ish"}, TABLE_COLS)
        assert feed["update_cadence"] == "unknown"

    def test_not_written_if_table_has_no_column(self):
        feed = push_feeds._csv_row_to_feed({"feed_url": "u"}, {"feed_url"})
        assert "update_cadence" not in feed

    def test_unknown_cadence_is_actually_polled_by_a_workflow(self):
        """Every cadence push_feeds can emit must be polled by some workflow."""
        polled = set()
        for wf in (ROOT / ".github" / "workflows").glob("ingest-*.yml"):
            polled |= set(re.findall(r"main\.py --cadence (\w+)", wf.read_text(encoding="utf-8")))
        assert push_feeds._KNOWN_CADENCES <= polled


class TestPushFeedsNormalise:

    def test_uppercase_scheme_and_http_collapse(self):
        assert push_feeds._normalise("HTTP://X.com/feed/") == push_feeds._normalise("https://www.x.com/feed")

    def test_ambiguous_params_kept(self):
        assert push_feeds._normalise("https://x.com/rss?source=a") != push_feeds._normalise("https://x.com/rss?source=b")


# ── item 12: requirements-ingest.txt ─────────────────────────────────────────

def _pins(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([\w.]+)$", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


class TestRequirementsIngest:

    def test_pins_match_requirements_txt(self):
        ingest = _pins(ROOT / "requirements-ingest.txt")
        full = _pins(ROOT / "requirements.txt")
        assert ingest
        for name, ver in ingest.items():
            if name in full:                       # requirements.txt may later `-r` this file
                assert full[name] == ver, name

    def test_has_every_layer1_third_party_import(self):
        ingest = set(_pins(ROOT / "requirements-ingest.txt"))
        for pkg in ("feedparser", "supabase", "mmh3", "python-dotenv", "trafilatura",
                    "readability-lxml", "newspaper3k", "beautifulsoup4", "lxml"):
            assert pkg in ingest

    def test_excludes_ml_stack(self):
        text = (ROOT / "requirements-ingest.txt").read_text(encoding="utf-8").lower()
        for heavy in ("torch", "spacy", "transformers", "sentence-transformers",
                      "lingua", "yake", "imagehash", "pillow"):
            assert not re.search(rf"^{heavy}\b", text, re.M)

    def test_ingest_workflows_use_it(self):
        wfs = list((ROOT / ".github" / "workflows").glob("ingest-*.yml"))
        assert len(wfs) == 7                     # 6 cadences + official (every 15 min)
        process = (ROOT / ".github" / "workflows" / "process.yml").read_text(encoding="utf-8")
        for wf in wfs:
            text = wf.read_text(encoding="utf-8")
            assert "pip install -r requirements-ingest.txt" in text, wf.name
            assert "pip install -r requirements.txt" not in text, wf.name
            # Every ingest run must hand its articles to processing, end before
            # its timeout, and trigger process.yml — or articles are lost.
            assert "WIZER_HANDOFF_PATH" in text and "INGEST_TIME_BUDGET_MINUTES" in text, wf.name
            assert "upload-artifact" in text and "if: always()" in text, wf.name
            name = text.splitlines()[0].removeprefix("name:").strip()
            assert f'"{name}"' in process, f"{wf.name} does not trigger process.yml"


# ── migrations (items 2, 4): static consistency checks ──────────────────────

class TestMigrationSql:

    MIG = (ROOT / "docs" / "migration.sql").read_text(encoding="utf-8")
    FIX = (ROOT / "docs" / "ingestion_fixes_migration.sql").read_text(encoding="utf-8")

    def test_feeds_table_created_before_articles(self):
        assert "CREATE TABLE IF NOT EXISTS feeds" in self.MIG
        assert self.MIG.index("CREATE TABLE IF NOT EXISTS feeds") < self.MIG.index("CREATE TABLE IF NOT EXISTS articles")

    def test_feeds_table_has_every_column_layer1_uses(self):
        block = self.MIG.split("CREATE TABLE IF NOT EXISTS feeds")[1].split(");")[0]
        for col in ("id", "feed_url", "final_url", "domain", "publisher_name", "update_cadence",
                    "language_code", "language_name", "country_code", "iab_tier1", "iab_tier2",
                    "has_paywall", "poll_interval_mins", "priority_score", "is_active",
                    "fail_count", "last_polled_at", "last_success_at", "articles_found",
                    "created_at", "last_new_article_at", "disabled_reason",
                    "validation_tier", "freshness_score"):
            assert re.search(rf"^\s*{col}\s", block, re.M), col
        assert re.search(r"feed_url\s+text\s+NOT NULL UNIQUE", block)

    def test_tier_added_before_step7_backfill(self):
        add = self.MIG.index("ADD COLUMN IF NOT EXISTS tier text")
        use = self.MIG.index("SET cadence = tier")
        assert add < use

    def test_fix_migration_is_idempotent_and_backfills(self):
        assert "ADD COLUMN IF NOT EXISTS last_new_article_at" in self.FIX
        assert "max(created_at)" in self.FIX
        assert "ADD COLUMN IF NOT EXISTS disabled_reason" in self.FIX
        assert "CREATE INDEX IF NOT EXISTS" in self.FIX
        # no bare ADD COLUMN without IF NOT EXISTS
        assert not re.search(r"ADD COLUMN (?!IF NOT EXISTS)", self.FIX)


class TestWorkflowTimingInvariants:
    """A budget that overruns its job timeout silently discards work (2026-10-06 incidents)."""

    def _jobs(self, name):
        import yaml
        return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))["jobs"]

    def test_clustering_budget_plus_setup_fits(self):
        for wf in ("process.yml", "cluster.yml"):
            job = self._jobs(wf)["cluster"]
            run = next(s["run"] for s in job["steps"] if "cluster.py run" in str(s.get("run", "")))
            budget = float(run.split("--time-budget", 1)[1].split()[0])
            assert budget + 15 <= job["timeout-minutes"], wf

    def test_enrichment_timeout_inside_claim_lease(self):
        from enrichment.config import ENRICH_CLAIM_LEASE_MINUTES
        for wf, job_name in (("process.yml", "enrich"), ("enrichment.yml", "enrich")):
            job = self._jobs(wf)[job_name]
            assert job["timeout-minutes"] < ENRICH_CLAIM_LEASE_MINUTES, wf
            text = (ROOT / ".github" / "workflows" / wf).read_text(encoding="utf-8")
            budget = 100 if "--time-budget 100" in text else float(job.get("env", {}).get("ENRICH_TIME_BUDGET_MINUTES", 100))
            assert budget + 15 <= job["timeout-minutes"], wf

    def test_ingest_budget_inside_timeout(self):
        import yaml
        for wf in (ROOT / ".github" / "workflows").glob("ingest-*.yml"):
            data = yaml.safe_load(wf.read_text(encoding="utf-8"))
            job = data["jobs"]["ingest"]
            budgets = [float(s["env"]["INGEST_TIME_BUDGET_MINUTES"]) for s in job["steps"]
                       if "env" in s and "INGEST_TIME_BUDGET_MINUTES" in s["env"]]
            assert budgets and sum(budgets) + 3 <= job["timeout-minutes"], wf.name
