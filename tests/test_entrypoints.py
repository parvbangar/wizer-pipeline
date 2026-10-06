"""
tests/test_entrypoints.py — every Python file in the repository compiles, and
the CLI entry points answer --help.

Why: on 2026-10-06 enrich.py shipped with a SyntaxError (an f-string broken
across lines). Unit tests import the packages, not the CLI scripts, so all
16 enrichment shards of a production run failed before this test existed.
"""

from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(p for p in ROOT.rglob("*.py")
                 if not any(part.startswith((".venv", ".git")) or part in ("node_modules", "__pycache__")
                            for part in p.relative_to(ROOT).parts))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("script", ["main.py", "enrich.py", "cluster.py", "archive.py",
                                    "tools/discover_sitemaps.py", "tools/register_official_sources.py",
                                    "tools/coverage_audit.py", "tools/language_eval/evaluate.py"])
def test_cli_help(script):
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8")       # help texts carry Indic / box-drawing characters
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
