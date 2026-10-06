#!/usr/bin/env python3
"""
tools/cronjob_setup.py
══════════════════════
Primary scheduler for WIZER: cron-job.org jobs that dispatch the GitHub
workflows at exact times (POST .../actions/workflows/<file>/dispatches).

WHY: GitHub's own cron did not fire this repository's schedules after the
workflows were re-enabled on 2026-10-06 (not one scheduled run in 3 hours).
The schedules in the workflow files stay as a fallback.

Same approach as the owner's other pipelines: the cron-job.org console is
driven with Playwright through a logged-in browser profile
(C:\\Users\\HP\\.cronjob-session), because its console API is binary. A job is
created by cloning a template job (one that already POSTs a GitHub dispatch
with the right headers/body), then its title, URL, schedule and
Authorization header are replaced.

TOKEN: a fine-grained GitHub token for parvbangar/wizer-pipeline with
"Actions: read and write", in Desktop\\WizerCronToken.txt. It is never printed.

USAGE:
  python tools/cronjob_setup.py --check            # profile logged in? list WIZER jobs
  python tools/cronjob_setup.py --create           # create every job in JOBS that is missing
  python tools/cronjob_setup.py --create --headed  # watch it (or sign in when asked)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROFILE = r"C:\Users\HP\.cronjob-session"
TOKEN_FILE = Path(r"C:\Users\HP\Desktop\WizerCronToken.txt")
TEMPLATE_JOB = 7682357          # an existing GitHub-dispatch job (BAC deals report) used as the clone source
REPO = "parvbangar/wizer-pipeline"
SHOTS = Path(__file__).resolve().parent / "cronjob_shots"

# (title, workflow file, crontab in UTC) — identical to each workflow's own `schedule:`.
JOBS = [
    ("WIZER official (15 min)",          "ingest-official.yml",         "*/15 * * * *"),
    ("WIZER breaking + sitemaps (1 h)",  "ingest-breaking-news.yml",    "0 * * * *"),
    ("WIZER multiple_daily (3 h)",       "ingest-multiple-daily.yml",   "0 */3 * * *"),
    ("WIZER daily + unknown (12 h)",     "ingest-daily.yml",            "0 */12 * * *"),
    ("WIZER several_weekly",             "ingest-several-weekly.yml",   "0 1 * * *"),
    ("WIZER weekly",                     "ingest-weekly.yml",           "0 2 * * *"),
    ("WIZER monthly",                    "ingest-monthly.yml",          "0 3 * * *"),
    ("WIZER sweeper (2 h)",              "enrichment.yml",              "40 */2 * * *"),
    ("WIZER clustering fallback (2 h)",  "cluster.yml",                 "15 */2 * * *"),
    ("WIZER cluster maintenance (3 h)",  "cluster_maintenance.yml",     "45 */3 * * *"),
    ("WIZER archive (03:00 IST)",        "archive.yml",                 "30 21 * * *"),
    ("WIZER coverage audit (07:00 IST)", "coverage.yml",                "30 1 * * *"),
]


def _token() -> str:
    raw = TOKEN_FILE.read_text(encoding="utf-8-sig")
    m = re.search(r"(github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]+)", raw)
    if not m:
        sys.exit(f"no GitHub token found in {TOKEN_FILE}")
    return m.group(1)


def _open(p, headed: bool):
    ctx = p.chromium.launch_persistent_context(PROFILE, headless=not headed, viewport={"width": 1400, "height": 1400})
    pg = ctx.pages[0] if ctx.pages else ctx.new_page()
    pg.goto("https://console.cron-job.org/jobs", wait_until="networkidle")
    pg.wait_for_timeout(2500)
    if "/login" in pg.url:
        if not headed:
            ctx.close()
            sys.exit("NEED_LOGIN: run with --headed and sign in to cron-job.org")
        import time
        t0 = time.time()
        while "/login" in pg.url and time.time() - t0 < 900:
            time.sleep(2)
        pg.wait_for_timeout(3000)
    return ctx, pg


def existing_titles(pg) -> set[str]:
    body = pg.inner_text("body")
    return {t for t, _, _ in JOBS if t in body}


def create(pg, title: str, workflow: str, cron: str, token: str) -> str:
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{workflow}/dispatches"
    slug = re.sub(r"\W+", "_", title)[:40]
    pg.goto(f"https://console.cron-job.org/jobs/{TEMPLATE_JOB}", wait_until="networkidle")
    pg.wait_for_timeout(2000)
    pg.get_by_role("button", name=re.compile("actions", re.I)).click()
    pg.wait_for_timeout(800)
    pg.get_by_role("menuitem", name=re.compile("clone", re.I)).click()
    try:
        pg.wait_for_url(re.compile(r"/jobs/(?!%d)\d+" % TEMPLATE_JOB), timeout=20000)
    except Exception:
        pass
    pg.wait_for_load_state("networkidle")
    pg.wait_for_timeout(1500)
    if str(TEMPLATE_JOB) in pg.url:
        pg.screenshot(path=str(SHOTS / f"{slug}_noclone.png"), full_page=True)
        return "clone failed"
    texts = pg.locator("input[type=text]")
    texts.nth(0).fill(title)
    texts.nth(1).fill(url)
    enabled = pg.locator("input[type=checkbox]").nth(0)
    if not enabled.is_checked():
        enabled.check(force=True)
    box = None
    for i in range(texts.count()):
        v = texts.nth(i).input_value() or ""
        if " " in v and re.fullmatch(r"[\d*,/\- ]+", v):
            box = texts.nth(i)
            break
    if box is None:
        return "crontab field not found"
    box.fill(cron)
    box.press("Tab")
    pg.wait_for_timeout(1000)
    # Authorization header (Advanced tab): the WIZER token, never printed.
    pg.get_by_role("tab", name=re.compile("advanced", re.I)).click()
    pg.wait_for_timeout(1200)
    texts = pg.locator("input[type=text]")
    auth = False
    for i in range(texts.count() - 1):
        if texts.nth(i).input_value().strip().lower() == "authorization":
            texts.nth(i + 1).fill(f"Bearer {token}")
            auth = True
            break
    if not auth:
        return "Authorization header not found on the clone"
    pg.get_by_role("button", name=re.compile("^save$", re.I)).click()
    pg.wait_for_timeout(3000)
    pg.screenshot(path=str(SHOTS / f"{slug}_saved.png"), full_page=True)
    return "saved" if pg.locator("text=saved successfully").count() or str(TEMPLATE_JOB) not in pg.url else "save unconfirmed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        ctx, pg = _open(p, args.headed)
        have = existing_titles(pg)
        print(f"logged in; WIZER jobs present: {len(have)}/{len(JOBS)}")
        if args.create:
            token = _token()
            for title, wf, cron in JOBS:
                if title in have:
                    print(f"  exists   {title}")
                    continue
                print(f"  {create(pg, title, wf, cron, token):18s} {title}  [{cron}]  → {wf}")
        ctx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
