# Coverage: how complete is WIZER, measured

"Every article published in India" is a claim. This page is how it is checked, every day,
against references that do not depend on our own sources.

## Daily audit

`tools/coverage_audit.py` runs every day at 07:00 IST in `.github/workflows/coverage.yml`.
Results go to `coverage_daily` (`docs/coverage_migration.sql`), and `coverage_summary`
aggregates them per day.

| Reference | Ground truth | What is measured |
|---|---|---|
| `sitemap` | Each registered news sitemap: the publisher's own list of what it published | URLs dated D that we did not store → `missing`, `miss_rate` per publisher |
| `gdelt` | GDELT 2.0 GKG, English + translingual, every 2nd 15-min file of day D | Our miss rate on an independent sample; Chapman's Lincoln–Petersen estimate of each publisher's true daily output: `est_universe = (ours+1)(ref+1)/(both+1) − 1`; Indian hosts GDELT saw that have no source |
| `ingest` | Our own stored count per domain | Yield. A domain stored less than half its previous count has a dead source |

Notes:
- **"Stored" uses the ingestion's own URL hashing** (`pipeline.dedup.url_hash`). A miss
  therefore means precisely "the pipeline does not know this URL".
- **Alerts open a GitHub issue.** Two things trigger one: a top-100 sitemap publisher missing
  more than 2 % (with at least 20 URLs), or a yield drop above 50 %.

## Reading the numbers

- **Sitemap miss rate is the strict number.**
  - The publisher says it published the article, so a miss is a real miss.
  - Typical causes: the sitemap feed failed or timed out (see `feeds.fail_count`); an entry
    was older than the poll window (`pipeline/sitemap.py`, `SITEMAP_OVERLAP_HOURS`); the URL
    is normalised differently.
- **The GDELT miss rate covers publishers without a sitemap too.**
  - GDELT samples only part of Indian output: ~180/day of Dainik Bhaskar's ~10,000, and
    nothing in Odia or Assamese. So it is a sample, not a census.
  - Capture–recapture needs the two discovery processes to be independent. They are (GDELT
    crawls on its own), but both favour prominent articles, so `est_universe` is a lower
    bound.
- **The "no source" list is the to-do list for coverage.** Add each host's news sitemap
  (`tools/discover_sitemaps.py --domains <host> --register`) or its RSS.

## Baseline (2026-10-06)

Measured before the news sitemaps were first polled (RSS + breaking news only): see the first
`coverage_summary` row. Section 2 of `docs/OFFICIAL_SOURCES.md` has the reference
measurement: sitemaps listed 28,508 articles/day on 50 publishers, of which our RSS snapshot
held 11.7 %.

## Monthly deep check

CC-NEWS (Common Crawl's news WARCs, ~1 GB/day, hours of latency, no URL index) is not part of
the daily audit. Use it for an occasional long-tail check of hosts outside both the sitemaps
and GDELT.
