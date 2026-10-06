# Sources: what WIZER collects, from where, and why

Verified on 2026-10-06 by fetching every endpoint (residential IP, plus GitHub runners for
the sources already in production). Re-run `tools/register_official_sources.py` and
`tools/discover_sitemaps.py` to re-verify. Both report every source that stops answering.

## 1. How a source enters the pipeline

Every source is a row in `feeds`. Its `feed_type` picks the fetcher, and every fetcher returns
feedparser-shaped entries. From there the path is the same for all sources: dedup, discovery,
insert, hand-off, deferred crawl (page or PDF), enrichment and clustering.

| `feed_type` | Fetcher | Cadence |
|---|---|---|
| `rss` / NULL | feedparser (`pipeline/poller.py`) | by `update_cadence` |
| `news_sitemap` | Google News sitemap reader (`pipeline/sitemap.py`) | hourly (`breaking_news`) |
| `pib_daily`, `nse_announcements`, `bse_announcements`, `sebi_listing` | `pipeline/official.py` | every 15 min (`official`) |

**User-Agent.** PIB answers 403, and NSE never answers, to a User-Agent containing "Bot" or a
"+https://" URL. The old `NewsIngestBot/2.0` silently failed every PIB, BSE and NSE feed.
WIZER now sends `WIZER-NewsReader/2.0`.

## 2. News publishers: news sitemaps (completeness) + RSS

A publisher's news sitemap lists every article from the last ~48 h, with title, publication
date and language. On 50 major Indian publishers, sitemaps listed ~28,500 articles/day; their
RSS feeds listed ~3,400. RSS covered 11.7 % of the union, sitemaps 96 %. Examples:
- Jagran: 1,066/day in the sitemap vs 2 in RSS
- Navbharat Times: 479 vs 0
- Dainik Bhaskar: ~10,000/day across 28 edition sitemaps

`tools/discover_sitemaps.py` finds and registers sitemaps:
- **Candidates:** `Sitemap:` lines in `robots.txt`, plus well-known paths.
- **Measurement:** each candidate is measured with the production reader.
- **Accepted only if** at least half its entries use the news protocol (per-article date and
  title) and it has few query-string URLs. This rule rejects sitemaps that are really
  homepages or archives.

First run (2026-10-06): 106 sitemaps on 98 curated publishers, 37,428 articles in 24 h, across
11 Indian languages. Domain list: `data/india_news_domains.txt`. `--from-db` also probes every
Indian domain in `feeds`.

Publishers without a news sitemap keep their RSS feeds:
- Lokmat, Manorama, Sakshi, Tribune (section feeds), Gujarat Samachar, Sandesh, Udayavani.
- NDTV, Zee, Telegraph and Anandabazar block datacentre IPs (Akamai). They are reached via RSS
  where it answers.

## 3. Official sources

| Source | `feed_type` / endpoint | Quirks handled | Volume |
|---|---|---|---|
| PIB press releases (14 languages, 28 offices) | `pib_daily`: `pib.gov.in/allRel.aspx?reg=R&lang=L` | See below | 200–350 releases/day |
| NSE corporate announcements | `nse_announcements`: `nsearchives.nseindia.com/content/RSS/Online_announcements.xml` | Dates are `06-Oct-2026 12:39:10` IST. Mutual-fund NAV rows (empty link / NAV text) are dropped. Links are PDFs, so their text is extracted in the deferred crawl | ~700 filings/day |
| BSE corporate announcements | `bse_announcements`: `bseindia.com/data/xml/announcements.xml` | ~4,100 of ~4,400 daily rows are fund NAVs (scrip code ≥ 9,000,000 or NAV text) and are dropped | ~950 filings/day |
| SEBI: orders, circulars, PRs, consultations, filings | `sebi_listing`: POST `sebi.gov.in/sebiweb/ajax/home/getnewslistallinfo.jsp`, yesterday + today, every page | See below | 15–25/business day |
| RBI press releases, notifications, speeches, publications, bulletin | `rss`: `rbi.org.in/*_rss.xml` | Description holds the full text. pubDate is naive IST, so it is read as IST (see §4) | ~10–15/day |
| NSE circulars, board meetings, corporate actions, offer documents; BSE notices | `rss` | — | small |
| TRAI; Maharashtra DGIPR (Mahasamvad); MP Jansampark (English, Hindi, cabinet) | `rss` | — | small |

**PIB details.**
- The RSS is capped at 20 items and carries no dates or ministry. The daily listing is complete
  and names the ministry, which is stored as a tag.
- The canonical link is `PressReleasePage.aspx?PRID=N`. The same release is listed by several
  offices, so the release ID alone de-duplicates it.
- Between 00:00 and 03:00 IST the previous day is re-swept (ASP.NET date postback). This catches
  late-evening releases.
- The publish time comes from the page's "Posted On: 06 OCT 2026 9:53AM" (IST).

**SEBI details.**
- The RSS lags by hours.
- POST requests need `Referer` and `X-Requested-With`; without them the firewall blocks them.
- Detail pages wrap their document in a PDF.js viewer (`iframe src='…/web/?file=<pdf>'`). The
  crawler follows it, so the order or prospectus text is the article body.

## 4. Dates and time zones

- **Naive feed dates on Indian feeds** (country IND or an Indic language) are IST. Before the
  fix, feedparser read "Tue, 06 Oct 2026 10:30:00" as UTC, which put RBI releases 5 h 30 min in
  the future.
- **Exchange timestamps and PIB "Posted On"** are parsed as IST.
- **Sitemap dates without a zone** are read as IST (`pipeline/sitemap.parse_date`).
- **All storage is UTC.**

## 5. Considered and not collected

| Source | Why not |
|---|---|
| **ePapers** (newspaper page replicas) | Copyrighted compilations behind logins/subscriptions; The Hindu's terms ban scraping outright; robots.txt disallows the ePaper paths of BS, Telegraph, Anandabazar, NBT. Fair dealing does not cover systematic copying. The same journalism is collected from each publisher's public sitemap / RSS. Replicas would need a licence (PressReader, Magzter, or direct syndication). |
| Supreme Court / High Courts | Judgment search is captcha-gated; bypassing a captcha is not acceptable. The SC homepage lists the 25 latest judgments; a future adapter can read that list. |
| MCA | Akamai-blocked and JS-rendered; MCA's press releases arrive via PIB (Ministry of Corporate Affairs). |
| Gazette of India | Postback search, broken TLS chain; low news value per document. |
| NSE / BSE JSON APIs | Complete but undocumented and behind bot protection (NSE's terms forbid automated collection beyond its published RSS). The official RSS feeds are the defensible route; polled every 15 min, they hold the whole current day. |
| Google News RSS | Terms restrict commercial use; publisher sitemaps give the same articles first-hand. |
| GDELT | Not a source; used to measure our miss rate (see `docs/COVERAGE.md` when built). |
