"""
pipeline/official.py
════════════════════
Official Indian sources whose own RSS is incomplete or broken, as poller
adapters. Each is a `feeds` row with its own feed_type; the adapter returns
feedparser-shaped entries, so dedup, discovery, insert, hand-off, deferred
crawl (PDF text included) and enrichment are the shared pipeline.

Verified 2026-10-06 (endpoints, quirks and volumes: docs/OFFICIAL_SOURCES.md):

  pib_daily          PIB "all releases of the day" for one office × language
                     (allRel.aspx?reg=R&lang=L). The RSS is capped at 20 items
                     with no dates or ministry; this listing is complete and
                     carries the ministry. Between 00:00 and 03:00 IST the
                     previous day is swept too (ASP.NET date postback), so
                     releases posted late in the evening are never missed.
                     ~200–350 releases/day across 14 languages.
  nse_announcements  NSE corporate announcements RSS (whole current day).
                     Dates are "06-Oct-2026 12:39:10" IST; mutual-fund NAV
                     rows (empty link / "Declaration of NAV") are dropped.
                     Links are PDFs: the deferred crawl extracts their text.
  bse_announcements  BSE corporate announcements RSS (whole current day,
                     ~4,400 rows of which ~4,100 are fund NAVs with scrip
                     codes ≥ 9,000,000 — dropped).
  sebi_listing       SEBI "all news" listing (POST getnewslistallinfo.jsp) for
                     yesterday + today: orders, circulars, press releases,
                     consultation papers, filings. The RSS lags by hours.

Plain RSS official feeds (RBI, TRAI, state governments) are ordinary feeds;
their naive pubDates are read as IST by crawler.discover_article.
"""

from __future__ import annotations

import html as html_module
import logging
import re
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
PIB_BASE = "https://www.pib.gov.in"
SEBI_LIST_PAGE = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListingAll=yes"
SEBI_LIST_AJAX = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistallinfo.jsp"

FEED_TYPES = ("pib_daily", "nse_announcements", "bse_announcements", "sebi_listing")


def _entry(link: str, title: str, published: datetime | None = None, summary: str = "",
           tags: list[str] | None = None, language: str | None = None) -> dict:
    e = {"link": link, "id": link, "title": re.sub(r"\s+", " ", html_module.unescape(title or "")).strip()}
    if summary:
        e["summary"] = re.sub(r"\s+", " ", html_module.unescape(summary)).strip()
    if published:
        p = published.astimezone(timezone.utc)
        e["published"] = p.isoformat()
        e["published_parsed"] = p.utctimetuple()
    if tags:
        e["tags"] = [{"term": t} for t in tags if t]
    if language:
        e["language"] = language
    return e


# ─────────────────────────────────────────────────────────────────────────────
# PIB
# ─────────────────────────────────────────────────────────────────────────────

_PIB_LANG = {"1": "en", "2": "hi", "3": "ur", "4": "bn", "6": "pa", "8": "kn", "9": "mr", "10": "as",
             "11": "ta", "13": "gu", "14": "mni", "15": "ml", "16": "te", "18": "or"}
_PIB_BLOCK = re.compile(r"<h3 class=['\"]font104['\"]>(.*?)</h3>(.*?)(?=<h3 class=['\"]font104|</div>)", re.S)
_PIB_LINK = re.compile(r"<a[^>]+title=['\"](.*?)['\"][^>]+href=['\"][^'\"]*PRID=(\d+)[^'\"]*['\"][^>]*>(.*?)</a>", re.S)


def _pib_params(url: str) -> tuple[str, str]:
    reg = re.search(r"[?&]reg=(\d+)", url)
    lang = re.search(r"[?&]lang=(\d+)", url)
    return (reg.group(1) if reg else "3"), (lang.group(1) if lang else "1")


def parse_pib_listing(html: str, reg: str, lang: str) -> list[dict]:
    """Every release on an allRel.aspx page: canonical link, title, ministry tag, language."""
    out, seen = [], set()
    language = _PIB_LANG.get(lang)
    for ministry_raw, body in _PIB_BLOCK.findall(html):
        ministry = re.sub(r"<[^>]+>", "", html_module.unescape(ministry_raw)).strip()
        for title_attr, prid, text in _PIB_LINK.findall(body):
            if prid in seen:
                continue
            seen.add(prid)
            title = re.sub(r"<[^>]+>", "", text).strip() or title_attr
            # PRID alone: the same release appears on several offices' listings;
            # the page serves it without reg/lang (redirect), so this dedups them.
            out.append(_entry(f"{PIB_BASE}/PressReleasePage.aspx?PRID={prid}",
                              title, tags=[ministry, "PIB"], language=language))
    return out


def _pib_postback_previous_day(client, url: str, html: str, day: datetime) -> str | None:
    """allRel.aspx for another date: the page's own ASP.NET date drop-down postback."""
    fields = dict(re.findall(r"<input[^>]+type=['\"]hidden['\"][^>]+name=['\"]([^'\"]+)['\"][^>]+value=['\"]([^'\"]*)['\"]", html))
    if "__VIEWSTATE" not in fields:
        return None
    fields.update({
        "__EVENTTARGET": "ctl00$ContentPlaceHolder1$ddlday",
        "ctl00$ContentPlaceHolder1$ddlday": str(day.day),
        "ctl00$ContentPlaceHolder1$ddlMonth": str(day.month),
        "ctl00$ContentPlaceHolder1$ddlYear": str(day.year),
    })
    r = client.post(url, data=fields)
    return r.text if r.status_code == 200 else None


def fetch_pib_daily(url: str, client, now: datetime | None = None) -> tuple[list[dict], dict, str | None]:
    now_ist = (now or datetime.now(timezone.utc)).astimezone(IST)
    reg, lang = _pib_params(url)
    try:
        r = client.get(url)
        if r.status_code != 200:
            return [], {}, f"fetch failed: HTTP {r.status_code}"
        entries = parse_pib_listing(r.text, reg, lang)
        meta = {"today": len(entries)}
        if now_ist.hour < 3:                       # late releases of yesterday
            prev = _pib_postback_previous_day(client, url, r.text, now_ist - timedelta(days=1))
            if prev:
                y = parse_pib_listing(prev, reg, lang)
                known = {e["link"] for e in entries}
                entries += [e for e in y if e["link"] not in known]
                meta["yesterday"] = len(y)
        return entries, meta, None
    except Exception as e:
        return [], {}, f"fetch failed: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# NSE / BSE corporate announcements (RSS of the current day)
# ─────────────────────────────────────────────────────────────────────────────

_ITEM = re.compile(r"<item>(.*?)</item>", re.S)
# Mutual-fund / SIF daily NAV notices: thousands a day on both exchanges, no news value.
_NAV = re.compile(r"\b(?:i?SIF\s+)?NAV\b\s*(?:as\b|upload|on\b|for\b|of\b|dated|-|:|\d)|declaration of nav", re.I)


def _field(block: str, name: str) -> str:
    m = re.search(rf"<{name}>\s*(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?\s*</{name}>", block, re.S)
    return html_module.unescape(m.group(1)).strip() if m else ""


def exchange_date(text: str) -> datetime | None:
    """'06-Oct-2026 12:39:10' (IST, as NSE and BSE print it) → aware UTC datetime."""
    try:
        return datetime.strptime(text.strip(), "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def parse_nse_announcements(xml: str) -> list[dict]:
    out = []
    for blk in _ITEM.findall(xml):
        link = _field(blk, "link")
        company = _field(blk, "title")
        desc = _field(blk, "description")
        if not link or _NAV.search(desc):
            continue
        subject = desc.split("|SUBJECT:", 1)[-1].strip() if "|SUBJECT:" in desc else desc
        symbol = link.rsplit("/", 1)[-1].split("_", 1)[0]
        out.append(_entry(link, f"{company}: {subject}" if subject else company,
                          exchange_date(_field(blk, "pubDate")), summary=desc.replace("|SUBJECT:", " — "),
                          tags=["NSE", "corporate filing", symbol], language="en"))
    return out


def parse_bse_announcements(xml: str) -> list[dict]:
    out = []
    for blk in _ITEM.findall(xml):
        link = _field(blk, "link")
        code = _field(blk, "scripcode")
        desc = _field(blk, "description")
        if (not link or (code.isdigit() and int(code) >= 9_000_000) or _NAV.search(desc)
                or desc.strip().lower() == "nav"):                                  # fund NAV rows
            continue
        company = re.sub(r"\s*\(\d+\)\s*$", "", _field(blk, "title"))
        title = f"{company}: {desc}" if desc and desc.lower() not in ("as attached", "-") else company
        out.append(_entry(link, title[:500], exchange_date(_field(blk, "pubDate")), summary=desc,
                          tags=["BSE", "corporate filing", code], language="en"))
    return out


def fetch_exchange_rss(url: str, client, parser) -> tuple[list[dict], dict, str | None]:
    try:
        r = client.get(url)
        if r.status_code != 200:
            return [], {}, f"fetch failed: HTTP {r.status_code}"
        entries = parser(r.text)
        return entries, {"items": r.text.count("<item>"), "kept": len(entries)}, None
    except Exception as e:
        return [], {}, f"fetch failed: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# SEBI
# ─────────────────────────────────────────────────────────────────────────────

_SEBI_ROW = re.compile(r"<tr[^>]*>\s*<td>([^<]+)</td>\s*<td>([^<]*)</td>\s*<td>\s*<a href='([^']+)'[^>]*title=\"(.*?)\"", re.S)


def parse_sebi_listing(html: str) -> list[dict]:
    out = []
    for date_s, category, link, title in _SEBI_ROW.findall(html):
        try:
            d = datetime.strptime(date_s.strip(), "%b %d, %Y").replace(tzinfo=IST)
        except ValueError:
            d = None
        clean = re.sub(r"<.*", "", html_module.unescape(title)).strip()   # titles may embed a PDF link
        out.append(_entry(link.strip(), clean, d, tags=["SEBI", category.strip()], language="en"))
    return out


def fetch_sebi_listing(url: str, client, now: datetime | None = None,
                       max_pages: int = 10) -> tuple[list[dict], dict, str | None]:
    """Yesterday + today, every page (25 rows each)."""
    now_ist = (now or datetime.now(timezone.utc)).astimezone(IST)
    frm, to = (now_ist - timedelta(days=1)).strftime("%d-%m-%Y"), now_ist.strftime("%d-%m-%Y")
    try:
        client.get(SEBI_LIST_PAGE)
        entries: list[dict] = []
        for page in range(max_pages):
            r = client.post(SEBI_LIST_AJAX, headers={"Referer": SEBI_LIST_PAGE, "X-Requested-With": "XMLHttpRequest"},
                            data={"nextValue": "1", "next": "n", "search": "", "fromDate": frm, "toDate": to,
                                  "deptId": "-1", "sid": "-1", "ssid": "-1", "smid": "0", "cid": "-1",
                                  "doDirect": str(page)})
            if r.status_code != 200 or "Blocked" in r.text[:2000]:
                return entries, {}, None if entries else f"fetch failed: HTTP {r.status_code}"
            rows = parse_sebi_listing(r.text)
            entries += rows
            total = re.search(r"of\s+([\d,]+)\s+records", r.text)
            if not rows or (total and len(entries) >= int(total.group(1).replace(",", ""))):
                break
        return entries, {"rows": len(entries)}, None
    except Exception as e:
        return [], {}, f"fetch failed: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch (pipeline/poller.py)
# ─────────────────────────────────────────────────────────────────────────────

def _client():
    import httpx
    from pipeline.config import USER_AGENT
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True)


def fetch_official(feed_type: str, url: str, client=None) -> tuple[list[dict], dict, str | None]:
    own = client is None
    client = client or _client()
    try:
        if feed_type == "pib_daily":
            return fetch_pib_daily(url, client)
        if feed_type == "nse_announcements":
            return fetch_exchange_rss(url, client, parse_nse_announcements)
        if feed_type == "bse_announcements":
            return fetch_exchange_rss(url, client, parse_bse_announcements)
        if feed_type == "sebi_listing":
            return fetch_sebi_listing(url, client)
        return [], {}, f"unknown official feed_type {feed_type!r}"
    finally:
        if own:
            client.close()
