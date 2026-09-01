"""SEC EDGAR access: ticker/CIK lookup, company facts (XBRL), full-text search.

Every request carries EDGAR_USER_AGENT per SEC's fair-access policy
(https://www.sec.gov/os/webmaster-faq#developers) — unset/generic user agents get
rate-limited or blocked.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

from app.config import settings

_HEADERS = {"User-Agent": settings.edgar_user_agent}

_ticker_map_cache: list[dict[str, Any]] = []
_ticker_map_loaded_at: float = 0.0
_ticker_map_lock = asyncio.Lock()
_TICKER_MAP_TTL_SECONDS = 24 * 3600


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=_HEADERS, timeout=httpx.Timeout(30.0))


async def _load_ticker_map(force: bool = False) -> list[dict[str, Any]]:
    """Fetch and cache SEC's company_tickers.json: {cik_str, ticker, title} per registrant."""
    global _ticker_map_cache, _ticker_map_loaded_at
    now = time.time()
    if not force and _ticker_map_cache and (now - _ticker_map_loaded_at) < _TICKER_MAP_TTL_SECONDS:
        return _ticker_map_cache
    async with _ticker_map_lock:
        # re-check after acquiring the lock in case another task just refreshed it
        now = time.time()
        if not force and _ticker_map_cache and (now - _ticker_map_loaded_at) < _TICKER_MAP_TTL_SECONDS:
            return _ticker_map_cache
        async with _client() as client:
            resp = await client.get(f"{settings.edgar_base_url}/files/company_tickers.json")
            resp.raise_for_status()
            data = resp.json()
        _ticker_map_cache = list(data.values())
        _ticker_map_loaded_at = now
        return _ticker_map_cache


def _pad_cik(cik: int | str) -> str:
    return str(cik).zfill(10)


async def search_candidates(query: str, limit: int = 8) -> list[dict[str, Any]]:
    """Deterministic, code-only search over SEC's registrant list — no LLM involved.

    Matches on exact ticker, exact title, or title containing every whitespace-separated
    token of the query. Returns [{cik, ticker, title}], CIK zero-padded to 10 digits.
    """
    tickers = await _load_ticker_map()
    q = query.strip().lower()
    if not q:
        return []
    tokens = q.split()

    exact: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for row in tickers:
        title = str(row.get("title", ""))
        ticker = str(row.get("ticker", ""))
        title_l = title.lower()
        ticker_l = ticker.lower()
        if q == ticker_l or q == title_l:
            exact.append(row)
        elif all(tok in title_l for tok in tokens):
            partial.append(row)

    ordered = exact + [r for r in partial if r not in exact]
    results = [
        {
            "cik": _pad_cik(row["cik_str"]),
            "ticker": row.get("ticker", ""),
            "title": row.get("title", ""),
        }
        for row in ordered[:limit]
    ]
    return results


async def get_submission_profile(cik: str) -> dict[str, Any]:
    """Company profile from data.sec.gov/submissions/CIK##########.json.

    Includes stateOfIncorporation (jurisdiction), SIC description, exchanges, etc.
    """
    padded = _pad_cik(cik)
    async with _client() as client:
        resp = await client.get(f"{settings.edgar_data_url}/submissions/CIK{padded}.json")
        resp.raise_for_status()
        return resp.json()


async def get_company_facts(cik: str) -> dict[str, Any]:
    """Raw XBRL company facts from data.sec.gov/api/xbrl/companyfacts/CIK##########.json."""
    padded = _pad_cik(cik)
    async with _client() as client:
        resp = await client.get(
            f"{settings.edgar_data_url}/api/xbrl/companyfacts/CIK{padded}.json"
        )
        resp.raise_for_status()
        return resp.json()


def filing_index_url(cik: str, accession: str) -> str:
    cik_int = str(int(cik))
    accn_nodash = accession.replace("-", "")
    return f"{settings.edgar_base_url}/Archives/edgar/data/{cik_int}/{accn_nodash}/{accession}-index.htm"


def filing_document_url(cik: str, accession: str, primary_document: str) -> str:
    cik_int = str(int(cik))
    accn_nodash = accession.replace("-", "")
    return f"{settings.edgar_base_url}/Archives/edgar/data/{cik_int}/{accn_nodash}/{primary_document}"


async def get_latest_filing_meta(cik: str, forms: tuple[str, ...] = ("10-K",)) -> dict[str, Any] | None:
    """Most recent filing from the submissions API's recent-filings window.

    `forms` is a PRIORITY ORDER, not a set: the newest "10-K" wins over a strictly
    newer "10-K/A", and a later form is only considered if no filing of an earlier
    one exists at all.

    This ordering is the whole point. A 10-K/A is almost always a Part III amendment
    carrying only executive-compensation content -- no MD&A, no Competition section,
    no Legal Proceedings -- so picking it merely because it is newer silently swaps
    the annual report for a compensation exhibit, and every narrative section then
    grounds itself in the wrong document. Observed live on Tesla (CIK 0001318605):
    the 2026-04-30 10-K/A contains zero occurrences of "competition", "legal
    proceedings" or "results of operations", while the 2026-01-29 10-K contains
    12, 7 and 18 respectively.
    """
    profile = await get_submission_profile(cik)
    recent = profile.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    report_dates = recent.get("reportDate", [None] * len(form_list))

    for wanted in forms:
        matches = [i for i, form in enumerate(form_list) if form == wanted]
        if not matches:
            continue
        # EDGAR returns these arrays newest-first, but sort explicitly rather than
        # trusting that ordering for a choice this consequential.
        i = max(matches, key=lambda idx: recent["filingDate"][idx])
        return {
            "form": form_list[i],
            "accession": recent["accessionNumber"][i],
            "filing_date": recent["filingDate"][i],
            "primary_document": recent["primaryDocument"][i],
            "report_date": report_dates[i] if i < len(report_dates) else None,
        }
    return None


_filing_text_cache: dict[tuple[str, str], str] = {}


async def fetch_filing_text(cik: str, accession: str, primary_document: str) -> str:
    """Fetch a filing's primary document and strip it down to plain text.

    Cached in-process by (cik, accession): the financials and legal/regulatory agents
    both pull text from the same latest 10-K, and it's routinely 1MB+ of HTML -- no
    reason to fetch it twice in the same run.
    """
    cache_key = (cik, accession)
    if cache_key in _filing_text_cache:
        return _filing_text_cache[cache_key]

    url = filing_document_url(cik, accession, primary_document)
    async with _client() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    result = "\n".join(line for line in lines if line)
    _filing_text_cache[cache_key] = result
    return result


async def full_text_search(query: str, forms: list[str] | None = None, limit: int = 10) -> list[dict[str, Any]]:
    """EDGAR full-text search (efts.sec.gov) over filings since 2001. Returns raw hit dicts."""
    params: dict[str, Any] = {"q": query, "forms": ",".join(forms) if forms else None}
    params = {k: v for k, v in params.items() if v is not None}
    async with _client() as client:
        resp = await client.get("https://efts.sec.gov/LATEST/search-index", params=params)
        resp.raise_for_status()
        data = resp.json()
    hits = data.get("hits", {}).get("hits", [])
    return hits[:limit]
