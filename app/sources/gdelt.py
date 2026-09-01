"""GDELT DOC 2.0 API: free, no key, no auth. Article metadata (title/url/date/domain)
only -- GDELT does not return per-article body text or per-article tone in ArtList
mode, so claim passages are built from titles, and the tone/sentiment "signal" the
plan asks for is a deterministic, code-computed approximation: how many of the most
recent N articles fall in GDELT's negative-tone (<-5) vs positive-tone (>5) buckets,
via two separate tone-filtered queries. This is capped and approximate (a bucket
count pinned at max_records means "at least", not "exactly") -- it is reported as
such, never dressed up as a precise sentiment score.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import settings

_BASE_URL = settings.gdelt_base_url

# GDELT is a free, unauthenticated service behind a single IP (api.gdeltproject.org
# resolves to one A record, no IPv6). When it throttles a client it does not return
# 429 -- it silently blackholes the TCP handshake, so every request hangs until the
# connect timeout instead of failing cleanly. Observed live: requests succeeding, then
# every subsequent connection timing out for 20+ minutes.
#
# Two consequences shape the settings below:
#  1. The CONNECT timeout must be short and separate from the read timeout. /research
#     is synchronous, so a long connect timeout on a blackholed host stalls the whole
#     report -- at the old flat 45s across 3 calls x 3 retries that was minutes of
#     hanging before the news section gave up.
#  2. Retries must not amplify the burst that triggered the throttle in the first
#     place, so the news path makes fewer calls and backs off harder between them.
_TIMEOUT = httpx.Timeout(connect=8.0, read=30.0, write=10.0, pool=5.0)

# GDELT's docs ask callers to identify themselves; an unset UA is also the first thing
# a free API rate-limits.
_HEADERS = {"User-Agent": "InsightIQ due-diligence research agent (+https://github.com/insightiq)"}


class GdeltUnavailableError(RuntimeError):
    """GDELT could not be reached or returned something that wasn't article JSON.

    Carries a message that names the failure mode: httpx connection errors frequently
    stringify to "" (empty), which previously surfaced in reports as the useless
    "GDELT request failed: " with nothing after the colon.
    """


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2.0, min=2, max=20),
    reraise=True,
)
async def _get(params: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
        resp = await client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        # GDELT answers malformed queries with HTTP 200 and a plain-text error body,
        # so a successful status is not proof of parseable JSON.
        try:
            return resp.json()
        except ValueError as exc:
            snippet = resp.text.strip()[:200]
            raise GdeltUnavailableError(
                f"GDELT returned non-JSON content (HTTP {resp.status_code}): {snippet!r}"
            ) from exc


def format_query(name: str) -> str:
    """Turn a company name into a valid GDELT query term.

    GDELT treats bare whitespace as an implicit OR of the individual words, so an
    unquoted "Tesla, Inc." matches every article mentioning "Inc" -- and trailing
    punctuation can make the query outright invalid once a `tone<-5` operator is
    appended. Multi-word names are therefore quoted into a phrase.
    """
    cleaned = " ".join(name.replace('"', " ").split()).strip(" ,.")
    if not cleaned:
        return '""'
    return f'"{cleaned}"' if " " in cleaned else cleaned


async def search_articles(
    query: str, timespan: str = "30d", max_records: int = 20, sort: str = "DateDesc"
) -> list[dict[str, Any]]:
    """Recent article metadata for `query`. No body text -- see module docstring."""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": sort,
        "timespan": timespan,
    }
    try:
        data = await _get(params)
    except GdeltUnavailableError:
        raise
    except httpx.HTTPStatusError as exc:
        raise GdeltUnavailableError(
            f"GDELT returned HTTP {exc.response.status_code} for query {query!r}"
        ) from exc
    except httpx.TransportError as exc:
        # str(exc) is routinely empty for connection failures -- name the type so the
        # report's data_gaps entry is actually diagnosable.
        detail = str(exc) or "no error detail"
        raise GdeltUnavailableError(
            f"could not connect to GDELT ({type(exc).__name__}: {detail}). The service is "
            "unauthenticated and throttles by IP by dropping connections rather than "
            "returning an error."
        ) from exc
    return data.get("articles", [])


async def tone_snapshot(query: str, timespan: str = "30d", max_records: int = 50) -> dict[str, Any]:
    """Approximate negative/positive coverage counts via GDELT's tone query operator.

    Sequential, not concurrent: these are the 2nd and 3rd calls of the news path and
    firing them together is exactly the burst that gets the client's IP throttled.
    """
    negative = await search_articles(f"{query} tone<-5", timespan=timespan, max_records=max_records)
    positive = await search_articles(f"{query} tone>5", timespan=timespan, max_records=max_records)
    return {
        "negative_count": len(negative),
        "positive_count": len(positive),
        "capped_at": max_records,
        "negative_at_cap": len(negative) >= max_records,
        "positive_at_cap": len(positive) >= max_records,
    }
