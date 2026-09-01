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


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1, max=10),
    reraise=True,
)
async def _get(params: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as client:
        resp = await client.get(_BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json()


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
    data = await _get(params)
    return data.get("articles", [])


async def tone_snapshot(query: str, timespan: str = "30d", max_records: int = 50) -> dict[str, Any]:
    """Approximate negative/positive coverage counts via GDELT's tone query operator."""
    negative = await search_articles(f"{query} tone<-5", timespan=timespan, max_records=max_records)
    positive = await search_articles(f"{query} tone>5", timespan=timespan, max_records=max_records)
    return {
        "negative_count": len(negative),
        "positive_count": len(positive),
        "capped_at": max_records,
        "negative_at_cap": len(negative) >= max_records,
        "positive_at_cap": len(positive) >= max_records,
    }
