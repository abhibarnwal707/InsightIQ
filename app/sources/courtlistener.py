"""CourtListener RECAP search. Free tier: 5 req/min, 50/hr, 125/day, no auth needed
for basic reads -- budgeted independently of the OpenRouter budget (design note in
PLAN.md), via app.cache.store's courtlistener_usage table.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.cache import store
from app.config import settings


class CourtListenerBudgetExceededError(RuntimeError):
    pass


_LIMITS = (  # (window_seconds, max_calls, label)
    (60, 5, "5 requests/minute"),
    (3600, 50, "50 requests/hour"),
    (86400, 125, "125 requests/day"),
)


def _check_budget() -> None:
    for window_seconds, max_calls, label in _LIMITS:
        used = store.get_courtlistener_usage(window_seconds)
        if used >= max_calls:
            raise CourtListenerBudgetExceededError(
                f"CourtListener free-tier limit reached: {label} ({used} used)."
            )


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1, max=10),
    reraise=True,
)
async def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    _check_budget()
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as client:
        resp = await client.get(f"{settings.courtlistener_base_url}{path}", params=params)
        store.record_courtlistener_call()
        resp.raise_for_status()
        return resp.json()


async def search_dockets(query: str, max_records: int = 10) -> list[dict[str, Any]]:
    """RECAP (federal court docket) search by party/case name text.

    The query is wrapped in an exact-phrase match -- free-text search over a company
    name alone returns a lot of noise (any case mentioning the word anywhere), and
    exact phrase is a meaningfully better precision/recall tradeoff for a named entity.
    """
    data = await _get("/search/", {"q": f'"{query}"', "type": "r", "order_by": "score desc"})
    return data.get("results", [])[:max_records]
