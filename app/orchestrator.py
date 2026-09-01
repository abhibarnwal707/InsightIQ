"""Orchestration: entity resolution -> bounded-concurrency fan-out across the five
section agents -> cross-section consistency check -> assembled report.

Caching (design: SQLite, TTL from settings.cache_ttl_hours) has two tiers so a
repeat request makes truly zero new LLM calls, not just zero for the expensive part:

1. Entity resolution, cached by the raw (normalized) query string -- resolving
   "Apple" costs one LLM call; without this tier, a cache hit on the report below
   would still burn that call every time just to compute the report's cache key.
2. The full report, cached by the resolved entity (CIK for public companies, the
   normalized name for private ones, which have no CIK).

True network concurrency across the 5 section agents is bounded by
OpenRouterClient's own semaphore (they all share one client instance), not by
anything here -- "bounded concurrent fan-out" doesn't need a second limiter on top.
"""
from __future__ import annotations

import asyncio

from app.agents.competitors import run_competitors_agent
from app.agents.consistency import run_consistency_check
from app.agents.entity_resolution import resolve_entity
from app.agents.financials import run_financials_agent
from app.agents.key_people import run_key_people_agent
from app.agents.legal_regulatory import run_legal_regulatory_agent
from app.agents.news import run_news_agent
from app.cache import store
from app.llm.client import OpenRouterClient
from app.llm.schemas import DueDiligenceReport, ResolvedEntity


def _entity_cache_key(company_name: str) -> str:
    return f"entity:{company_name.strip().lower()}"


def _report_cache_key(entity: ResolvedEntity) -> str:
    if entity.is_public and entity.cik:
        return f"report:cik:{entity.cik}"
    return f"report:private:{entity.company_name.strip().lower()}"


async def _resolve_entity_cached(company_name: str, client: OpenRouterClient) -> ResolvedEntity:
    cache_key = _entity_cache_key(company_name)
    cached = store.get_cached_report(cache_key)
    if cached:
        return ResolvedEntity.model_validate_json(cached)
    entity = await resolve_entity(company_name, client)
    store.set_cached_report(cache_key, entity.model_dump_json())
    return entity


async def run_due_diligence(company_name: str, client: OpenRouterClient) -> DueDiligenceReport:
    calls_before = store.get_usage_today()

    entity = await _resolve_entity_cached(company_name, client)
    report_cache_key = _report_cache_key(entity)

    cached_report = store.get_cached_report(report_cache_key)
    if cached_report:
        report = DueDiligenceReport.model_validate_json(cached_report)
        calls_used = store.get_usage_today() - calls_before
        return report.model_copy(update={"llm_calls_used": calls_used})

    sections = await asyncio.gather(
        run_financials_agent(entity, client),
        run_news_agent(entity, client),
        run_legal_regulatory_agent(entity, client),
        run_competitors_agent(entity, client),
        run_key_people_agent(entity, client),
    )

    contradictions = await run_consistency_check(list(sections), client)

    calls_used = store.get_usage_today() - calls_before
    report = DueDiligenceReport(
        company=entity.company_name,
        resolved_entity=entity.model_dump(mode="json"),
        sections=list(sections),
        contradictions=contradictions,
        llm_calls_used=calls_used,
    )
    store.set_cached_report(report_cache_key, report.model_dump_json())
    return report
