"""News section agent. Reuses the Phase 2 pattern from app/agents/common.py exactly.

GDELT only returns article metadata (title/url/date/domain), not body text, so
passages are headline-only and the extraction instruction is scoped accordingly --
claims can't say more than a headline actually states. That honestly-thin sourcing
is exactly what design principle #7 wants reflected in a lower confidence, not
papered over.
"""
from __future__ import annotations

from datetime import datetime

from app.agents.common import extract_and_verify, split_verified_claims
from app.llm.client import OpenRouterClient
from app.llm.schemas import ReportSection, ResolvedEntity, SourcePassage
from app.scoring.confidence import ClaimScoringInput, days_since, score_section
from app.sources import gdelt


def _parse_seendate(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError:
        return None


async def run_news_agent(entity: ResolvedEntity, client: OpenRouterClient) -> ReportSection:
    query = gdelt.format_query(entity.company_name)
    try:
        articles = await gdelt.search_articles(query, timespan="60d", max_records=20)
    except Exception as exc:  # noqa: BLE001
        detail = str(exc) or f"{type(exc).__name__} (no error detail)"
        return ReportSection(
            section="news",
            summary=f"News retrieval failed for {entity.company_name}.",
            claims=[],
            confidence=0.0,
            confidence_rationale="GDELT request failed; nothing to ground news claims in.",
            data_gaps=[
                f"GDELT news retrieval failed: {detail}",
                "News coverage is therefore absent from this report entirely -- treat the "
                "lack of a news section as missing data, not as an absence of news.",
            ],
            model_used="n/a",
        )

    if not articles:
        return ReportSection(
            section="news",
            summary=f"No recent news coverage of {entity.company_name} was found via GDELT.",
            claims=[],
            confidence=0.0,
            confidence_rationale="No articles retrieved; nothing to score.",
            data_gaps=["No recent news articles found via GDELT for this query."],
            model_used="n/a",
        )

    passages = [
        SourcePassage(
            id=f"news_{i + 1}",
            url=a["url"],
            title=a.get("title", ""),
            text=a.get("title", ""),
            source_tier="news",
            published_at=_parse_seendate(a.get("seendate")),
        )
        for i, a in enumerate(articles)
        if a.get("url") and a.get("title")
    ]

    instruction = (
        "Extract factual claims about recent news or events concerning the company, based "
        "ONLY on these headlines (full article text is not available). Do not infer or add "
        "detail beyond what a headline itself states."
    )
    claims, summary, data_gaps, model_used = await extract_and_verify(
        client, passages, instruction, max_claims=8
    )
    verified, drop_notes = split_verified_claims(claims)

    tone_note = ""
    try:
        tone = await gdelt.tone_snapshot(query, timespan="60d", max_records=50)
        neg = f"{tone['negative_count']}{'+' if tone['negative_at_cap'] else ''}"
        pos = f"{tone['positive_count']}{'+' if tone['positive_at_cap'] else ''}"
        tone_note = (
            f" Of the last {tone['capped_at']} articles checked, {neg} had strongly negative "
            f"tone and {pos} had strongly positive tone (GDELT tone filter, approximate)."
        )
    except Exception:  # noqa: BLE001
        data_gaps.append("Could not compute a tone/sentiment snapshot from GDELT.")

    passage_by_id = {p.id: p for p in passages}
    scoring_inputs = [
        ClaimScoringInput(
            entailment=c.entailment or "no",
            source_tier="news",
            recency_days=(
                days_since(passage_by_id[c.source_id].published_at)
                if passage_by_id.get(c.source_id) and passage_by_id[c.source_id].published_at
                else 30.0
            ),
            corroboration_count=1,
        )
        for c in verified
    ]
    section_score = score_section(scoring_inputs)

    data_gaps = data_gaps + drop_notes
    data_gaps.append(
        "Claims are grounded in headlines only -- GDELT does not provide full article body "
        "text, so claim granularity is limited to what a headline states."
    )

    return ReportSection(
        section="news",
        summary=f"{summary}{tone_note}".strip(),
        claims=verified,
        confidence=section_score.confidence,
        confidence_rationale=section_score.rationale,
        data_gaps=data_gaps,
        model_used=model_used,
    )
