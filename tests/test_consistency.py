import httpx
import pytest
import respx

from app.agents.consistency import run_consistency_check
from app.config import settings
from app.llm.schemas import ExtractedClaim, ReportSection


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


def _section(name: str, claims: list[ExtractedClaim]) -> ReportSection:
    return ReportSection(
        section=name, summary="", claims=claims, confidence=0.5,
        confidence_rationale="", data_gaps=[], model_used="stub",
    )


@pytest.mark.asyncio
async def test_no_claims_anywhere_short_circuits_without_llm_call(isolated_db):
    from app.llm.client import OpenRouterClient

    sections = [_section("financials", []), _section("news", [])]
    with respx.mock(base_url=settings.openrouter_base_url, assert_all_called=False) as mock:
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion('{"contradictions": []}'))
        )
        async with OpenRouterClient() as client:
            contradictions = await run_consistency_check(sections, client)
        assert route.call_count == 0
    assert contradictions == []


@pytest.mark.asyncio
async def test_detected_contradiction_is_returned(isolated_db):
    from app.llm.client import OpenRouterClient

    sections = [
        _section("financials", [ExtractedClaim(text="Revenue was $10 million in 2023", source_id="xbrl_1", entailment="yes")]),
        _section("news", [ExtractedClaim(text="Revenue was reported as $50 million in 2023", source_id="news_1", entailment="yes")]),
    ]
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_completion(
                    '{"contradictions": ["financials reports $10 million 2023 revenue while '
                    'news reports $50 million for the same year."]}'
                ),
            )
        )
        async with OpenRouterClient() as client:
            contradictions = await run_consistency_check(sections, client)

    assert len(contradictions) == 1
    assert "financials" in contradictions[0]
