import httpx
import pytest
import respx

from app.agents import competitors
from app.config import settings
from app.llm.schemas import ResolvedEntity
from app.sources import edgar


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


@pytest.mark.asyncio
async def test_private_company_has_no_web_fallback_and_zero_confidence(isolated_db):
    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="Cargill", is_public=False)
    async with OpenRouterClient() as client:
        section = await competitors.run_competitors_agent(entity, client)

    assert section.confidence == 0.0
    assert section.claims == []
    assert any("no general web-search" in gap for gap in section.data_gaps)


@pytest.mark.asyncio
async def test_confidence_is_capped_even_for_strong_entailment(isolated_db, monkeypatch):
    async def fake_meta(cik, forms=("10-K",)):
        return {"form": "10-K", "accession": "0000000000-24-000001", "filing_date": "2024-01-01", "primary_document": "doc.htm"}

    async def fake_fetch_text(cik, accession, primary_document):
        return "We compete with several large companies in our industry. Competition is intense."

    monkeypatch.setattr(edgar, "get_latest_filing_meta", fake_meta)
    monkeypatch.setattr(edgar, "fetch_filing_text", fake_fetch_text)

    from app.llm.client import OpenRouterClient

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_completion(
                        '{"summary": "TestCo competes with several large companies.", '
                        '"claims": [{"text": "TestCo competes with several large companies in its industry", '
                        '"source_id": "comp_1"}], "data_gaps": []}'
                    ),
                ),
                httpx.Response(200, json=_completion('{"entailment": "yes", "rationale": "directly stated"}')),
            ]
        )
        entity = ResolvedEntity(company_name="TestCo", is_public=True, cik="0000000000")
        async with OpenRouterClient() as client:
            section = await competitors.run_competitors_agent(entity, client)

    assert section.confidence <= competitors.CONFIDENCE_CEILING
    assert section.claims[0].entailment == "yes"
