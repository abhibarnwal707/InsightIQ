import httpx
import pytest
import respx

from app.agents import key_people
from app.config import settings
from app.llm.schemas import ResolvedEntity
from app.sources import edgar


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


@pytest.mark.asyncio
async def test_private_company_zero_confidence(isolated_db):
    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="Cargill", is_public=False)
    async with OpenRouterClient() as client:
        section = await key_people.run_key_people_agent(entity, client)

    assert section.confidence == 0.0
    assert section.claims == []


@pytest.mark.asyncio
async def test_no_def14a_found_is_a_data_gap_not_a_guess(isolated_db, monkeypatch):
    async def fake_meta(cik, forms=("DEF 14A",)):
        return None

    monkeypatch.setattr(edgar, "get_latest_filing_meta", fake_meta)

    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="TestCo", is_public=True, cik="0000000000")
    async with OpenRouterClient() as client:
        section = await key_people.run_key_people_agent(entity, client)

    assert section.confidence == 0.0
    assert section.claims == []
    assert "Could not locate a DEF 14A" in section.data_gaps[0]


@pytest.mark.asyncio
async def test_extracts_named_executive_from_proxy_text(isolated_db, monkeypatch):
    async def fake_meta(cik, forms=("DEF 14A",)):
        return {"form": "DEF 14A", "accession": "0000000000-24-000002", "filing_date": "2024-03-01", "primary_document": "proxy.htm"}

    async def fake_fetch_text(cik, accession, primary_document):
        return "Jane Smith, age 52, has served as Chief Executive Officer and director since 2019."

    monkeypatch.setattr(edgar, "get_latest_filing_meta", fake_meta)
    monkeypatch.setattr(edgar, "fetch_filing_text", fake_fetch_text)

    from app.llm.client import OpenRouterClient

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_completion(
                        '{"summary": "Jane Smith is CEO.", '
                        '"claims": [{"text": "Jane Smith, age 52, has served as CEO and director since 2019", '
                        '"source_id": "proxy_1"}], "data_gaps": []}'
                    ),
                ),
                httpx.Response(200, json=_completion('{"entailment": "yes", "rationale": "directly stated"}')),
            ]
        )
        entity = ResolvedEntity(company_name="TestCo", is_public=True, cik="0000000000")
        async with OpenRouterClient() as client:
            section = await key_people.run_key_people_agent(entity, client)

    assert len(section.claims) == 1
    assert section.claims[0].entailment == "yes"
    assert "Jane Smith" in section.claims[0].text
