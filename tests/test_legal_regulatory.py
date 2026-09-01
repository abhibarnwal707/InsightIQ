import httpx
import pytest
import respx

from app.agents import legal_regulatory
from app.cache import store
from app.config import settings
from app.llm.schemas import ResolvedEntity
from app.sources import courtlistener, edgar


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


@pytest.mark.asyncio
async def test_no_sources_returns_zero_confidence(isolated_db, monkeypatch):
    async def fake_search_dockets(query, max_records=6):
        return []

    monkeypatch.setattr(courtlistener, "search_dockets", fake_search_dockets)

    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="Cargill", is_public=False)
    async with OpenRouterClient() as client:
        section = await legal_regulatory.run_legal_regulatory_agent(entity, client)

    assert section.confidence == 0.0
    assert section.claims == []
    assert any("not SEC-registered" in gap for gap in section.data_gaps)


@pytest.mark.asyncio
async def test_docket_claims_are_extracted_and_verified(isolated_db, monkeypatch):
    async def fake_search_dockets(query, max_records=6):
        return [
            {
                "caseName": "Doe v. TestCo, Inc.",
                "court_citation_string": "N.D. Cal.",
                "dateFiled": "2024-05-01",
                "docketNumber": "3:24-cv-01234",
                "docket_absolute_url": "/docket/123/doe-v-testco/",
                "recap_documents": [{"description": "COMPLAINT filed by Jane Doe"}],
            }
        ]

    monkeypatch.setattr(courtlistener, "search_dockets", fake_search_dockets)

    from app.llm.client import OpenRouterClient

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_completion(
                        '{"summary": "TestCo faces a civil suit.", '
                        '"claims": [{"text": "Jane Doe filed a complaint against TestCo, Inc. '
                        'in the Northern District of California on 2024-05-01", "source_id": "docket_1"}], '
                        '"data_gaps": []}'
                    ),
                ),
                httpx.Response(200, json=_completion('{"entailment": "yes", "rationale": "matches docket"}')),
            ]
        )
        entity = ResolvedEntity(company_name="TestCo", is_public=False)
        async with OpenRouterClient() as client:
            section = await legal_regulatory.run_legal_regulatory_agent(entity, client)

    assert len(section.claims) == 1
    assert section.claims[0].entailment == "yes"
    assert section.confidence > 0.0


def test_courtlistener_budget_guard_blocks_after_minute_limit(isolated_db):
    for _ in range(5):
        store.record_courtlistener_call()
    with pytest.raises(courtlistener.CourtListenerBudgetExceededError):
        courtlistener._check_budget()
