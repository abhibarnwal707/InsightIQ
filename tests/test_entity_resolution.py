import httpx
import pytest
import respx

from app.agents import entity_resolution
from app.config import settings
from app.sources import edgar


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


@pytest.mark.asyncio
async def test_resolves_to_chosen_candidate(isolated_db, monkeypatch):
    async def fake_search(query, limit=8):
        return [{"cik": "0000320193", "ticker": "AAPL", "title": "Apple Inc."}]

    async def fake_profile(cik):
        return {"stateOfIncorporation": "CA"}

    monkeypatch.setattr(edgar, "search_candidates", fake_search)
    monkeypatch.setattr(edgar, "get_submission_profile", fake_profile)

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_completion(
                    '{"chosen_candidate": "cand_1", "is_public": true, '
                    '"domain": null, "resolution_notes": "clear match"}'
                ),
            )
        )
        from app.llm.client import OpenRouterClient

        async with OpenRouterClient() as client:
            result = await entity_resolution.resolve_entity("Apple", client)

    assert result.is_public is True
    assert result.cik == "0000320193"
    assert result.ticker == "AAPL"
    assert result.jurisdiction == "CA"


@pytest.mark.asyncio
async def test_none_candidate_cannot_hallucinate_a_cik(isolated_db, monkeypatch):
    async def fake_search(query, limit=8):
        return []  # private company: EDGAR has nothing

    monkeypatch.setattr(edgar, "search_candidates", fake_search)

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_completion(
                    '{"chosen_candidate": "none", "is_public": false, '
                    '"domain": null, "resolution_notes": "privately held"}'
                ),
            )
        )
        from app.llm.client import OpenRouterClient

        async with OpenRouterClient() as client:
            result = await entity_resolution.resolve_entity("Cargill", client)

    assert result.is_public is False
    assert result.cik is None
    assert result.ticker is None


@pytest.mark.asyncio
async def test_edgar_downtime_degrades_to_none_without_crashing(isolated_db, monkeypatch):
    async def failing_search(query, limit=8):
        raise httpx.ConnectError("EDGAR is down")

    monkeypatch.setattr(edgar, "search_candidates", failing_search)

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_completion(
                    '{"chosen_candidate": "none", "is_public": false, '
                    '"domain": null, "resolution_notes": "no candidates given"}'
                ),
            )
        )
        from app.llm.client import OpenRouterClient

        async with OpenRouterClient() as client:
            result = await entity_resolution.resolve_entity("SomeCo", client)

    assert result.is_public is False
    assert result.cik is None
    assert "EDGAR" in result.resolution_notes


def test_choice_model_rejects_ids_not_in_candidate_list():
    model = entity_resolution._build_choice_model(["cand_1", "cand_2"])
    with pytest.raises(Exception):
        model(
            chosen_candidate="cand_99",
            is_public=True,
            domain=None,
            resolution_notes="x",
        )
    # a real candidate id and "none" both validate fine
    model(chosen_candidate="cand_1", is_public=True, domain=None, resolution_notes="x")
    model(chosen_candidate="none", is_public=False, domain=None, resolution_notes="x")
