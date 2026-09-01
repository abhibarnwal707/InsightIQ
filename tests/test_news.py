import httpx
import pytest
import respx

from app.agents import news
from app.config import settings
from app.llm.schemas import ResolvedEntity
from app.sources import gdelt


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


@pytest.mark.asyncio
async def test_no_articles_returns_zero_confidence_with_data_gap(isolated_db, monkeypatch):
    async def fake_search(query, timespan="60d", max_records=20, sort="DateDesc"):
        return []

    monkeypatch.setattr(gdelt, "search_articles", fake_search)

    from app.llm.client import OpenRouterClient

    entity = ResolvedEntity(company_name="Obscuro Corp", is_public=False)
    async with OpenRouterClient() as client:
        section = await news.run_news_agent(entity, client)

    assert section.confidence == 0.0
    assert section.claims == []
    assert "No recent news articles" in section.data_gaps[0]


@pytest.mark.asyncio
async def test_headline_claims_are_extracted_and_verified(isolated_db, monkeypatch):
    async def fake_search(query, timespan="60d", max_records=20, sort="DateDesc"):
        return [
            {
                "url": "https://example.com/a1",
                "title": "TestCo announces new product line",
                "seendate": "20260101T000000Z",
                "domain": "example.com",
            }
        ]

    async def fake_tone(query, timespan="60d", max_records=50):
        return {"negative_count": 1, "positive_count": 3, "capped_at": 50, "negative_at_cap": False, "positive_at_cap": False}

    monkeypatch.setattr(gdelt, "search_articles", fake_search)
    monkeypatch.setattr(gdelt, "tone_snapshot", fake_tone)

    from app.llm.client import OpenRouterClient

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(
                    200,
                    json=_completion(
                        '{"summary": "TestCo launched a product.", '
                        '"claims": [{"text": "TestCo announced a new product line", "source_id": "news_1"}], '
                        '"data_gaps": []}'
                    ),
                ),
                httpx.Response(200, json=_completion('{"entailment": "yes", "rationale": "matches headline"}')),
            ]
        )
        entity = ResolvedEntity(company_name="TestCo", is_public=True)
        async with OpenRouterClient() as client:
            section = await news.run_news_agent(entity, client)

    assert len(section.claims) == 1
    assert section.claims[0].entailment == "yes"
    assert section.confidence > 0.0
    assert "tone" in section.summary.lower() or "negative" in section.summary.lower()
    assert any("headlines only" in gap for gap in section.data_gaps)
