import httpx
import pytest
import respx

from app.config import settings
from app.cache import store
from app.llm.client import (
    AllModelsFailedError,
    BudgetExceededError,
    OpenRouterClient,
)
from pydantic import BaseModel


class Answer(BaseModel):
    answer: str


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"content": content, "role": "assistant"}}]}


@pytest.mark.asyncio
async def test_structured_call_succeeds_on_first_model(isolated_db):
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion('{"answer": "pong"}'))
        )
        async with OpenRouterClient() as client:
            result, model_used = await client.structured(
                [{"role": "user", "content": "ping"}],
                Answer,
                models=["some/model:free"],
            )
        assert result.answer == "pong"
        assert model_used == "some/model:free"
        assert store.get_usage_today() == 1


@pytest.mark.asyncio
async def test_falls_back_to_next_model_on_error(isolated_db):
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(400, json={"error": "bad model"}),
                httpx.Response(200, json=_completion('{"answer": "pong"}')),
            ]
        )
        async with OpenRouterClient() as client:
            result, model_used = await client.structured(
                [{"role": "user", "content": "ping"}],
                Answer,
                models=["bad/model:free", "good/model:free"],
            )
        assert result.answer == "pong"
        assert model_used == "good/model:free"


@pytest.mark.asyncio
async def test_repair_retry_recovers_from_wrong_shape(isolated_db):
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(200, json=_completion('{"reply": "pong"}')),  # wrong field name
                httpx.Response(200, json=_completion('{"answer": "pong"}')),  # repaired
            ]
        )
        async with OpenRouterClient() as client:
            result, model_used = await client.structured(
                [{"role": "user", "content": "ping"}],
                Answer,
                models=["only/model:free"],
            )
        assert result.answer == "pong"
        assert model_used == "only/model:free"


@pytest.mark.asyncio
async def test_all_models_failed_raises(isolated_db):
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": "nope"})
        )
        async with OpenRouterClient() as client:
            with pytest.raises(AllModelsFailedError):
                await client.structured(
                    [{"role": "user", "content": "ping"}],
                    Answer,
                    models=["bad1/model:free", "bad2/model:free"],
                )


@pytest.mark.asyncio
async def test_budget_exceeded_blocks_call_without_network(isolated_db):
    # budget is tracked per model (OpenRouter's free-tier limit is per-model, not
    # account-wide) -- so quota must be filled for the exact model(s) under test.
    for _ in range(settings.openrouter_daily_call_budget):
        store.record_llm_call("some/model:free")

    with respx.mock(base_url=settings.openrouter_base_url, assert_all_called=False) as mock:
        route = mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion('{"answer": "pong"}'))
        )
        async with OpenRouterClient() as client:
            with pytest.raises(BudgetExceededError):
                await client.structured(
                    [{"role": "user", "content": "ping"}],
                    Answer,
                    models=["some/model:free"],
                )
        assert route.call_count == 0


@pytest.mark.asyncio
async def test_budget_exceeded_on_one_model_falls_back_to_another(isolated_db):
    for _ in range(settings.openrouter_daily_call_budget):
        store.record_llm_call("exhausted/model:free")

    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion('{"answer": "pong"}'))
        )
        async with OpenRouterClient() as client:
            result, model_used = await client.structured(
                [{"role": "user", "content": "ping"}],
                Answer,
                models=["exhausted/model:free", "fresh/model:free"],
            )
    assert result.answer == "pong"
    assert model_used == "fresh/model:free"


def test_markdown_fenced_json_is_parsed():
    fenced = '```json\n{"answer": "pong"}\n```'
    parsed, err = OpenRouterClient._parse_and_validate(fenced, Answer)
    assert err is None
    assert parsed.answer == "pong"


@pytest.mark.asyncio
async def test_rejected_429_does_not_consume_daily_budget(isolated_db):
    """A refused request produces no completion, so it must not count against the
    per-model daily cap. It used to: record_llm_call ran before raise_for_status."""
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(return_value=httpx.Response(429))
        async with OpenRouterClient() as client:
            with pytest.raises(AllModelsFailedError):
                await client.structured(
                    [{"role": "user", "content": "ping"}],
                    Answer,
                    models=["some/model:free"],
                )
    assert store.get_usage_today() == 0
    assert store.get_usage_by_model_today() == {}


@pytest.mark.asyncio
async def test_429_falls_through_to_next_model_without_retrying(isolated_db):
    """One attempt per model on a 429 -- the fallback list IS the retry strategy.
    Previously this burned 4 attempts (and 4 budget slots) before moving on."""
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        route = mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(429),
                httpx.Response(200, json=_completion('{"answer": "pong"}')),
            ]
        )
        async with OpenRouterClient() as client:
            result, model_used = await client.structured(
                [{"role": "user", "content": "ping"}],
                Answer,
                models=["busy/model:free", "good/model:free"],
            )

        assert route.call_count == 2  # exactly one attempt on the busy model

    assert result.answer == "pong"
    assert model_used == "good/model:free"
    # only the model that actually answered was charged
    assert store.get_usage_by_model_today() == {"good/model:free": 1}


@pytest.mark.asyncio
async def test_server_errors_are_still_retried_on_the_same_model(isolated_db):
    """5xx is a transient fault, unlike 429 -- retry it rather than burning a model."""
    with respx.mock(base_url=settings.openrouter_base_url) as mock:
        mock.post("/chat/completions").mock(
            side_effect=[
                httpx.Response(503),
                httpx.Response(200, json=_completion('{"answer": "pong"}')),
            ]
        )
        async with OpenRouterClient() as client:
            result, model_used = await client.structured(
                [{"role": "user", "content": "ping"}], Answer, models=["flaky/model:free"]
            )

    assert result.answer == "pong"
    assert model_used == "flaky/model:free"
    assert store.get_usage_by_model_today() == {"flaky/model:free": 1}


def test_default_model_list_is_long_enough_to_survive_exhaustion():
    from app.config import DEFAULT_OPENROUTER_MODELS

    assert len(DEFAULT_OPENROUTER_MODELS) >= 10
    assert len(set(DEFAULT_OPENROUTER_MODELS)) == len(DEFAULT_OPENROUTER_MODELS)
    assert all(m.endswith(":free") for m in DEFAULT_OPENROUTER_MODELS)
