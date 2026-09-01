"""OpenRouter wrapper: model fallback, bounded concurrency, 429 backoff,
structured-output parse -> validate -> repair -> fallback.

This is the only module in the codebase that talks to OpenRouter. Every LLM call
in every agent goes through OpenRouterClient.chat() or .structured().
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.cache import store
from app.config import settings

logger = logging.getLogger("insightiq.llm")

T = TypeVar("T", bound=BaseModel)


class OpenRouterError(RuntimeError):
    """Base error for all OpenRouter client failures."""


class BudgetExceededError(OpenRouterError):
    """The configured daily call budget has already been spent."""


class AllModelsFailedError(OpenRouterError):
    """Every model in the fallback list failed for this call."""

    def __init__(self, attempts: list[str]):
        super().__init__(f"All models failed: {attempts}")
        self.attempts = attempts


def _is_retryable(exc: BaseException) -> bool:
    """Retry transient server/transport faults on the SAME model.

    A 429 is deliberately NOT retried here. On OpenRouter's free tier a 429 means
    that model is out of capacity or over its rate limit right now, and burning
    stop_after_attempt(4) backing off against it accomplishes nothing that trying the
    next model in the fallback list doesn't accomplish faster -- the fallback list is
    the retry strategy for a 429. Retrying it here also used to cost real budget: see
    the note in _post_once about failed calls being recorded.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class OpenRouterClient:
    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set. Put it in .env (see .env.example)."
            )
        self._models = list(settings.openrouter_models)
        if not self._models:
            raise OpenRouterError("OPENROUTER_MODELS is empty — configure at least one model.")
        self._semaphore = asyncio.Semaphore(settings.openrouter_max_concurrency)
        self._http = httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/insightiq",
                "X-Title": "InsightIQ Due Diligence Agent",
            },
            timeout=httpx.Timeout(90.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _check_budget(self, model: str) -> None:
        # OpenRouter's free-tier daily limit is per-model, not account-wide -- confirmed
        # live: two models in a 4-model fallback list kept returning real 429s from
        # OpenRouter itself while the other two kept succeeding, all in the same run. A
        # single global counter would under-use the fallback list's real combined
        # capacity, so budget is tracked and checked per model.
        used = store.get_usage_by_model_today().get(model, 0)
        if used >= settings.openrouter_daily_call_budget:
            raise BudgetExceededError(
                f"Daily call budget for {model} ({settings.openrouter_daily_call_budget}) "
                f"already used today ({used} calls). Other models in your fallback list may "
                "still have headroom -- this only blocks this one model. Spend $10+ on "
                "OpenRouter credits to raise the per-model cap to 1000/day, or raise "
                "OPENROUTER_DAILY_CALL_BUDGET in .env if your account already has that headroom."
            )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=2, max=30),
        reraise=True,
    )
    async def _post_once(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            self._check_budget(model)
            response = await self._http.post(
                "/chat/completions", json={**payload, "model": model}
            )
            # Only an ACCEPTED request counts against the daily budget. This used to
            # record before raise_for_status(), so every rejection counted -- and since
            # 429s were also retried up to 4 times, one logical call could burn four
            # budget slots without ever producing a completion. Two models in a
            # three-model fallback list hit their 50/day cap that way in a single run,
            # on rejections alone. OpenRouter doesn't bill or quota a request it
            # refused, so neither do we.
            if response.status_code < 400:
                store.record_llm_call(model)
        response.raise_for_status()
        return response.json()

    async def _try_models(
        self, payload: dict[str, Any], models: list[str] | None = None
    ) -> tuple[dict[str, Any], str]:
        """Try each model in the fallback list in order. Returns (response_json, model_used)."""
        model_list = models or self._models
        attempts: list[str] = []
        budget_blocked: set[str] = set()
        last_exc: Exception | None = None
        for model in model_list:
            try:
                result = await self._post_once(model, payload)
                return result, model
            except Exception as exc:  # noqa: BLE001 — deliberately broad: any model failure falls through
                logger.warning("model %s failed: %s", model, exc)
                attempts.append(f"{model}: {exc}")
                if isinstance(exc, BudgetExceededError):
                    budget_blocked.add(model)
                last_exc = exc
                continue
        if budget_blocked == set(model_list):
            raise BudgetExceededError(
                f"All {len(model_list)} configured model(s) have hit their daily call budget."
            ) from last_exc
        raise AllModelsFailedError(attempts) from last_exc

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        models: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Raw chat completion with model fallback. Returns (message_dict, model_used)."""
        payload: dict[str, Any] = {"messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        response, model_used = await self._try_models(payload, models)
        message = response["choices"][0]["message"]
        return message, model_used

    async def structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        models: list[str] | None = None,
    ) -> tuple[T, str]:
        """Structured-output call: parse -> validate -> one repair retry -> next model.

        Returns (validated_instance, model_used).
        """
        schema = response_model.model_json_schema()
        schema_text = json.dumps(schema)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": schema,
            },
        }
        # Free models frequently ignore response_format and invent their own JSON shape
        # (observed live: a model asked for {"answer": str} returned {"reply": ...} and
        # {"response": ...} on separate calls despite strict=True). Don't rely on
        # response_format alone — always inline the schema as an explicit instruction too.
        schema_instruction = {
            "role": "system",
            "content": (
                "You must respond with ONLY valid JSON matching this JSON Schema — no prose, "
                f"no markdown code fences, no extra keys, exact field names:\n{schema_text}"
            ),
        }
        model_list = models or self._models
        attempts: list[str] = []
        budget_blocked: set[str] = set()
        for model in model_list:
            payload: dict[str, Any] = {
                "messages": [schema_instruction] + list(messages),
                "temperature": temperature,
                "response_format": response_format,
            }
            if tools:
                payload["tools"] = tools
            try:
                raw = await self._post_once(model, payload)
            except Exception as exc:  # noqa: BLE001
                logger.warning("model %s request failed: %s", model, exc)
                attempts.append(f"{model} (request): {exc}")
                if isinstance(exc, BudgetExceededError):
                    budget_blocked.add(model)
                continue

            content = raw["choices"][0]["message"].get("content", "")
            parsed, err = self._parse_and_validate(content, response_model)
            if parsed is not None:
                return parsed, model

            # one repair retry: feed the validation error AND the schema back to the same model
            repair_messages = [schema_instruction] + list(messages) + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "That response was not valid JSON matching the required schema. "
                        f"Validation error: {err}\n"
                        f"The exact schema you must match:\n{schema_text}\n"
                        "Reply again with ONLY corrected JSON matching the schema, no prose, "
                        "no markdown fences."
                    ),
                },
            ]
            try:
                raw2 = await self._post_once(
                    model,
                    {
                        "messages": repair_messages,
                        "temperature": temperature,
                        "response_format": response_format,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("model %s repair request failed: %s", model, exc)
                attempts.append(f"{model} (repair request): {exc}")
                if isinstance(exc, BudgetExceededError):
                    budget_blocked.add(model)
                continue

            content2 = raw2["choices"][0]["message"].get("content", "")
            parsed2, err2 = self._parse_and_validate(content2, response_model)
            if parsed2 is not None:
                return parsed2, model

            logger.warning("model %s failed schema validation twice: %s", model, err2)
            attempts.append(f"{model} (repair): {err2}")

        if budget_blocked == set(model_list):
            raise BudgetExceededError(
                f"All {len(model_list)} configured model(s) have hit their daily call budget."
            )
        raise AllModelsFailedError(attempts)

    @staticmethod
    def _parse_and_validate(
        content: str, response_model: type[T]
    ) -> tuple[T | None, str | None]:
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON: {exc}"
        try:
            return response_model.model_validate(data), None
        except ValidationError as exc:
            return None, str(exc)
