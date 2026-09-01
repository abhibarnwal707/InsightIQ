"""FastAPI app entrypoint."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.cache import store
from app.llm.client import AllModelsFailedError, BudgetExceededError, OpenRouterClient
from app.orchestrator import run_due_diligence
from app.render.markdown import render_markdown

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="InsightIQ", description="Multi-source due-diligence agent")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_calls_used_today": store.get_usage_today()}


class PingResult(BaseModel):
    answer: str


@app.get("/_debug/ping-llm")
async def ping_llm(prompt: str = "Reply with the single word: pong") -> dict:
    """Minimal OpenRouter connectivity/fallback diagnostic, independent of EDGAR/GDELT/
    CourtListener -- useful when /research fails and you need to know which layer broke.
    """
    try:
        async with OpenRouterClient() as client:
            result, model_used = await client.structured(
                [{"role": "user", "content": prompt}],
                PingResult,
            )
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except AllModelsFailedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"answer": result.answer, "model_used": model_used}


class ResearchRequest(BaseModel):
    company_name: str


@app.post("/research")
async def research(
    request: ResearchRequest, format: Literal["json", "markdown"] = "json"
):
    """Run (or serve from cache) a full due-diligence report for `company_name`.

    This is synchronous: it awaits the full pipeline (entity resolution -> 5 section
    agents -> consistency check) before responding. No progress streaming (SSE/polling)
    is implemented despite PLAN.md's tech-stack section suggesting it -- a disclosed
    scope cut, not a silent one; see README for why.
    """
    if not request.company_name.strip():
        raise HTTPException(status_code=422, detail="company_name must not be empty")
    try:
        async with OpenRouterClient() as client:
            report = await run_due_diligence(request.company_name, client)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except AllModelsFailedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if format == "markdown":
        return PlainTextResponse(render_markdown(report), media_type="text/markdown")
    return report.model_dump(mode="json")
