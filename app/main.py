"""FastAPI app entrypoint."""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from app.cache import store
from app.config import settings
from app.llm.client import AllModelsFailedError, BudgetExceededError, OpenRouterClient
from app.orchestrator import run_due_diligence
from app.render.export import default_filename, save_report
from app.render.html import render_html
from app.render.markdown import render_markdown

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="InsightIQ", description="Multi-source due-diligence agent")


@app.get("/health")
async def health() -> dict:
    """Health plus remaining per-model budget.

    The daily cap is per model, so a single total hides the number that actually
    decides whether a run can finish: how many models still have headroom.
    """
    budget = settings.openrouter_daily_call_budget
    used_by_model = store.get_usage_by_model_today()
    models = [
        {"model": m, "used": used_by_model.get(m, 0), "remaining": max(0, budget - used_by_model.get(m, 0))}
        for m in settings.openrouter_models
    ]
    return {
        "status": "ok",
        "llm_calls_used_today": store.get_usage_today(),
        "daily_budget_per_model": budget,
        "models_with_headroom": sum(1 for m in models if m["remaining"] > 0),
        "total_calls_remaining": sum(m["remaining"] for m in models),
        "models": models,
    }


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
    request: ResearchRequest,
    format: Literal["json", "markdown", "html"] = "json",
    download: bool = False,
    save: bool = False,
):
    """Run (or serve from cache) a full due-diligence report for `company_name`.

    This is synchronous: it awaits the full pipeline (entity resolution -> 5 section
    agents -> consistency check) before responding. No progress streaming (SSE/polling)
    is implemented despite PLAN.md's tech-stack section suggesting it -- a disclosed
    scope cut, not a silent one; see README for why.

    `format=html` returns a self-contained report document (see app/render/html.py for
    why HTML is the shareable-file format here). `download=true` sets a
    Content-Disposition attachment header so a browser saves it instead of rendering it
    inline; `save=true` additionally writes a copy server-side under ./reports/.
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

    saved_path = save_report(report, fmt=format) if save else None

    if format == "json":
        body = report.model_dump(mode="json")
        if saved_path:
            body["saved_to"] = str(saved_path)
        return body

    headers: dict[str, str] = {}
    if download:
        headers["Content-Disposition"] = (
            f'attachment; filename="{default_filename(report, format)}"'
        )
    if saved_path:
        headers["X-Report-Saved-To"] = str(saved_path)

    if format == "html":
        return HTMLResponse(render_html(report), headers=headers)
    return PlainTextResponse(
        render_markdown(report), media_type="text/markdown", headers=headers
    )
