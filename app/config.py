"""Environment configuration. Load once, import `settings` everywhere else."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


# Fallback list, tried in order. The per-model daily cap is what makes length matter:
# OpenRouter's free-tier budget is per model, not per account, so N models is N x the
# daily headroom (16 x 50 = 800 calls/day here) and a run only dies once EVERY model
# is exhausted. A three-model list could not finish a single report once two of them
# were spent.
#
# Ordered by how reliably each model returns valid JSON for the structured-output
# calls that dominate this pipeline: native structured_outputs support first, then
# response_format support, then the rest (which still work, because
# OpenRouterClient.structured() also inlines the schema as a prompt instruction and
# gets one repair retry).
#
# Verified against https://openrouter.ai/api/v1/models on 2026-09-01. That roster
# rotates often -- re-check it rather than trusting this list indefinitely. Two free
# models are deliberately excluded: nvidia/nemotron-3.5-content-safety:free is a
# safety classifier, not a chat model, and cohere/north-mini-code:free is a
# code-completion specialist; neither does claim extraction usefully.
DEFAULT_OPENROUTER_MODELS: list[str] = [
    # native structured_outputs
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "dots-studio/dots-3-note-preview:free",
    # response_format
    "minimax/minimax-m3:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "minimax/minimax-m2.7:free",
    # schema-by-prompt only
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "thinkingmachines/inkling:free",
    "thinkingmachines/inkling-small:free",
    "inclusionai/ling-3.0-flash-fin:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    # last resort: structured-output capable but only 2.6B params, so weakest at
    # extraction quality -- better than failing the run outright.
    "liquid/lfm-2.5-2.6b:free",
]


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_models: list[str] = field(
        default_factory=lambda: _split_csv(
            os.environ.get("OPENROUTER_MODELS", ",".join(DEFAULT_OPENROUTER_MODELS))
        )
    )
    openrouter_max_concurrency: int = field(
        default_factory=lambda: int(os.environ.get("OPENROUTER_MAX_CONCURRENCY", "3"))
    )
    openrouter_daily_call_budget: int = field(
        default_factory=lambda: int(os.environ.get("OPENROUTER_DAILY_CALL_BUDGET", "50"))
    )
    openrouter_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("OPENROUTER_MAX_RETRIES", "4"))
    )

    courtlistener_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "COURTLISTENER_BASE_URL", "https://www.courtlistener.com/api/rest/v4"
        )
    )
    edgar_user_agent: str = field(
        default_factory=lambda: os.environ.get(
            "EDGAR_USER_AGENT", "InsightIQ research-agent contact@example.com"
        )
    )
    edgar_base_url: str = field(
        default_factory=lambda: os.environ.get("EDGAR_BASE_URL", "https://www.sec.gov")
    )
    edgar_data_url: str = field(
        default_factory=lambda: os.environ.get("EDGAR_DATA_URL", "https://data.sec.gov")
    )
    gdelt_base_url: str = field(
        default_factory=lambda: os.environ.get("GDELT_BASE_URL", "https://api.gdeltproject.org/api/v2/doc/doc")
    )

    db_path: str = field(default_factory=lambda: os.environ.get("DB_PATH", "insightiq.db"))
    cache_ttl_hours: int = field(default_factory=lambda: int(os.environ.get("CACHE_TTL_HOURS", "24")))


settings = Settings()
