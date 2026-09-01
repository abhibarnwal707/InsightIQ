"""Environment configuration. Load once, import `settings` everywhere else."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openrouter_models: list[str] = field(
        default_factory=lambda: _split_csv(
            os.environ.get(
                "OPENROUTER_MODELS",
                "z-ai/glm-5.2:free,minimax/minimax-m3:free,google/gemma-4-31b-it:free",
            )
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
