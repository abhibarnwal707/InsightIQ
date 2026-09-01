"""SQLite-backed LLM call budget tracking (Phase 0) and report cache (Phase 5).

Plain sqlite3, synchronous. Call volume here is one row per LLM call / cache lookup —
far below where blocking the event loop briefly would matter for a portfolio build.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

from app.config import settings

_DB_PATH = Path(__file__).resolve().parents[2] / settings.db_path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _cursor() -> Iterator[sqlite3.Cursor]:
    conn = _connect()
    try:
        yield conn.cursor()
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                called_at TEXT NOT NULL,
                call_date TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_date ON llm_usage(call_date)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS report_cache (
                entity_key TEXT PRIMARY KEY,
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS courtlistener_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                called_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_courtlistener_usage_called_at ON courtlistener_usage(called_at)"
        )


def record_llm_call(model: str) -> None:
    now = datetime.utcnow()
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO llm_usage (model, called_at, call_date) VALUES (?, ?, ?)",
            (model, now.isoformat(), now.date().isoformat()),
        )


def get_usage_today() -> int:
    today = date.today().isoformat()
    with _cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM llm_usage WHERE call_date = ?", (today,))
        row = cur.fetchone()
        return int(row["n"]) if row else 0


def get_usage_by_model_today() -> dict[str, int]:
    today = date.today().isoformat()
    with _cursor() as cur:
        cur.execute(
            "SELECT model, COUNT(*) AS n FROM llm_usage WHERE call_date = ? GROUP BY model",
            (today,),
        )
        return {row["model"]: int(row["n"]) for row in cur.fetchall()}


def get_cached_report(entity_key: str, ttl_hours: int | None = None) -> str | None:
    ttl = ttl_hours if ttl_hours is not None else settings.cache_ttl_hours
    with _cursor() as cur:
        cur.execute(
            "SELECT report_json, created_at FROM report_cache WHERE entity_key = ?",
            (entity_key,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        created_at = datetime.fromisoformat(row["created_at"])
        if datetime.utcnow() - created_at > timedelta(hours=ttl):
            return None
        return row["report_json"]


def set_cached_report(entity_key: str, report_json: str) -> None:
    now = datetime.utcnow().isoformat()
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO report_cache (entity_key, report_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(entity_key) DO UPDATE SET report_json = excluded.report_json,
                                                    created_at = excluded.created_at
            """,
            (entity_key, report_json, now),
        )


def record_courtlistener_call() -> None:
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO courtlistener_usage (called_at) VALUES (?)",
            (datetime.utcnow().isoformat(),),
        )


def get_courtlistener_usage(window_seconds: int) -> int:
    """Calls in the trailing `window_seconds`. Used for CourtListener's own 5/min,
    50/hr, 125/day free-tier limits, tracked independently of the OpenRouter budget.
    """
    cutoff = (datetime.utcnow() - timedelta(seconds=window_seconds)).isoformat()
    with _cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM courtlistener_usage WHERE called_at >= ?", (cutoff,)
        )
        row = cur.fetchone()
        return int(row["n"]) if row else 0


init_db()
