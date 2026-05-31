from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.utils.config import Settings


@dataclass(frozen=True)
class StartupCheck:
    ok: bool
    message: str


def run_startup_checks(settings: Settings) -> list[StartupCheck]:
    checks = [
        StartupCheck(bool(settings.groq_api_key) or settings.llm_provider != "groq", "Groq API key configured"),
        StartupCheck(settings.query_timeout_seconds > 0, "Query timeout is positive"),
        StartupCheck(settings.query_limit > 0, "Query limit is positive"),
    ]
    Path("cache").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    return checks
