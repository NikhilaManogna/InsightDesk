from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_timeout_seconds: int
    llm_max_retries: int
    llm_temperature: float
    sql_retry_count: int
    prompt_max_tables: int
    groq_api_key: str
    groq_model: str
    openrouter_api_key: str
    openrouter_model: str
    gemini_api_key: str
    gemini_model: str
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    duckdb_path: str
    query_timeout_seconds: int
    query_limit: int
    log_level: str
    cache_ttl_seconds: int
    max_visualization_rows: int
    visualization_sample_threshold: int
    chart_category_limit: int
    histogram_bins: int
    max_kpis: int
    insight_max_rows_sent: int
    insight_verbosity: str
    anomaly_sensitivity: float
    insight_max_kpis: int
    insight_temperature: float
    insight_max_tokens: int
    schema_rag_enabled: bool
    embedding_model: str
    business_aliases: str
    redis_url: str
    redis_cache_enabled: bool
    generated_sql_cache_ttl_seconds: int
    query_history_path: str
    query_history_limit: int

    @property
    def postgres_url(self) -> str:
        return (
            "postgresql+psycopg2://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    Path("logs").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        llm_timeout_seconds=_as_int("LLM_TIMEOUT_SECONDS", 45),
        llm_max_retries=_as_int("LLM_MAX_RETRIES", 2),
        llm_temperature=_as_float("LLM_TEMPERATURE", 0.0),
        sql_retry_count=_as_int("SQL_RETRY_COUNT", 2),
        prompt_max_tables=_as_int("PROMPT_MAX_TABLES", 6),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=_as_int("POSTGRES_PORT", 5432),
        postgres_db=os.getenv("POSTGRES_DB", "analytics"),
        postgres_user=os.getenv("POSTGRES_USER", "postgres"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        duckdb_path=os.getenv("DUCKDB_PATH", "./cache/insightdesk.duckdb"),
        query_timeout_seconds=_as_int("QUERY_TIMEOUT_SECONDS", 30),
        query_limit=_as_int("QUERY_LIMIT", 1000),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        cache_ttl_seconds=_as_int("CACHE_TTL_SECONDS", 900),
        max_visualization_rows=_as_int("MAX_VISUALIZATION_ROWS", 5000),
        visualization_sample_threshold=_as_int("VISUALIZATION_SAMPLE_THRESHOLD", 10000),
        chart_category_limit=_as_int("CHART_CATEGORY_LIMIT", 25),
        histogram_bins=_as_int("HISTOGRAM_BINS", 30),
        max_kpis=_as_int("MAX_KPIS", 4),
        insight_max_rows_sent=_as_int("INSIGHT_MAX_ROWS_SENT", 30),
        insight_verbosity=os.getenv("INSIGHT_VERBOSITY", "concise"),
        anomaly_sensitivity=_as_float("ANOMALY_SENSITIVITY", 2.0),
        insight_max_kpis=_as_int("INSIGHT_MAX_KPIS", 4),
        insight_temperature=_as_float("INSIGHT_TEMPERATURE", 0.1),
        insight_max_tokens=_as_int("INSIGHT_MAX_TOKENS", 700),
        schema_rag_enabled=_as_bool("SCHEMA_RAG_ENABLED", False),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        business_aliases=os.getenv("BUSINESS_ALIASES", ""),
        redis_url=os.getenv("REDIS_URL", ""),
        redis_cache_enabled=_as_bool("REDIS_CACHE_ENABLED", False),
        generated_sql_cache_ttl_seconds=_as_int("GENERATED_SQL_CACHE_TTL_SECONDS", 900),
        query_history_path=os.getenv("QUERY_HISTORY_PATH", "./cache/query_history.jsonl"),
        query_history_limit=_as_int("QUERY_HISTORY_LIMIT", 200),
    )
