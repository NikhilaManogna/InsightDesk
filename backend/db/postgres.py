from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from backend.utils.config import Settings


def create_postgres_engine(settings: Settings) -> Engine:
    engine = create_engine(
        settings.postgres_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        future=True,
    )
    with engine.connect() as conn:
        conn.execute(text(f"SET statement_timeout = {settings.query_timeout_seconds * 1000}"))
    return engine
