from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from backend.utils.config import Settings


def create_duckdb_engine(settings: Settings) -> Engine:
    db_path = Path(settings.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"duckdb:///{db_path}", future=True)
