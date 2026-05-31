from __future__ import annotations

from sqlalchemy.engine import Engine

from backend.db.duckdb_engine import create_duckdb_engine
from backend.db.postgres import create_postgres_engine
from backend.utils.config import Settings


class EngineRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def dialect_for(self, database: str) -> str:
        return "postgres" if database == "PostgreSQL" else "duckdb"

    def engine_for(self, database: str) -> Engine:
        if database == "PostgreSQL":
            return create_postgres_engine(self.settings)
        return create_duckdb_engine(self.settings)
