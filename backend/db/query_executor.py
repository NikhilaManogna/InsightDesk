from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from time import perf_counter

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from backend.security.timeout_manager import TimeoutConfig
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class QueryExecutionError(RuntimeError):
    """Raised when the database cannot execute a validated query."""


class QueryExecutor:
    def __init__(self, engine: Engine, timeout_seconds: int) -> None:
        self.engine = engine
        self.timeout_seconds = timeout_seconds
        TimeoutConfig(timeout_seconds).validate()

    def run(self, sql: str) -> pd.DataFrame:
        logger.info("query_execution_start timeout_seconds=%s", self.timeout_seconds)
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._read_sql, sql)
            try:
                frame = future.result(timeout=self.timeout_seconds)
                elapsed_ms = int((perf_counter() - started) * 1000)
                logger.info("query_execution_success rows=%s latency_ms=%s", len(frame), elapsed_ms)
                return frame
            except FutureTimeout as exc:
                future.cancel()
                logger.warning("query_execution_timeout timeout_seconds=%s", self.timeout_seconds)
                raise QueryExecutionError(
                    f"Query exceeded {self.timeout_seconds} seconds."
                ) from exc
            except Exception as exc:
                elapsed_ms = int((perf_counter() - started) * 1000)
                logger.exception("query_execution_failed latency_ms=%s", elapsed_ms)
                raise QueryExecutionError(str(exc)) from exc

    def _read_sql(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn)
