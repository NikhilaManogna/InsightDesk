from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any

import pandas as pd


@dataclass
class CacheEntry:
    frame: pd.DataFrame
    created_at: datetime


class QueryCache:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self._items: dict[str, CacheEntry] = {}

    def key(self, database: str, sql: str) -> str:
        raw = f"{database}:{sql}".encode("utf-8")
        return sha256(raw).hexdigest()

    def get(self, key: str) -> pd.DataFrame | None:
        entry = self._items.get(key)
        if not entry:
            return None
        if datetime.utcnow() - entry.created_at > self.ttl:
            self._items.pop(key, None)
            return None
        return entry.frame.copy()

    def set(self, key: str, frame: pd.DataFrame) -> None:
        self._items[key] = CacheEntry(frame=frame.copy(), created_at=datetime.utcnow())

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._items), "ttl_seconds": int(self.ttl.total_seconds())}
