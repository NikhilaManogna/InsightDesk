from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryCacheEntry:
    value: Any
    expires_at: datetime


class CacheService:
    def __init__(self, ttl_seconds: int, redis_url: str = "", enabled: bool = False) -> None:
        self.ttl_seconds = ttl_seconds
        self._memory: dict[str, MemoryCacheEntry] = {}
        self._redis = self._connect(redis_url) if enabled and redis_url else None

    def make_key(self, namespace: str, *parts: object) -> str:
        raw = ":".join(str(part) for part in parts)
        return f"{namespace}:{sha256(raw.encode('utf-8')).hexdigest()}"

    def get(self, key: str) -> Any | None:
        if self._redis is not None:
            payload = self._redis.get(key)
            if payload is not None:
                logger.info("cache_hit backend=redis key=%s", key)
                return pickle.loads(payload)
            logger.info("cache_miss backend=redis key=%s", key)
            return None

        entry = self._memory.get(key)
        if not entry or entry.expires_at < datetime.now(UTC):
            self._memory.pop(key, None)
            logger.info("cache_miss backend=memory key=%s", key)
            return None
        logger.info("cache_hit backend=memory key=%s", key)
        return entry.value

    def set(self, key: str, value: Any) -> None:
        if self._redis is not None:
            self._redis.setex(key, self.ttl_seconds, pickle.dumps(value))
            return
        self._memory[key] = MemoryCacheEntry(
            value=value,
            expires_at=datetime.now(UTC) + timedelta(seconds=self.ttl_seconds),
        )

    def invalidate_prefix(self, prefix: str) -> None:
        for key in list(self._memory):
            if key.startswith(prefix):
                self._memory.pop(key, None)

    @staticmethod
    def _connect(redis_url: str):
        try:
            import redis

            client = redis.from_url(redis_url)
            client.ping()
            return client
        except Exception:
            logger.warning("redis_unavailable using_memory_cache=true")
            return None
