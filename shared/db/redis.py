"""Redis connection factory — async with in-memory fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_pool = None


async def get_redis():
    """Get or create async Redis connection pool."""
    global _redis_pool
    if _redis_pool is not None:
        return _redis_pool
    try:
        import redis.asyncio as aioredis
        _redis_pool = aioredis.from_url(REDIS_URL, decode_responses=True)
        await _redis_pool.ping()
        logger.info("Redis connected: %s", REDIS_URL)
        return _redis_pool
    except Exception as exc:
        logger.warning("Redis unavailable (%s), using in-memory fallback", exc)
        _redis_pool = InMemoryRedis()
        return _redis_pool


class InMemoryRedis:
    """In-memory Redis fallback for dev/test environments."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._ttls: dict[str, float] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value

    async def hgetall(self, key: str) -> dict:
        val = self._store.get(key, {})
        return val if isinstance(val, dict) else {}

    async def hset(self, key: str, mapping: dict) -> None:
        self._store[key] = mapping

    async def delete(self, *keys: str) -> None:
        for k in keys:
            self._store.pop(k, None)

    async def smembers(self, key: str) -> set:
        val = self._store.get(key, set())
        return val if isinstance(val, set) else set()

    async def sadd(self, key: str, *members: str) -> None:
        if key not in self._store:
            self._store[key] = set()
        self._store[key].update(members)

    async def ping(self) -> bool:
        return True

    def pipeline(self, transaction: bool = False):
        return InMemoryPipeline(self)


class InMemoryPipeline:
    def __init__(self, redis: InMemoryRedis):
        self._redis = redis
        self._commands: list = []

    def get(self, key: str):
        self._commands.append(("get", key))
        return self

    async def execute(self) -> list:
        results = []
        for cmd, key in self._commands:
            if cmd == "get":
                results.append(await self._redis.get(key))
        self._commands.clear()
        return results
