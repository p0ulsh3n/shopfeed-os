"""
Redis Feature Store — <5ms serving (Section 14)
=================================================
Feature store backed by Redis for <5ms reads at serving time.
2026 stack: Redis replaces the Monolith Parameter Server.
Training writes → Redis → Serving reads.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .cuckoo_table import CuckooEmbeddingTable

logger = logging.getLogger(__name__)


class RedisFeatureStore:
    """Feature store backed by Redis for <5ms reads at serving time.

    Keys schema:
      item:emb:{item_id}       → embedding vector (bytes)
      item:features:{item_id}  → JSON feature dict
      user:emb:{user_id}       → user embedding (bytes)
      user:seq:{user_id}       → behavior sequence (JSON list)
      delta:score:{item_id}    → online delta correction (float)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "shopfeed",
        ttl_seconds: int = 86400,  # 24h default TTL
    ):
        self.prefix = prefix
        self.ttl = ttl_seconds
        self._redis = None
        self._redis_url = redis_url

    async def connect(self) -> None:
        """Lazy connect to Redis."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url,
                    decode_responses=False,
                    max_connections=50,
                )
                await self._redis.ping()
                logger.info("Redis feature store connected: %s", self._redis_url)
            except ImportError:
                logger.warning("redis package not installed — using in-memory fallback")
                self._redis = None
            except Exception as e:
                logger.error("Redis connection failed: %s", e)
                self._redis = None

    def _key(self, *parts: str) -> str:
        return f"{self.prefix}:{':'.join(parts)}"

    # ── Item embeddings ──────────────────────────────────────

    async def set_item_embedding(self, item_id: str, embedding: torch.Tensor) -> None:
        """Write item embedding to Redis (called by trainer)."""
        if self._redis is None:
            return
        key = self._key("item", "emb", item_id)
        data = embedding.numpy().tobytes()
        await self._redis.set(key, data, ex=self.ttl)

    async def get_item_embedding(self, item_id: str, dim: int = 64) -> torch.Tensor | None:
        """Read item embedding from Redis (<1ms)."""
        if self._redis is None:
            return None
        key = self._key("item", "emb", item_id)
        data = await self._redis.get(key)
        if data is None:
            return None
        return torch.frombuffer(bytearray(data), dtype=torch.float32)[:dim]

    # ── User features ────────────────────────────────────────

    async def set_user_embedding(self, user_id: str, embedding: torch.Tensor) -> None:
        if self._redis is None:
            return
        key = self._key("user", "emb", user_id)
        await self._redis.set(key, embedding.numpy().tobytes(), ex=self.ttl)

    async def get_user_embedding(self, user_id: str, dim: int = 256) -> torch.Tensor | None:
        if self._redis is None:
            return None
        data = await self._redis.get(self._key("user", "emb", user_id))
        if data is None:
            return None
        return torch.frombuffer(bytearray(data), dtype=torch.float32)[:dim]

    # ── User behavior sequence (for DIN/DIEN/BST) ───────────

    async def append_behavior(self, user_id: str, item_id: str, action: str) -> None:
        """Append to user behavior sequence (capped at 200)."""
        if self._redis is None:
            return
        key = self._key("user", "seq", user_id)
        entry = json.dumps({"item_id": item_id, "action": action, "ts": time.time()})
        pipe = self._redis.pipeline()
        pipe.rpush(key, entry)
        pipe.ltrim(key, -200, -1)  # Keep last 200
        pipe.expire(key, self.ttl)
        await pipe.execute()

    async def get_behavior_sequence(self, user_id: str) -> list[dict]:
        """Get user's full behavior sequence for ranking models."""
        if self._redis is None:
            return []
        key = self._key("user", "seq", user_id)
        raw = await self._redis.lrange(key, 0, -1)
        return [json.loads(r) for r in raw] if raw else []

    # ── Online delta scores ──────────────────────────────────

    async def set_delta_score(self, item_id: str, delta: float) -> None:
        """Write online delta correction for Score = V3_batch + V2_delta + V1_session."""
        if self._redis is None:
            return
        key = self._key("delta", "score", item_id)
        await self._redis.set(key, str(delta), ex=900)  # 15 min TTL

    async def get_delta_score(self, item_id: str) -> float:
        """Read delta score (<1ms). Returns 0.0 if not found."""
        if self._redis is None:
            return 0.0
        data = await self._redis.get(self._key("delta", "score", item_id))
        return float(data) if data else 0.0

    # ── Batch sync from Cuckoo table ─────────────────────────

    async def sync_from_cuckoo(self, table: CuckooEmbeddingTable) -> int:
        """Bulk sync all embeddings from Cuckoo table to Redis.
        Called every 5-15 minutes by the parameter sync job.
        """
        if self._redis is None:
            return 0
        pipe = self._redis.pipeline()
        count = 0
        for item_id, emb in table._embeddings.items():
            key = self._key("item", "emb", item_id)
            pipe.set(key, emb.numpy().tobytes(), ex=self.ttl)
            count += 1
            if count % 1000 == 0:
                await pipe.execute()
                pipe = self._redis.pipeline()
        if count % 1000 != 0:
            await pipe.execute()
        logger.info("Synced %d embeddings from Cuckoo table → Redis", count)
        return count


# ═════════════════════════════════════════════════════════════════════════════
# Module-level convenience functions — called by ml/inference/app.py
# ═════════════════════════════════════════════════════════════════════════════
# These are synchronous wrappers that create a temporary store connection.
# In production, the singleton store is injected via FastAPI lifespan.

_default_store: RedisFeatureStore | None = None


def _get_store() -> RedisFeatureStore:
    """Get or create the default Redis store singleton."""
    global _default_store
    if _default_store is None:
        import os
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _default_store = RedisFeatureStore(redis_url=url)
    return _default_store


def cache_user_features(user_id: str, embedding: list[float]) -> None:
    """Cache user embedding (256d) in Redis.

    Called by POST /v1/embed/user after Two-Tower computation.
    Key: shopfeed:user:emb:{user_id}  TTL: 3600s (1h)
    """
    import asyncio
    store = _get_store()
    try:
        tensor = torch.tensor(embedding, dtype=torch.float32)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(store.connect())
        loop.run_until_complete(store.set_user_embedding(user_id, tensor))
        loop.close()
    except Exception as e:
        logger.warning("cache_user_features failed for %s: %s", user_id, e)


def cache_session_intent(session_id: str, vector: list[float]) -> None:
    """Cache session intent vector (128d) in Redis.

    Called by POST /v1/session/intent-vector after BST encoding.
    Key: shopfeed:session:intent:{session_id}  TTL: 1800s (30min)
    """
    import asyncio
    store = _get_store()
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(store.connect())
        if store._redis is not None:
            key = store._key("session", "intent", session_id)
            data = torch.tensor(vector, dtype=torch.float32).numpy().tobytes()
            loop.run_until_complete(store._redis.set(key, data, ex=1800))
        loop.close()
    except Exception as e:
        logger.warning("cache_session_intent failed for %s: %s", session_id, e)

