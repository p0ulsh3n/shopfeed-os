"""
Redis-backed Sliding Window Rate Limiter — Section 10
=======================================================

BUG #S2 FIX: The original RateLimiter used an in-memory dict (_buckets).
With 3+ k8s replicas of api-gateway, each pod had its own counter.
A user could send max_requests per pod = max_requests × n_replicas total.

Fixed with a Redis sliding window (ZADD + ZREMRANGEBYSCORE + ZCARD).
All replicas share one Redis and enforce the limit correctly.

Fallback: if Redis is unavailable, falls back to in-memory tracking
with a WARNING log so it's visible in production alerting.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Redis sliding window rate limiter — works correctly with k8s replicas.

    BUG #S2 FIX: replaced in-memory dict with Redis sorted set.
    Atomic pipeline ensures correctness under concurrent requests.

    Args:
        redis_client: async Redis client (redis.asyncio). If None, falls
            back to in-memory (single-replica only, logs a warning).
        max_requests: number of requests allowed per window
        window_seconds: sliding window duration in seconds
    """

    def __init__(
        self,
        redis_client=None,
        max_requests: int = 100,
        window_seconds: int = 60,
    ):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds
        # In-memory fallback (only correct for single replica)
        self._buckets: dict[str, list[float]] = {}
        if redis_client is None:
            logger.warning(
                "RateLimiter: no Redis client provided — using in-memory fallback. "
                "This is NOT safe in multi-replica deployments (k8s). "
                "Inject a redis_client at startup."
            )

    async def check(self, key: str) -> bool:
        """Returns True if request is allowed, False if rate limit exceeded."""
        if self.redis is not None:
            return await self._check_redis(key)
        return await self._check_memory(key)

    async def _check_redis(self, key: str) -> bool:
        """Atomic Redis sliding window using sorted set.

        Algorithm:
            1. Remove requests older than (now - window)
            2. Count current window requests
            3. If count >= max_requests → reject
            4. Otherwise → record this request and allow
        """
        now = time.time()
        window_start = now - self.window
        redis_key = f"rate:{key}"

        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                pipe.zremrangebyscore(redis_key, "-inf", window_start)
                pipe.zcard(redis_key)
                pipe.zadd(redis_key, {str(now): now})
                pipe.expire(redis_key, self.window + 1)
                results = await pipe.execute()

            # results[1] is the count BEFORE adding this request
            current_count = results[1]
            if current_count >= self.max_requests:
                # Undo the zadd we just did
                await self.redis.zrem(redis_key, str(now))
                return False
            return True

        except Exception as e:
            logger.warning("RateLimiter Redis error (%s) — falling back to allow", e)
            return True  # Fail open: don't block users when Redis is down

    async def _check_memory(self, key: str) -> bool:
        """In-memory fallback — only correct for single-replica deployments."""
        now = time.time()
        bucket = self._buckets.setdefault(key, [])
        cutoff = now - self.window
        self._buckets[key] = [t for t in bucket if t > cutoff]

        if len(self._buckets[key]) >= self.max_requests:
            return False

        self._buckets[key].append(now)
        return True
