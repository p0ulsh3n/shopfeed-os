"""Rate Limiter — Token Bucket — Section 10."""

from __future__ import annotations

import time


class RateLimiter:
    """Token bucket rate limiter — Section 10."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[str, list[float]] = {}

    async def check(self, key: str) -> bool:
        """Returns True if request is allowed."""
        now = time.time()
        bucket = self._buckets.setdefault(key, [])

        # Remove expired entries
        cutoff = now - self.window
        self._buckets[key] = [t for t in bucket if t > cutoff]

        if len(self._buckets[key]) >= self.max_requests:
            return False

        self._buckets[key].append(now)
        return True
