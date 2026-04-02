"""
Zeigarnik Collection Tracker (t.md §3)
========================================
Exploits the Zeigarnik effect: incomplete tasks create persistent
mental tension that drives return visits. When a user owns 3/7 pieces
of a collection, their brain "can't let go" until completion.

This module is ADDITIVE: it produces boost scores that the re-ranking
stage multiplies alongside existing scores — it never replaces them.

Key class:
    ZeigarnikTracker — tracks per-user collection progress and
    produces a boost score for candidates that belong to incomplete sets.
"""

from __future__ import annotations

import math
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ZeigarnikTracker:
    """Tracks user collection progress for Zeigarnik-effect boosting.

    A "collection" is any logical grouping of products:
        - Capsule collections (vendor-curated)
        - Category sets ("complete the look")
        - Brand product lines
        - Algorithmic "story arcs" (auto-generated)

    Data is stored in Redis (or in-memory fallback).

    Usage:
        tracker = ZeigarnikTracker(redis_client)
        await tracker.record_purchase(user_id, collection_id, item_id)
        boost = tracker.compute_zeigarnik_boost(
            completion_ratio=3/7
        )  # → 0.25 (peak zone)
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        # In-memory fallback: {user_id: {collection_id: {items_total, items_owned}}}
        self._collections: dict[str, dict[str, dict[str, Any]]] = {}

    def _key(self, user_id: str) -> str:
        return f"zeigarnik:{user_id}"

    async def record_purchase(
        self,
        user_id: str,
        collection_id: str,
        item_id: str,
        collection_total: int = 7,
    ) -> None:
        """Record that a user purchased an item from a collection."""
        if self.redis:
            field = f"{collection_id}:owned"
            total_field = f"{collection_id}:total"
            pipe = self.redis.pipeline()
            pipe.sadd(f"{self._key(user_id)}:{collection_id}:items", item_id)
            pipe.hset(self._key(user_id), total_field, str(collection_total))
            await pipe.execute()
            # Update owned count from set cardinality
            owned = await self.redis.scard(f"{self._key(user_id)}:{collection_id}:items")
            await self.redis.hset(self._key(user_id), field, str(owned))
        else:
            user_data = self._collections.setdefault(user_id, {})
            coll = user_data.setdefault(collection_id, {
                "items_total": collection_total,
                "items_owned": set(),
            })
            coll["items_owned"].add(item_id)

    async def get_completion_ratio(self, user_id: str, collection_id: str) -> float:
        """Get completion ratio for a specific collection. Returns 0.0 if unknown."""
        if self.redis:
            total = await self.redis.hget(self._key(user_id), f"{collection_id}:total")
            owned = await self.redis.hget(self._key(user_id), f"{collection_id}:owned")
            if total and owned:
                t = int(total)
                o = int(owned)
                return o / max(t, 1) if t > 0 else 0.0
            return 0.0
        else:
            user_data = self._collections.get(user_id, {})
            coll = user_data.get(collection_id)
            if coll:
                total = coll.get("items_total", 1)
                owned = len(coll.get("items_owned", set()))
                return owned / max(total, 1)
            return 0.0

    async def get_incomplete_collections(self, user_id: str) -> list[dict[str, Any]]:
        """Get all collections where 0 < ratio < 1.0."""
        result = []
        if self.redis:
            data = await self.redis.hgetall(self._key(user_id))
            collections: dict[str, dict] = {}
            for field, value in data.items():
                parts = field.split(":")
                if len(parts) == 2:
                    coll_id, attr = parts
                    collections.setdefault(coll_id, {})[attr] = value
            for coll_id, attrs in collections.items():
                total = int(attrs.get("total", 0))
                owned = int(attrs.get("owned", 0))
                if total > 0 and 0 < owned < total:
                    result.append({
                        "collection_id": coll_id,
                        "items_total": total,
                        "items_owned": owned,
                        "ratio": owned / total,
                    })
        else:
            for coll_id, coll in self._collections.get(user_id, {}).items():
                total = coll.get("items_total", 1)
                owned = len(coll.get("items_owned", set()))
                if total > 0 and 0 < owned < total:
                    result.append({
                        "collection_id": coll_id,
                        "items_total": total,
                        "items_owned": owned,
                        "ratio": owned / total,
                    })
        return result

    @staticmethod
    def compute_zeigarnik_boost(completion_ratio: float) -> float:
        """Compute Zeigarnik boost from collection completion ratio.

        The boost peaks at 40-70% completion (the "tension zone"):
            - 0% complete → 0.0 (no investment, no tension)
            - 40-70% complete → ~0.25-0.30 (peak tension)
            - 100% complete → 0.0 (satisfaction, tension released)

        This follows a bell curve centered around 0.55 completion.

        Returns: float in [0.0, 0.3]
        """
        if completion_ratio <= 0.0 or completion_ratio >= 1.0:
            return 0.0

        # Gaussian-like curve peaking at ratio=0.55
        center = 0.55
        spread = 0.25
        raw = math.exp(-((completion_ratio - center) ** 2) / (2 * spread ** 2))

        # Scale to [0, 0.3] max boost
        return round(raw * 0.3, 4)
