"""
Real-Time Counters — Redis Atomic Operations (Section 06 / 13)
================================================================
"Chaque view, like, purchase met à jour score_cvr, score_watch_time,
score_engagement atomiquement via Redis. Batch sync vers PostgreSQL
toutes les 5 minutes."

This module provides TikTok-style real-time counters:
  - Views, likes, shares, add_to_cart, purchases → atomic INCR
  - Watch time / engagement scores → atomic HINCRBYFLOAT
  - Live viewer counts → INCR / DECR on join/leave
  - Batch sync to PostgreSQL every 5 minutes for persistence

Redis is the "source of truth" for real-time metrics. PostgreSQL
stores the persisted snapshot every 5 minutes.

Key schema:
  video:{id}:counters     → HASH {views, likes, shares, cart, purchases}
  video:{id}:scores       → HASH {score_cvr, score_watch_time, score_engagement}
  live:{id}:viewers       → INT (INCR/DECR on join/leave)
  live:{id}:metrics       → HASH {gmv, buy_now_count, peak_viewers}
  product:{id}:counters   → HASH {views, cart, purchases}
  seller:{id}:counters    → HASH {daily_gmv, daily_orders}
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Redis Client Wrapper
# ═══════════════════════════════════════════════════════════════

class RedisCounterClient:
    """Async Redis client for real-time atomic counters.

    Uses hiredis parser for maximum throughput. All operations
    are O(1) and return in <1ms under normal load.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._redis = None
        self._redis_url = redis_url

    async def connect(self) -> None:
        if self._redis is not None:
            return
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url,
                decode_responses=True,  # Return strings, not bytes
                max_connections=100,
            )
            await self._redis.ping()
            logger.info("RedisCounterClient connected: %s", self._redis_url)
        except ImportError:
            logger.error("redis package not installed — pip install redis[hiredis]")
        except Exception as e:
            logger.error("Redis connection failed: %s", e)

    @property
    def connected(self) -> bool:
        return self._redis is not None


# ═══════════════════════════════════════════════════════════════
# 1. Video / Content Counters (TikTok-style)
# ═══════════════════════════════════════════════════════════════

class ContentCounters:
    """Atomic counters for feed videos and content items.

    Every view, like, share triggers an atomic Redis INCR.
    This is how TikTok makes view counts update in real-time:
    the counter lives in Redis (in-memory, <1ms), not in PostgreSQL.

    Usage:
        counters = ContentCounters(redis_client)
        await counters.increment_view("video_123")
        await counters.increment_like("video_123")
        stats = await counters.get_all("video_123")
        # → {"views": 1502, "likes": 340, "shares": 12, ...}
    """

    COUNTER_FIELDS = ("views", "likes", "shares", "cart", "purchases", "comments")

    def __init__(self, client: RedisCounterClient, key_prefix: str = "video"):
        self._client = client
        self._prefix = key_prefix

    def _key(self, content_id: str) -> str:
        return f"{self._prefix}:{content_id}:counters"

    def _score_key(self, content_id: str) -> str:
        return f"{self._prefix}:{content_id}:scores"

    # ── Atomic increments ────────────────────────────────────

    async def increment_view(self, content_id: str) -> int:
        """Atomic +1 view. Returns new count. <1ms."""
        r = self._client._redis
        if r is None:
            return 0
        return await r.hincrby(self._key(content_id), "views", 1)

    async def increment_like(self, content_id: str) -> int:
        r = self._client._redis
        if r is None:
            return 0
        return await r.hincrby(self._key(content_id), "likes", 1)

    async def increment_share(self, content_id: str) -> int:
        r = self._client._redis
        if r is None:
            return 0
        return await r.hincrby(self._key(content_id), "shares", 1)

    async def increment_add_to_cart(self, content_id: str) -> int:
        r = self._client._redis
        if r is None:
            return 0
        return await r.hincrby(self._key(content_id), "cart", 1)

    async def increment_purchase(self, content_id: str, gmv: float = 0.0) -> int:
        """Atomic purchase increment + GMV attribution."""
        r = self._client._redis
        if r is None:
            return 0
        pipe = r.pipeline()
        pipe.hincrby(self._key(content_id), "purchases", 1)
        if gmv > 0:
            pipe.hincrbyfloat(self._key(content_id), "gmv_attributed", gmv)
        results = await pipe.execute()
        return results[0]

    async def increment_comment(self, content_id: str) -> int:
        r = self._client._redis
        if r is None:
            return 0
        return await r.hincrby(self._key(content_id), "comments", 1)

    # ── Score updates (Section 06) ───────────────────────────

    async def update_scores(
        self,
        content_id: str,
        score_cvr: float | None = None,
        score_watch_time: float | None = None,
        score_engagement: float | None = None,
    ) -> None:
        """Atomically update scoring fields.

        "Chaque view, like, purchase met à jour score_cvr,
        score_watch_time, score_engagement atomiquement via Redis."
        """
        r = self._client._redis
        if r is None:
            return
        key = self._score_key(content_id)
        pipe = r.pipeline()
        if score_cvr is not None:
            pipe.hset(key, "score_cvr", str(score_cvr))
        if score_watch_time is not None:
            pipe.hset(key, "score_watch_time", str(score_watch_time))
        if score_engagement is not None:
            pipe.hset(key, "score_engagement", str(score_engagement))
        await pipe.execute()

    # ── Reads ────────────────────────────────────────────────

    async def get_all(self, content_id: str) -> dict[str, int]:
        """Get all counters for a content item. <1ms."""
        r = self._client._redis
        if r is None:
            return {}
        raw = await r.hgetall(self._key(content_id))
        return {k: int(float(v)) for k, v in raw.items()} if raw else {}

    async def get_scores(self, content_id: str) -> dict[str, float]:
        """Get scoring fields for a content item."""
        r = self._client._redis
        if r is None:
            return {}
        raw = await r.hgetall(self._score_key(content_id))
        return {k: float(v) for k, v in raw.items()} if raw else {}

    async def get_view_count(self, content_id: str) -> int:
        """Get just the view count. <0.5ms."""
        r = self._client._redis
        if r is None:
            return 0
        val = await r.hget(self._key(content_id), "views")
        return int(val) if val else 0


# ═══════════════════════════════════════════════════════════════
# 2. Live Viewer Counters (WebSocket-driven)
# ═══════════════════════════════════════════════════════════════

class LiveCounters:
    """Real-time live session metrics — Redis-backed.

    "Index Redis : {sellerId → isLive, viewerCount, startedAt}.
     Mis à jour en temps réel."

    Unlike content counters (which accumulate), live viewer counts
    go up AND down as viewers join/leave.

    LiveScore (Section 07) is recomputed every 60s:
        LiveScore = CV × 1.0 + PV × 0.5 + GMV_rate × 2.0
                  + Buy_now_5min × 1.5 + Engagement × 0.3
    """

    def __init__(self, client: RedisCounterClient):
        self._client = client

    def _viewers_key(self, live_id: str) -> str:
        return f"live:{live_id}:viewers"

    def _metrics_key(self, live_id: str) -> str:
        return f"live:{live_id}:metrics"

    def _seller_key(self, seller_id: str) -> str:
        return f"seller:{seller_id}:live"

    async def viewer_joined(self, live_id: str, seller_id: str) -> int:
        """Atomic INCR when viewer joins live. Returns new count."""
        r = self._client._redis
        if r is None:
            return 0
        pipe = r.pipeline()
        pipe.incr(self._viewers_key(live_id))
        # Track peak
        pipe.hget(self._metrics_key(live_id), "peak_viewers")
        results = await pipe.execute()
        current = results[0]

        # Update peak if new high
        peak = int(results[1] or 0)
        if current > peak:
            await r.hset(self._metrics_key(live_id), "peak_viewers", str(current))

        # Update seller live index
        await r.hset(self._seller_key(seller_id), mapping={
            "is_live": "1",
            "viewer_count": str(current),
            "live_id": live_id,
        })

        return current

    async def viewer_left(self, live_id: str, seller_id: str) -> int:
        """Atomic DECR when viewer leaves. Cannot go below 0."""
        r = self._client._redis
        if r is None:
            return 0
        count = await r.decr(self._viewers_key(live_id))
        if count < 0:
            await r.set(self._viewers_key(live_id), "0")
            count = 0

        # Update seller index
        await r.hset(self._seller_key(seller_id), "viewer_count", str(count))
        return count

    async def get_viewer_count(self, live_id: str) -> int:
        """Current concurrent viewers. <0.5ms."""
        r = self._client._redis
        if r is None:
            return 0
        val = await r.get(self._viewers_key(live_id))
        return int(val) if val else 0

    async def record_buy_now(self, live_id: str, amount: float) -> None:
        """Atomic buy-now event during live."""
        r = self._client._redis
        if r is None:
            return
        pipe = r.pipeline()
        pipe.hincrby(self._metrics_key(live_id), "buy_now_count", 1)
        pipe.hincrbyfloat(self._metrics_key(live_id), "gmv", amount)
        await pipe.execute()

    async def get_live_metrics(self, live_id: str) -> dict[str, Any]:
        """Get all live metrics. <1ms."""
        r = self._client._redis
        if r is None:
            return {}
        raw = await r.hgetall(self._metrics_key(live_id))
        viewers = await self.get_viewer_count(live_id)
        return {
            "concurrent_viewers": viewers,
            "peak_viewers": int(raw.get("peak_viewers", 0)),
            "buy_now_count": int(raw.get("buy_now_count", 0)),
            "gmv": float(raw.get("gmv", 0.0)),
        }

    async def end_live(self, live_id: str, seller_id: str) -> None:
        """Clean up when live ends."""
        r = self._client._redis
        if r is None:
            return
        await r.hset(self._seller_key(seller_id), "is_live", "0")
        # Keep metrics for 24h for analytics
        await r.expire(self._metrics_key(live_id), 86400)
        await r.expire(self._viewers_key(live_id), 86400)


# ═══════════════════════════════════════════════════════════════
# 3. Product / Seller Counters
# ═══════════════════════════════════════════════════════════════

class ProductCounters:
    """Per-product atomic counters for marketplace items."""

    def __init__(self, client: RedisCounterClient):
        self._client = client

    def _key(self, product_id: str) -> str:
        return f"product:{product_id}:counters"

    async def increment_view(self, product_id: str) -> int:
        r = self._client._redis
        if r is None:
            return 0
        return await r.hincrby(self._key(product_id), "views", 1)

    async def increment_cart(self, product_id: str) -> int:
        r = self._client._redis
        if r is None:
            return 0
        return await r.hincrby(self._key(product_id), "cart", 1)

    async def increment_purchase(self, product_id: str, amount: float) -> int:
        r = self._client._redis
        if r is None:
            return 0
        pipe = self._client._redis.pipeline()
        pipe.hincrby(self._key(product_id), "purchases", 1)
        pipe.hincrbyfloat(self._key(product_id), "gmv", amount)
        results = await pipe.execute()
        return results[0]

    async def get_all(self, product_id: str) -> dict[str, int]:
        r = self._client._redis
        if r is None:
            return {}
        raw = await r.hgetall(self._key(product_id))
        return {k: int(float(v)) for k, v in raw.items()} if raw else {}


# ═══════════════════════════════════════════════════════════════
# 4. Batch Sync — Redis → PostgreSQL (every 5 min)
# ═══════════════════════════════════════════════════════════════

class CounterSyncService:
    """Periodically flushes Redis counters to PostgreSQL.

    "Batch sync vers PostgreSQL toutes les 5 minutes."

    Redis is the real-time source. PostgreSQL is persistence.
    If Redis crashes, we lose at most 5 minutes of counter increments,
    but the service recovers from the last PostgreSQL snapshot.

    In production, run this as a background task in the feed service
    or as a dedicated Flink job.
    """

    def __init__(
        self,
        counter_client: RedisCounterClient,
        sync_interval_s: int = 300,  # 5 minutes
    ):
        self._client = counter_client
        self._interval = sync_interval_s
        self._running = False
        self._synced_count = 0

    async def run(self, db_pool=None) -> None:
        """Background loop: scan Redis counters → upsert into PostgreSQL."""
        self._running = True
        logger.info("CounterSyncService started — interval=%ds", self._interval)

        while self._running:
            try:
                await self._sync_round(db_pool)
            except Exception as e:
                logger.error("Sync error: %s", e)
            await asyncio.sleep(self._interval)

    async def _sync_round(self, db_pool=None) -> None:
        """One sync round: read counters from Redis, write to PostgreSQL."""
        r = self._client._redis
        if r is None:
            return

        # Scan for all video counter keys
        cursor = 0
        batch_count = 0
        while True:
            cursor, keys = await r.scan(cursor, match="video:*:counters", count=500)
            for key in keys:
                content_id = key.split(":")[1]
                counters = await r.hgetall(key)
                scores_key = f"video:{content_id}:scores"
                scores = await r.hgetall(scores_key)

                if db_pool is not None:
                    await self._upsert_to_db(db_pool, content_id, counters, scores)
                batch_count += 1

            if cursor == 0:
                break

        self._synced_count += batch_count
        if batch_count > 0:
            logger.info(
                "Redis → PostgreSQL sync: %d items updated (total: %d)",
                batch_count, self._synced_count,
            )

    async def _upsert_to_db(
        self,
        db_pool,  # gardé pour compatibilité signature, non utilisé
        content_id: str,
        counters: dict[str, str],
        scores: dict[str, str],
    ) -> None:
        """
        Flush compteurs Redis → table feed_videos via SQLAlchemy ORM.

        MIGRATION SÉCURITÉ:
        - AVANT: asyncpg conn.execute("UPDATE feed_videos SET ... WHERE id = $1", ...)
          → SQL brut, pool asyncpg directe, pas de session SQLAlchemy
        - APRÈS: AnalyticsRepository.upsert_feed_video_counters() via pg_insert
          ON CONFLICT DO UPDATE — 100% ORM, zéro SQL brut
        """
        from shared.db.session import AsyncSessionLocal
        from shared.repositories.analytics_repository import AnalyticsRepository

        _repo = AnalyticsRepository()
        data = {
            "total_views": int(counters.get("views", 0)),
            "total_likes": int(counters.get("likes", 0)),
            "total_shares": int(counters.get("shares", 0)),
            "total_add_to_cart": int(counters.get("cart", 0)),
            "total_purchases": int(counters.get("purchases", 0)),
            "gmv_attributed": float(counters.get("gmv_attributed", 0)),
            "score_cvr": float(scores.get("score_cvr", 0)),
            "score_watch_time": float(scores.get("score_watch_time", 0)),
            "score_engagement": float(scores.get("score_engagement", 0)),
        }
        try:
            async with AsyncSessionLocal() as session:
                await _repo.upsert_feed_video_counters(session, content_id, data)
                await session.commit()
        except Exception as exc:
            logger.warning("ORM upsert failed for %s: %s", content_id, exc)

    def stop(self) -> None:
        self._running = False
