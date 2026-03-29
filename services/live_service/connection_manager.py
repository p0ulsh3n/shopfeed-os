"""
WebSocket Connection Manager for Live Rooms — Section 07
=========================================================

BUG #S5 FIX: The original LiveConnectionManager stored WebSocket objects
in a local in-memory dict (_rooms). With k8s deploying 3+ replicas of
live-service, a vendor connected to pod-1 and a viewer on pod-2 would
never share the same room dict — broadcast() would send to nobody.

Fixed with Redis Pub/Sub as the cross-replica message bus:
- Each pod subscribes to a Redis channel per live_id
- broadcast() publishes to Redis, all pods receive and forward to their
  local WebSocket connections
- Local connections are still tracked per-pod for direct delivery
- Viewer counts are stored in Redis (INCR/DECR) so they sum correctly
  across all replicas

Fallback: Without Redis (dev/local), falls back to in-memory (single pod only).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class LiveConnectionManager:
    """Cross-replica WebSocket manager via Redis Pub/Sub.

    BUG #S5 FIX: replaced pure in-memory dict with Redis Pub/Sub for
    multi-replica correctness. Each pod still maintains its local
    WebSocket registry, but messages broadcast via Redis reach all pods.

    Args:
        redis_client: async Redis client (redis.asyncio). If None,
            falls back to single-pod in-memory mode.
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        # Local WebSocket registry (connections on THIS pod)
        self._rooms: dict[str, set[WebSocket]] = defaultdict(set)
        # Redis Pub/Sub listeners per room (one asyncio task per room)
        self._pubsub_tasks: dict[str, asyncio.Task] = {}

    async def connect(self, ws: WebSocket, room: str) -> None:
        """Accept WebSocket and subscribe to Redis channel for cross-pod messages."""
        await ws.accept()
        self._rooms[room].add(ws)

        # Increment viewer count in Redis (shared across all replicas)
        if self.redis and "viewer" in room:
            live_id = room.split("_", 1)[1] if "_" in room else room
            await self.redis.incr(f"live:viewers:{live_id}")

        # Start a Redis sub listener for this room (one per room per pod)
        if self.redis and room not in self._pubsub_tasks:
            task = asyncio.create_task(self._redis_listener(room))
            self._pubsub_tasks[room] = task
            logger.debug("Started Redis pub/sub listener for room: %s", room)

    def disconnect(self, ws: WebSocket, room: str) -> None:
        """Remove WebSocket from local registry. Viewer count decremented async."""
        self._rooms[room].discard(ws)

    async def disconnect_async(self, ws: WebSocket, room: str) -> None:
        """Full async disconnect — decrements Redis viewer count."""
        self._rooms[room].discard(ws)
        if self.redis and "viewer" in room:
            live_id = room.split("_", 1)[1] if "_" in room else room
            await self.redis.decr(f"live:viewers:{live_id}")

        # Cancel listener if no local connections remain
        if not self._rooms[room] and room in self._pubsub_tasks:
            self._pubsub_tasks[room].cancel()
            del self._pubsub_tasks[room]

    async def broadcast(self, room: str, data: dict) -> None:
        """Publish to Redis (reaches ALL pods) + deliver to local connections.

        BUG #S5 FIX: Without Redis, broadcast only reached local connections.
        Now broadcasts via Redis channel so all replicas forward the message.
        """
        payload = json.dumps(data)

        if self.redis:
            try:
                # Publish to Redis channel — all pods subscribed to this room will deliver
                await self.redis.publish(f"live:{room}", payload)
            except Exception as e:
                logger.warning("Redis publish failed (%s) — delivering to local only", e)
                await self._deliver_local(room, data)
        else:
            # No Redis: local-only delivery (single pod mode)
            await self._deliver_local(room, data)

    async def _deliver_local(self, room: str, data: dict) -> None:
        """Deliver to WebSocket connections on THIS pod."""
        dead: list[WebSocket] = []
        for ws in list(self._rooms.get(room, set())):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._rooms[room].discard(ws)

    async def _redis_listener(self, room: str) -> None:
        """Subscribe to Redis Pub/Sub channel and forward messages to local WebSockets.

        Runs as a background asyncio task per room per pod.
        Cancelled when the last local viewer disconnects.
        """
        if self.redis is None:
            return
        try:
            async with self.redis.pubsub() as pubsub:
                await pubsub.subscribe(f"live:{room}")
                logger.debug("Redis sub active: channel=live:%s", room)
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            await self._deliver_local(room, data)
                        except Exception as e:
                            logger.warning("Redis message decode error: %s", e)
        except asyncio.CancelledError:
            logger.debug("Redis pub/sub listener cancelled for room: %s", room)
        except Exception as e:
            logger.error("Redis pub/sub listener error for room %s: %s", room, e)

    async def get_viewer_count(self, live_id: str) -> int:
        """Get viewer count across ALL replicas from Redis."""
        if self.redis:
            try:
                val = await self.redis.get(f"live:viewers:{live_id}")
                return int(val) if val else 0
            except Exception:
                pass
        # Fallback: count local connections only
        return len(self._rooms.get(f"viewer_{live_id}", set()))
