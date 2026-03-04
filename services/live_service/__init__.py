"""Live Service — WebSocket Real-Time Commerce — Section 07.

Handles:
    - Live session lifecycle (create, start, end)
    - WebSocket metrics push to vendors (every 5s)
    - WebSocket updates to viewers (stock, flash sale, viewers count)
    - Live scoring (updated every 60s)
    - Flash Sale management
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any
from uuid import UUID

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from shared.models.live import LiveSession, LiveStatus, LiveType

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Live Service", version="1.0.0")


# ──────────────────────────────────────────────────────────────
# Connection Manager
# ──────────────────────────────────────────────────────────────

class LiveConnectionManager:
    """Manages WebSocket connections for live rooms."""

    def __init__(self):
        self._rooms: dict[str, set[WebSocket]] = defaultdict(set)
        self._viewer_counts: dict[str, int] = defaultdict(int)

    async def connect(self, ws: WebSocket, room: str) -> None:
        await ws.accept()
        self._rooms[room].add(ws)
        if "viewer" in room:
            live_id = room.split("_")[1] if "_" in room else room
            self._viewer_counts[live_id] += 1

    def disconnect(self, ws: WebSocket, room: str) -> None:
        self._rooms[room].discard(ws)
        if "viewer" in room:
            live_id = room.split("_")[1] if "_" in room else room
            self._viewer_counts[live_id] = max(0, self._viewer_counts[live_id] - 1)

    async def broadcast(self, room: str, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._rooms.get(room, set()):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._rooms[room].discard(ws)

    def get_viewer_count(self, live_id: str) -> int:
        return self._viewer_counts.get(live_id, 0)


manager = LiveConnectionManager()

# In-memory live sessions (would be PostgreSQL in production)
_live_sessions: dict[str, dict] = {}
_live_metrics: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

class CreateLiveRequest(BaseModel):
    vendor_id: str
    title: str
    live_type: str = "instant"
    scheduled_at: str | None = None
    pinned_product_ids: list[str] = []


class LiveMetricsResponse(BaseModel):
    concurrent_viewers: int = 0
    peak_viewers: int = 0
    total_gmv: float = 0.0
    buy_now_per_min: float = 0.0
    ctr_product: float = 0.0
    live_score: float = 0.0
    items_sold: int = 0
    co_rate: float = 0.0               # Click-to-Order rate


@app.post("/api/v1/live/create")
async def create_live(req: CreateLiveRequest):
    """Create a new live session."""
    import uuid
    live_id = str(uuid.uuid4())
    _live_sessions[live_id] = {
        "id": live_id,
        "vendor_id": req.vendor_id,
        "title": req.title,
        "status": "scheduled" if req.scheduled_at else "live",
        "live_type": req.live_type,
        "pinned_product_ids": req.pinned_product_ids,
        "started_at": time.time(),
    }
    _live_metrics[live_id] = {
        "concurrent_viewers": 0,
        "peak_viewers": 0,
        "total_gmv": 0.0,
        "items_sold": 0,
        "product_clicks": 0,
        "buy_now_count": 0,
        "comments": 0,
        "gifts_value": 0.0,
    }
    return {"live_id": live_id, "status": "created"}


@app.get("/api/v1/live/{live_id}/metrics", response_model=LiveMetricsResponse)
async def get_live_metrics(live_id: str):
    """Get current live metrics."""
    metrics = _live_metrics.get(live_id, {})
    concurrent = manager.get_viewer_count(live_id)

    return LiveMetricsResponse(
        concurrent_viewers=concurrent,
        peak_viewers=metrics.get("peak_viewers", 0),
        total_gmv=metrics.get("total_gmv", 0.0),
        items_sold=metrics.get("items_sold", 0),
        live_score=_compute_live_score(live_id),
    )


def _compute_live_score(live_id: str) -> float:
    """Section 07 — LiveScore formula, updated every 60s."""
    m = _live_metrics.get(live_id, {})
    concurrent = manager.get_viewer_count(live_id)
    peak = m.get("peak_viewers", 0)

    elapsed_min = max((time.time() - _live_sessions.get(live_id, {}).get("started_at", time.time())) / 60.0, 1.0)
    purchase_rate = m.get("buy_now_count", 0) / elapsed_min
    comment_rate = m.get("comments", 0) / elapsed_min

    return (
        concurrent * 1.0
        + peak * 0.5
        + purchase_rate * 100.0
        + m.get("gifts_value", 0.0) * 2.0
        + comment_rate * 10.0
    )


# ──────────────────────────────────────────────────────────────
# WebSocket — Vendor Dashboard (Section 23)
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/live/{live_id}/vendor")
async def vendor_live_ws(websocket: WebSocket, live_id: str):
    """Push real-time metrics to vendor every 5 seconds."""
    room = f"vendor_{live_id}"
    await manager.connect(websocket, room)
    logger.info("Vendor WS connected: live=%s", live_id)

    try:
        while True:
            metrics = _live_metrics.get(live_id, {})
            concurrent = manager.get_viewer_count(live_id)

            # Update peak
            if concurrent > metrics.get("peak_viewers", 0):
                metrics["peak_viewers"] = concurrent

            await websocket.send_json({
                "event": "metrics_update",
                "data": {
                    "concurrent_viewers": concurrent,
                    "peak_viewers": metrics.get("peak_viewers", 0),
                    "gmv_live": metrics.get("total_gmv", 0.0),
                    "buy_now_per_min": metrics.get("buy_now_count", 0) / max(1, (time.time() - _live_sessions.get(live_id, {}).get("started_at", time.time())) / 60.0),
                    "items_sold": metrics.get("items_sold", 0),
                    "ctr_product": (metrics.get("product_clicks", 0) / max(1, concurrent)) if concurrent else 0,
                    "live_score": _compute_live_score(live_id),
                    "comments": metrics.get("comments", 0),
                },
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(websocket, room)


# ──────────────────────────────────────────────────────────────
# WebSocket — Viewer (Section 23)
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/live/{live_id}/viewer")
async def viewer_live_ws(websocket: WebSocket, live_id: str):
    """Bidirectional WS for viewers: receive actions, push updates."""
    room = f"viewer_{live_id}"
    await manager.connect(websocket, room)
    logger.info("Viewer joined live=%s", live_id)

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "")

            metrics = _live_metrics.setdefault(live_id, {})

            if action == "comment":
                metrics["comments"] = metrics.get("comments", 0) + 1
                # Broadcast to all viewers
                await manager.broadcast(room, {
                    "event": "new_comment",
                    "user": data.get("user_name", "Viewer"),
                    "text": data.get("text", ""),
                })

            elif action == "gift":
                value = data.get("value", 0)
                metrics["gifts_value"] = metrics.get("gifts_value", 0.0) + value
                await manager.broadcast(room, {
                    "event": "gift_received",
                    "value": value,
                    "from": data.get("user_name", "Anonymous"),
                })

            elif action == "product_click":
                metrics["product_clicks"] = metrics.get("product_clicks", 0) + 1

            elif action == "buy_now":
                metrics["buy_now_count"] = metrics.get("buy_now_count", 0) + 1
                metrics["items_sold"] = metrics.get("items_sold", 0) + 1
                amount = data.get("amount", 0)
                metrics["total_gmv"] = metrics.get("total_gmv", 0.0) + amount

                # Broadcast purchase to all viewers (social proof)
                await manager.broadcast(room, {
                    "event": "purchase_made",
                    "buyer": data.get("user_name", "Someone"),
                    "product_title": data.get("product_title", ""),
                    "items_sold_total": metrics["items_sold"],
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, room)
        logger.info("Viewer left live=%s", live_id)
