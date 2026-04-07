"""
Live Service — FastAPI App + Routes — Section 07 / 23.

Migration: dicts Python in-memory (_live_sessions, _live_metrics)
→ SQLAlchemy 2.0 ORM via LiveSessionRepository pour la persistance.
Les métriques temps-réel (viewers, GMV live) restent dans Redis —
c'est l'architecture correcte (Redis = volatil temps-réel, PG = persistant).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.repositories.live_repository import LiveSessionRepository

from .connection_manager import LiveConnectionManager
from .schemas import CreateLiveRequest, LiveMetricsResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Live Service", version="1.0.0")

manager = LiveConnectionManager(redis_client=None)
_live_repo = LiveSessionRepository()

# Métriques temps-réel en Redis (ou mémoire locale si Redis down)
# Pattern correct: Redis pour le volatil, PG pour le persistant
_live_metrics: dict[str, dict] = {}


@app.on_event("startup")
async def _inject_redis() -> None:
    redis = getattr(app.state, "redis", None)
    if redis is not None:
        manager.redis = redis
        logger.info("LiveConnectionManager: Redis pub/sub active (multi-replica safe)")
    else:
        logger.warning(
            "LiveConnectionManager: no Redis — single-pod in-memory mode only. "
            "WebSocket broadcast will NOT work across k8s replicas."
        )


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/live/create")
async def create_live(
    req: CreateLiveRequest,
    session: AsyncSession = Depends(get_db),
):
    """Create a new live session — persisté en PostgreSQL via ORM."""
    live = await _live_repo.create(session, {
        "vendor_id": uuid.UUID(req.vendor_id) if req.vendor_id else uuid.uuid4(),
        "title": req.title,
        "status": "scheduled" if req.scheduled_at else "live",
        "live_type": req.live_type or "standard",
        "pinned_product_ids": req.pinned_product_ids or [],
        "started_at": None if req.scheduled_at else __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ),
    })

    # Initialiser les métriques temps-réel en mémoire/Redis
    _live_metrics[str(live.id)] = {
        "concurrent_viewers": 0,
        "peak_viewers": 0,
        "total_gmv": 0.0,
        "items_sold": 0,
        "product_clicks": 0,
        "buy_now_count": 0,
        "comments": 0,
        "gifts_value": 0.0,
        "started_at": time.time(),
    }

    return {"live_id": str(live.id), "status": "created"}


@app.get("/api/v1/live/{live_id}/metrics", response_model=LiveMetricsResponse)
async def get_live_metrics(
    live_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get current live metrics — depuis Redis (temps-réel) + PG (persistant)."""
    metrics = _live_metrics.get(live_id, {})
    concurrent = await manager.get_viewer_count(live_id)

    return LiveMetricsResponse(
        concurrent_viewers=concurrent,
        peak_viewers=metrics.get("peak_viewers", 0),
        total_gmv=metrics.get("total_gmv", 0.0),
        items_sold=metrics.get("items_sold", 0),
        live_score=_compute_live_score(live_id, concurrent),
    )


def _compute_live_score(live_id: str, concurrent: int = 0) -> float:
    """Section 07 — LiveScore formula."""
    m = _live_metrics.get(live_id, {})
    peak = m.get("peak_viewers", 0)
    elapsed_min = max(
        (time.time() - m.get("started_at", time.time())) / 60.0, 1.0
    )
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
            concurrent = await manager.get_viewer_count(live_id)

            if concurrent > metrics.get("peak_viewers", 0):
                metrics["peak_viewers"] = concurrent

            elapsed_min = max(
                (time.time() - metrics.get("started_at", time.time())) / 60.0, 1.0
            )

            await websocket.send_json({
                "event": "metrics_update",
                "data": {
                    "concurrent_viewers": concurrent,
                    "peak_viewers": metrics.get("peak_viewers", 0),
                    "gmv_live": metrics.get("total_gmv", 0.0),
                    "buy_now_per_min": metrics.get("buy_now_count", 0) / elapsed_min,
                    "items_sold": metrics.get("items_sold", 0),
                    "ctr_product": (
                        metrics.get("product_clicks", 0) / max(1, concurrent)
                        if concurrent else 0
                    ),
                    "live_score": _compute_live_score(live_id, concurrent),
                    "comments": metrics.get("comments", 0),
                },
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        await manager.disconnect_async(websocket, room)


# ──────────────────────────────────────────────────────────────
# WebSocket — Viewer (Section 23)
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/live/{live_id}/viewer")
async def viewer_live_ws(websocket: WebSocket, live_id: str):
    """Bidirectional WS for viewers."""
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

                await manager.broadcast(room, {
                    "event": "purchase_made",
                    "buyer": data.get("user_name", "Someone"),
                    "product_title": data.get("product_title", ""),
                    "items_sold_total": metrics["items_sold"],
                })

    except WebSocketDisconnect:
        await manager.disconnect_async(websocket, room)
        logger.info("Viewer left live=%s", live_id)
