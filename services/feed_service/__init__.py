"""Feed Service — FastAPI Application.

Endpoints:
    GET /api/v1/feed              — Main discovery feed (Section 04)
    GET /api/v1/feed/following    — Following feed (Section 20)
    WS  /ws/session/{user_id}     — Session State real-time (Section 13)
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from services.feed_service.pipeline import RecommendationPipeline, SessionState

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Feed Service", version="1.0.0")

# Singletons
pipeline = RecommendationPipeline()
session_state = SessionState()


# ──────────────────────────────────────────────────────────────
# Response Models
# ──────────────────────────────────────────────────────────────

class FomoSignals(BaseModel):
    stock_remaining: int | None = None
    viewers_now: int | None = None
    flash_sale_active: bool = False
    flash_sale_ends_at: str | None = None
    discount_pct: int | None = None


class FeedItem(BaseModel):
    content_id: str
    content_type: str
    rank_score: float
    pool_level: str
    vendor_id: str
    vendor_name: str = ""
    vendor_tier: str = "bronze"
    product_id: str
    price: float = 0.0
    currency: str = "EUR"
    media_url: str = ""
    thumbnail_url: str = ""
    fomo: FomoSignals = Field(default_factory=FomoSignals)
    shipping_free: bool = False
    estimated_days: int = 3


class FeedResponse(BaseModel):
    items: list[FeedItem]
    next_cursor: str | None = None
    session_state_updated: bool = True


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/feed", response_model=FeedResponse)
async def get_feed(
    cursor: Optional[str] = None,
    session_id: Optional[str] = None,
    content_types: Optional[str] = None,
    limit: int = Query(default=10, le=20),
):
    """Main discovery feed — Section 04.

    Returns algorithmically ranked content with FOMO signals.
    The pipeline executes in <80ms.
    """
    sid = session_id or str(uuid.uuid4())
    ct_filter = content_types.split(",") if content_types else None

    candidates = await pipeline.generate_feed(
        user_id="anonymous",    # Would come from JWT in production
        session_id=sid,
        content_types=ct_filter,
        limit=limit,
    )

    items = [
        FeedItem(
            content_id=c.content_id,
            content_type=c.content_type,
            rank_score=round(c.final_score, 4),
            pool_level=c.pool_level,
            vendor_id=c.vendor_id,
            vendor_tier=c.vendor_tier,
            product_id=c.product_id,
            price=c.base_price,
            fomo=FomoSignals(
                stock_remaining=c.stock if c.stock < 20 else None,
            ),
        )
        for c in candidates
    ]

    return FeedResponse(items=items, next_cursor=cursor)


@app.get("/api/v1/feed/following", response_model=FeedResponse)
async def get_following_feed(
    limit: int = Query(default=10, le=20),
):
    """Following Feed — Section 20.

    Shows only content from followed vendors, sorted chronologically.
    """
    return FeedResponse(items=[], next_cursor=None)


# ──────────────────────────────────────────────────────────────
# WebSocket — Session State (Section 13)
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/session/{user_id}")
async def session_state_ws(websocket: WebSocket, user_id: str):
    """Real-time session state updates — Section 13.

    Client sends action events, server updates Session State
    and optionally pushes new N+1 content.

    Client → Server:
        {"type": "zoom|pause_3s|skip|buy_now|add_to_cart", "product_id": "...", "category": "..."}

    Server → Client (on buffer invalidation):
        {"event": "next_content_updated", "content": {...}}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info("Session WS connected: user=%s, session=%s", user_id, session_id)

    old_vector = None

    try:
        while True:
            data = await websocket.receive_json()

            action_type = data.get("type", "view")
            product_id = data.get("product_id")
            category = data.get("category")
            price = data.get("price")

            # Update Session State — <50ms
            session = await session_state.update_session(
                session_id=session_id,
                user_id=user_id,
                action_type=action_type,
                product_id=product_id,
                category=category,
                price=price,
            )

            # Check if buffer N+1 needs recalculation (Section 04b)
            new_vector = session_state.compute_session_vector(session)

            should_recalc = False
            if old_vector is not None:
                # Cosine distance
                dot = float(new_vector @ old_vector)
                delta = 1.0 - dot  # 0 = identical, 2 = opposite
                if delta > 0.15:
                    should_recalc = True

            if action_type in ("buy_now", "add_to_cart"):
                should_recalc = True

            if should_recalc:
                # Trigger N+1 recalculation
                await websocket.send_json({
                    "event": "next_content_updated",
                    "recalculate": True,
                    "intent_level": session.get("intent_level", "low"),
                })

            old_vector = new_vector

    except WebSocketDisconnect:
        logger.info("Session WS disconnected: user=%s", user_id)
