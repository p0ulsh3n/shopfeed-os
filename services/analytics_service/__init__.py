"""Analytics Service — Vendor Dashboard + Audience — Section 21 / 36.

Handles:
    - Vendor performance dashboard (GMV, CVR, views, etc.)
    - Real-time WebSocket analytics during live streams
    - Audience segmentation
    - ClickHouse query interface
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Analytics Service", version="1.0.0")


# ──────────────────────────────────────────────────────────────
# Dashboard Models — Section 21
# ──────────────────────────────────────────────────────────────

class DashboardMetrics(BaseModel):
    """Vendor dashboard — 100% metrics, zero subjectivity."""
    # Revenue
    gmv_total: float = 0.0
    gmv_today: float = 0.0
    gmv_7d: float = 0.0
    gmv_30d: float = 0.0

    # Conversion funnel
    impressions_30d: int = 0
    clicks_30d: int = 0
    adds_to_cart_30d: int = 0
    orders_30d: int = 0
    ctr_30d: float = 0.0
    cvr_30d: float = 0.0

    # Content performance
    total_products: int = 0
    active_products: int = 0
    avg_commerce_score: float = 0.0
    top_content_type: str = "photo"

    # Tier & Account
    tier: str = "bronze"
    account_weight: float = 1.0
    avg_rating: float = 0.0

    # Live performance
    total_lives: int = 0
    total_live_gmv: float = 0.0
    avg_live_viewers: int = 0

    # Delivery
    on_time_rate: float = 1.0
    return_rate: float = 0.0


class ProductAnalytics(BaseModel):
    product_id: str
    title: str = ""
    impressions: int = 0
    clicks: int = 0
    purchases: int = 0
    ctr: float = 0.0
    cvr: float = 0.0
    gmv: float = 0.0
    pool_level: str = "L1"
    commerce_score: float = 0.0


class AudienceSegment(BaseModel):
    """Section 36 — Audience analytics."""
    segment_name: str
    user_count: int = 0
    avg_order_value: float = 0.0
    purchase_frequency: float = 0.0
    top_categories: list[str] = []
    top_persona: str = "unknown"
    geo_distribution: dict[str, float] = {}


# In-memory analytics store
_vendor_metrics: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/analytics/{vendor_id}/dashboard", response_model=DashboardMetrics)
async def get_dashboard(vendor_id: str):
    """Get vendor dashboard metrics — Section 21."""
    metrics = _vendor_metrics.get(vendor_id, {})
    return DashboardMetrics(**metrics)


@app.get("/api/v1/analytics/{vendor_id}/products")
async def get_product_analytics(
    vendor_id: str,
    sort: str = Query(default="gmv", regex="^(gmv|ctr|cvr|impressions|commerce_score)$"),
    limit: int = Query(default=20, le=50),
):
    """Per-product analytics breakdown."""
    # In production: query ClickHouse
    return {"products": [], "vendor_id": vendor_id}


@app.get("/api/v1/analytics/{vendor_id}/audience")
async def get_audience(vendor_id: str):
    """Audience segmentation — Section 36."""
    return {
        "vendor_id": vendor_id,
        "total_unique_buyers": 0,
        "segments": [
            AudienceSegment(
                segment_name="Acheteurs fidèles",
                user_count=0,
                avg_order_value=0,
                top_persona="quality_seeker",
            ).model_dump(),
            AudienceSegment(
                segment_name="Nouveaux visiteurs",
                user_count=0,
                avg_order_value=0,
                top_persona="unknown",
            ).model_dump(),
        ],
    }


@app.post("/api/v1/analytics/ingest")
async def ingest_event(event: dict):
    """Ingest analytics event (from Kafka consumer)."""
    vendor_id = event.get("vendor_id", "")
    event_type = event.get("event_type", "")

    metrics = _vendor_metrics.setdefault(vendor_id, {
        "gmv_total": 0, "impressions_30d": 0, "clicks_30d": 0,
        "orders_30d": 0, "total_products": 0,
    })

    if event_type == "order.completed":
        metrics["gmv_total"] = metrics.get("gmv_total", 0) + event.get("amount", 0)
        metrics["orders_30d"] = metrics.get("orders_30d", 0) + 1
    elif event_type == "product.viewed":
        metrics["impressions_30d"] = metrics.get("impressions_30d", 0) + 1
    elif event_type == "product.add_to_cart":
        metrics["clicks_30d"] = metrics.get("clicks_30d", 0) + 1

    # Update derived metrics
    impressions = metrics.get("impressions_30d", 0)
    clicks = metrics.get("clicks_30d", 0)
    orders = metrics.get("orders_30d", 0)
    metrics["ctr_30d"] = clicks / max(impressions, 1)
    metrics["cvr_30d"] = orders / max(clicks, 1)

    return {"status": "ingested"}


# ──────────────────────────────────────────────────────────────
# WebSocket — Real-Time Analytics during Live
# ──────────────────────────────────────────────────────────────

@app.websocket("/ws/analytics/{vendor_id}")
async def analytics_ws(websocket: WebSocket, vendor_id: str):
    """Push real-time analytics to vendor dashboard."""
    await websocket.accept()
    logger.info("Analytics WS connected: vendor=%s", vendor_id)

    try:
        while True:
            metrics = _vendor_metrics.get(vendor_id, {})
            await websocket.send_json({
                "event": "analytics_update",
                "data": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await asyncio.sleep(10)  # Every 10 seconds
    except WebSocketDisconnect:
        logger.info("Analytics WS disconnected: vendor=%s", vendor_id)
    except Exception:
        pass


import asyncio
