"""Analytics Service — REST + WebSocket Routes — Section 21 / 36."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from .schemas import AudienceSegment, DashboardMetrics

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Analytics Service", version="1.0.0")

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
