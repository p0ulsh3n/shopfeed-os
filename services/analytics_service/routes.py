"""Analytics Service — REST + WebSocket Routes — Section 21 / 36."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect

from .schemas import AudienceSegment, DashboardMetrics, ProductAnalytics

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Analytics Service", version="1.0.0")

# In-memory analytics stores — will be replaced by ClickHouse in deployment
_vendor_metrics: dict[str, dict] = {}
_product_events: dict[str, dict[str, dict]] = {}   # vendor_id → product_id → counters
_buyer_profiles: dict[str, dict[str, list]] = {}    # vendor_id → buyer_id → [orders]


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
    """Per-product analytics breakdown — aggregated from ingested events."""
    products_data = _product_events.get(vendor_id, {})

    products = []
    for product_id, counters in products_data.items():
        impressions = counters.get("impressions", 0)
        clicks = counters.get("clicks", 0)
        purchases = counters.get("purchases", 0)
        gmv = counters.get("gmv", 0.0)

        ctr = clicks / max(impressions, 1)
        cvr = purchases / max(clicks, 1)

        products.append(ProductAnalytics(
            product_id=product_id,
            title=counters.get("title", ""),
            impressions=impressions,
            clicks=clicks,
            purchases=purchases,
            ctr=round(ctr, 4),
            cvr=round(cvr, 4),
            gmv=round(gmv, 2),
            pool_level=counters.get("pool_level", "L1"),
            commerce_score=counters.get("commerce_score", 0.0),
        ).model_dump())

    # Sort by requested metric
    sort_key = sort if sort in ("gmv", "ctr", "cvr", "impressions", "commerce_score") else "gmv"
    products.sort(key=lambda x: x.get(sort_key, 0), reverse=True)

    return {"products": products[:limit], "vendor_id": vendor_id, "total": len(products)}


@app.get("/api/v1/analytics/{vendor_id}/audience")
async def get_audience(vendor_id: str):
    """Audience segmentation — Section 36.

    Dynamically segments buyers from order history into cohorts.
    """
    buyer_data = _buyer_profiles.get(vendor_id, {})

    if not buyer_data:
        return {
            "vendor_id": vendor_id,
            "total_unique_buyers": 0,
            "segments": [],
        }

    # Segment buyers by purchase frequency
    loyal_buyers = []      # 3+ purchases
    repeat_buyers = []     # 2 purchases
    one_time_buyers = []   # 1 purchase

    for buyer_id, orders in buyer_data.items():
        count = len(orders)
        avg_value = sum(o.get("amount", 0) for o in orders) / max(count, 1)
        buyer_info = {"buyer_id": buyer_id, "order_count": count, "avg_value": avg_value}

        if count >= 3:
            loyal_buyers.append(buyer_info)
        elif count == 2:
            repeat_buyers.append(buyer_info)
        else:
            one_time_buyers.append(buyer_info)

    segments = []
    if loyal_buyers:
        avg_ov = sum(b["avg_value"] for b in loyal_buyers) / len(loyal_buyers)
        segments.append(AudienceSegment(
            segment_name="Loyal Buyers",
            user_count=len(loyal_buyers),
            avg_order_value=round(avg_ov, 2),
            purchase_frequency=sum(b["order_count"] for b in loyal_buyers) / len(loyal_buyers),
            top_persona="loyal_repeat",
        ).model_dump())

    if repeat_buyers:
        avg_ov = sum(b["avg_value"] for b in repeat_buyers) / len(repeat_buyers)
        segments.append(AudienceSegment(
            segment_name="Repeat Buyers",
            user_count=len(repeat_buyers),
            avg_order_value=round(avg_ov, 2),
            purchase_frequency=2.0,
            top_persona="quality_seeker",
        ).model_dump())

    if one_time_buyers:
        avg_ov = sum(b["avg_value"] for b in one_time_buyers) / len(one_time_buyers)
        segments.append(AudienceSegment(
            segment_name="New Buyers",
            user_count=len(one_time_buyers),
            avg_order_value=round(avg_ov, 2),
            purchase_frequency=1.0,
            top_persona="first_time",
        ).model_dump())

    return {
        "vendor_id": vendor_id,
        "total_unique_buyers": len(buyer_data),
        "segments": segments,
    }


@app.post("/api/v1/analytics/ingest")
async def ingest_event(event: dict):
    """Ingest analytics event (from Kafka consumer).

    Populates vendor metrics, per-product analytics, and buyer profiles.
    """
    vendor_id = event.get("vendor_id", "")
    event_type = event.get("event_type", "")
    product_id = event.get("product_id", "")

    # ── Update vendor-level metrics ──
    metrics = _vendor_metrics.setdefault(vendor_id, {
        "gmv_total": 0, "impressions_30d": 0, "clicks_30d": 0,
        "orders_30d": 0, "total_products": 0,
    })

    # ── Update per-product counters ──
    product_counters = _product_events.setdefault(vendor_id, {}).setdefault(product_id, {
        "impressions": 0, "clicks": 0, "purchases": 0, "gmv": 0.0,
        "title": event.get("product_title", ""),
    })

    if event_type == "order.completed":
        amount = event.get("amount", 0)
        metrics["gmv_total"] = metrics.get("gmv_total", 0) + amount
        metrics["orders_30d"] = metrics.get("orders_30d", 0) + 1
        product_counters["purchases"] = product_counters.get("purchases", 0) + 1
        product_counters["gmv"] = product_counters.get("gmv", 0.0) + amount

        # Track buyer for audience segmentation
        buyer_id = event.get("buyer_id", "")
        if buyer_id:
            _buyer_profiles.setdefault(vendor_id, {}).setdefault(buyer_id, []).append({
                "amount": amount,
                "product_id": product_id,
                "timestamp": event.get("timestamp", ""),
            })

    elif event_type == "product.viewed":
        metrics["impressions_30d"] = metrics.get("impressions_30d", 0) + 1
        product_counters["impressions"] = product_counters.get("impressions", 0) + 1

    elif event_type == "product.add_to_cart":
        metrics["clicks_30d"] = metrics.get("clicks_30d", 0) + 1
        product_counters["clicks"] = product_counters.get("clicks", 0) + 1

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
