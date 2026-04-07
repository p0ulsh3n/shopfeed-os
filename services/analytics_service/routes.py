"""
Analytics Service — REST + WebSocket Routes — Section 21 / 36.

Migration: dicts Python in-memory (_vendor_metrics, _product_events, _buyer_profiles)
→ SQLAlchemy 2.0 ORM via AnalyticsRepository.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.repositories.analytics_repository import AnalyticsRepository

from .schemas import AudienceSegment, DashboardMetrics, ProductAnalytics

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Analytics Service", version="1.0.0")

_analytics_repo = AnalyticsRepository()


# ──────────────────────────────────────────────────────────────
# REST Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/v1/analytics/{vendor_id}/dashboard", response_model=DashboardMetrics)
async def get_dashboard(
    vendor_id: str,
    session: AsyncSession = Depends(get_db),
):
    """Get vendor dashboard metrics — Section 21."""
    metrics = await _analytics_repo.get_vendor_metrics(session, vendor_id)
    if not metrics:
        return DashboardMetrics()
    return DashboardMetrics(
        gmv_total=metrics.gmv_total,
        impressions_30d=metrics.impressions_30d,
        clicks_30d=metrics.clicks_30d,
        orders_30d=metrics.orders_30d,
        ctr_30d=metrics.ctr_30d,
        cvr_30d=metrics.cvr_30d,
        total_products=metrics.total_products,
    )


@app.get("/api/v1/analytics/{vendor_id}/products")
async def get_product_analytics(
    vendor_id: str,
    sort: str = Query(default="gmv", regex="^(gmv|ctr|cvr|impressions|commerce_score)$"),
    limit: int = Query(default=20, le=50),
    session: AsyncSession = Depends(get_db),
):
    """Per-product analytics breakdown."""
    counters = await _analytics_repo.list_product_counters(session, vendor_id, limit=limit)

    products = []
    for c in counters:
        ctr = c.clicks / max(c.impressions, 1)
        cvr = c.purchases / max(c.clicks, 1)
        products.append(ProductAnalytics(
            product_id=str(c.product_id),
            title=c.title,
            impressions=c.impressions,
            clicks=c.clicks,
            purchases=c.purchases,
            ctr=round(ctr, 4),
            cvr=round(cvr, 4),
            gmv=round(c.gmv, 2),
            pool_level=c.pool_level,
            commerce_score=c.commerce_score,
        ).model_dump())

    sort_key = sort if sort in ("gmv", "ctr", "cvr", "impressions", "commerce_score") else "gmv"
    products.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
    return {"products": products, "vendor_id": vendor_id, "total": len(products)}


@app.get("/api/v1/analytics/{vendor_id}/audience")
async def get_audience(
    vendor_id: str,
    session: AsyncSession = Depends(get_db),
):
    """
    Audience segmentation — Section 36.
    Note: segmentation avancée nécessite une table orders avec buyer lookup.
    Version simplifiée retournée depuis les compteurs produits.
    """
    counters = await _analytics_repo.list_product_counters(session, vendor_id, limit=1)
    if not counters:
        return {
            "vendor_id": vendor_id,
            "total_unique_buyers": 0,
            "segments": [],
        }
    total_purchases = sum(c.purchases for c in counters) if counters else 0
    return {
        "vendor_id": vendor_id,
        "total_unique_buyers": total_purchases,
        "segments": [],
        "note": "Segmentation détaillée disponible via ClickHouse pipeline",
    }


@app.post("/api/v1/analytics/ingest")
async def ingest_event(
    event: dict,
    session: AsyncSession = Depends(get_db),
):
    """Ingest analytics event — upsert via AnalyticsRepository ORM."""
    vendor_id_raw = event.get("vendor_id", "")
    product_id_raw = event.get("product_id", "")
    event_type = event.get("event_type", "")

    if not vendor_id_raw:
        return {"status": "ignored", "reason": "missing vendor_id"}

    try:
        vendor_uuid = uuid.UUID(str(vendor_id_raw))
    except (ValueError, AttributeError):
        return {"status": "ignored", "reason": "invalid vendor_id"}

    # Metric updates
    metric_updates: dict = {}
    counter_updates: dict = {}

    if event_type == "order.completed":
        amount = float(event.get("amount", 0))
        metric_updates = {
            "gmv_total": amount,
            "orders_30d": 1,
        }
        counter_updates = {
            "purchases": 1,
            "gmv": amount,
        }
    elif event_type == "product.viewed":
        metric_updates = {"impressions_30d": 1}
        counter_updates = {"impressions": 1}
    elif event_type == "product.add_to_cart":
        metric_updates = {"clicks_30d": 1}
        counter_updates = {"clicks": 1}

    if metric_updates:
        # Upsert incrémental via ORM pg_insert ON CONFLICT DO UPDATE
        existing = await _analytics_repo.get_vendor_metrics(session, vendor_uuid)
        if existing:
            updated = {
                "gmv_total": existing.gmv_total + metric_updates.get("gmv_total", 0),
                "impressions_30d": existing.impressions_30d + metric_updates.get("impressions_30d", 0),
                "clicks_30d": existing.clicks_30d + metric_updates.get("clicks_30d", 0),
                "orders_30d": existing.orders_30d + metric_updates.get("orders_30d", 0),
            }
            updated["ctr_30d"] = updated["clicks_30d"] / max(updated["impressions_30d"], 1)
            updated["cvr_30d"] = updated["orders_30d"] / max(updated["clicks_30d"], 1)
        else:
            updated = {
                "gmv_total": metric_updates.get("gmv_total", 0),
                "impressions_30d": metric_updates.get("impressions_30d", 0),
                "clicks_30d": metric_updates.get("clicks_30d", 0),
                "orders_30d": metric_updates.get("orders_30d", 0),
                "ctr_30d": 0.0,
                "cvr_30d": 0.0,
                "total_products": 0,
            }
        await _analytics_repo.upsert_vendor_metrics(session, vendor_uuid, updated)

    if counter_updates and product_id_raw:
        try:
            product_uuid = uuid.UUID(str(product_id_raw))
            await _analytics_repo.upsert_product_counter(
                session, product_uuid, vendor_uuid,
                {
                    **counter_updates,
                    "title": event.get("product_title", ""),
                }
            )
        except (ValueError, AttributeError):
            pass

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
            # Session dédiée pour ce polling périodique
            from shared.db.session import AsyncSessionLocal
            async with AsyncSessionLocal() as session:
                metrics = await _analytics_repo.get_vendor_metrics(session, vendor_id)
                data = {}
                if metrics:
                    data = {
                        "gmv_total": metrics.gmv_total,
                        "impressions_30d": metrics.impressions_30d,
                        "orders_30d": metrics.orders_30d,
                        "ctr_30d": metrics.ctr_30d,
                        "cvr_30d": metrics.cvr_30d,
                    }
            await websocket.send_json({
                "event": "analytics_update",
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        logger.info("Analytics WS disconnected: vendor=%s", vendor_id)
    except Exception:
        pass
