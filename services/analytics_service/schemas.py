"""Pydantic schemas for Analytics Service — Section 21 / 36."""

from __future__ import annotations

from pydantic import BaseModel


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
