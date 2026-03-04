"""Pydantic schemas for Live Service — Section 07."""

from __future__ import annotations

from pydantic import BaseModel


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
