"""Pydantic schemas for Feed Service — Section 04 / 13 / 20."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    currency: str = ""
    media_url: str = ""
    thumbnail_url: str = ""
    fomo: FomoSignals = Field(default_factory=FomoSignals)
    shipping_free: bool = False
    estimated_days: int = 3


class FeedResponse(BaseModel):
    items: list[FeedItem]
    next_cursor: str | None = None
    session_state_updated: bool = True
