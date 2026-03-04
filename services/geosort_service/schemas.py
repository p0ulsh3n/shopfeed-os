"""Pydantic schemas for GeoSort Service — Section 43."""

from __future__ import annotations

from pydantic import BaseModel


class ClassifyRequest(BaseModel):
    order_id: str
    vendor_id: str
    buyer_country: str = "CI"
    buyer_city: str = ""
    buyer_commune: str = ""
    buyer_lat: float | None = None
    buyer_lon: float | None = None
    raw_address: str | None = None
    vendor_commune: str = ""
    vendor_city: str = ""
    vendor_country: str = "CI"
    vendor_lat: float | None = None
    vendor_lon: float | None = None


class ClassifyResponse(BaseModel):
    order_id: str
    geo_level: str
    commune: str = ""
    city: str = ""
    country: str = ""
    distance_km: float = 0.0
    confidence: float = 1.0
    method: str = "geo_hierarchy"
    shipping_suggestion: str = ""
    needs_verification: bool = False
