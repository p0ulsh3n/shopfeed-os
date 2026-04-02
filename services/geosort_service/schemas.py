"""Pydantic schemas for GeoSort Service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Order classification request.

    At minimum, provide vendor and buyer GPS coordinates.
    The app tracks lat/lon from user geolocation.
    """
    order_id: str
    vendor_id: str

    # Vendor GPS (required)
    vendor_lat: float
    vendor_lon: float

    # Buyer GPS (required)
    buyer_lat: float
    buyer_lon: float

    # Optional: custom radius for Zone A (default 50km)
    same_zone_radius_km: float = Field(
        50.0,
        description="Max distance in km for Zone A (Livraison). Default 50km.",
    )


class ClassifyResponse(BaseModel):
    """Order classification result."""
    order_id: str

    # Zone classification
    zone: str = Field(..., description="A, B, or C")
    zone_label: str = Field(..., description="local_delivery / national_shipping / international_shipping")
    geo_level: str = Field(..., description="L1/L2/L3/L4 (backward compatible)")

    # Distance
    distance_km: float = 0.0

    # Resolved locations
    buyer_city: str = ""
    buyer_country: str = ""
    buyer_country_code: str = ""
    vendor_city: str = ""
    vendor_country: str = ""
    vendor_country_code: str = ""

    # Shipping
    shipping_suggestion: str = ""
    confidence: float = 1.0
