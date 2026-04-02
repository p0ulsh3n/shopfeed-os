"""Pydantic schemas for Order Service — Section 40."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AddToCartRequest(BaseModel):
    product_id: str
    vendor_id: str = ""
    variant_id: str | None = None
    quantity: int = 1
    unit_price: float = 0.0
    weight_g: int = 500              # Product weight in grams
    vendor_lat: float | None = None  # Vendor GPS for geosort
    vendor_lon: float | None = None


class CheckoutRequest(BaseModel):
    buyer_id: str
    payment_method: str = "stripe"      # stripe | cinetpay | cod
    shipping_name: str = ""
    shipping_phone: str = ""
    shipping_street: str = ""
    shipping_commune: str = ""
    shipping_city: str = ""
    shipping_country: str = ""
    shipping_lat: float | None = None
    shipping_lon: float | None = None
    raw_address: str | None = None      # For NLP geocoding


class EstimateShippingRequest(BaseModel):
    """Preview shipping costs before checkout."""
    buyer_lat: float
    buyer_lon: float
    items: list[dict] = Field(
        ...,
        description="Cart items with vendor_id, weight_g, quantity, unit_price, vendor_lat, vendor_lon",
    )
    vendors_configs: dict[str, dict] = Field(
        default_factory=dict,
        description="Vendor shipping configs keyed by vendor_id",
    )


class VendorShippingDetail(BaseModel):
    vendor_id: str
    zone: str
    zone_label: str
    total_weight_g: int
    item_count: int
    subtotal: float
    shipping_cost: float
    is_free: bool = False
    free_reason: str = ""
    distance_km: float = 0.0
    vendor_city: str = ""
    buyer_city: str = ""
    shipping_suggestion: str = ""


class EstimateShippingResponse(BaseModel):
    total_shipping_cost: float
    total_items_cost: float
    grand_total: float
    vendor_count: int
    vendors: list[VendorShippingDetail]
    free_shipping_hints: list[str] = Field(default_factory=list)


class UpdateStatusRequest(BaseModel):
    status: str
    tracking_number: str | None = None
