"""Pydantic schemas for Order Service — Section 40."""

from __future__ import annotations

from pydantic import BaseModel


class AddToCartRequest(BaseModel):
    product_id: str
    variant_id: str | None = None
    quantity: int = 1
    unit_price: float = 0.0


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


class UpdateStatusRequest(BaseModel):
    status: str
    tracking_number: str | None = None
