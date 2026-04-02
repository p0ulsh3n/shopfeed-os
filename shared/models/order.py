"""Order models — Section 40."""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class PaymentStatus(str, enum.Enum):
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"


class ShippingAddress(BaseModel):
    name: str = ""
    phone: str = ""
    street: str = ""
    commune: str = ""
    city: str = ""
    country: str = ""                       # ISO 3166 — set by app
    lat: float | None = None
    lon: float | None = None
    raw_text: str | None = None     # For NLP geocoding of informal addresses


class OrderItem(BaseModel):
    product_id: UUID
    variant_id: UUID | None = None
    quantity: int = 1
    unit_price: float = 0.0


class Order(BaseModel):
    """Order record — Section 40."""
    id: UUID = Field(default_factory=uuid4)
    buyer_id: UUID
    vendor_id: UUID
    status: OrderStatus = OrderStatus.PENDING
    items: list[OrderItem] = Field(default_factory=list)
    total_gmv: float = 0.0
    currency: str = "EUR"
    shipping_address: ShippingAddress = Field(default_factory=ShippingAddress)
    payment_method: str | None = None
    payment_status: PaymentStatus = PaymentStatus.PENDING
    tracking_number: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    confirmed_at: datetime | None = None
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
