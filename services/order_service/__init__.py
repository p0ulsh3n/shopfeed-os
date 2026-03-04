"""Order Service — Checkout & Payments — Section 40.

Handles:
    - Cart management
    - Checkout flow (Stripe / CinetPay)
    - Order tracking & status updates
    - Emits events to Kafka for GeoSort classification
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.events import Topic, CommerceEvent, CommerceEventType
from shared.models.order import OrderStatus, PaymentStatus

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Order Service", version="1.0.0")

_orders: dict[str, dict] = {}
_carts: dict[str, list[dict]] = {}


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────

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
    shipping_country: str = "CI"
    shipping_lat: float | None = None
    shipping_lon: float | None = None
    raw_address: str | None = None      # For NLP geocoding


class UpdateStatusRequest(BaseModel):
    status: str
    tracking_number: str | None = None


# ──────────────────────────────────────────────────────────────
# Cart Endpoints
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/cart/{user_id}/add")
async def add_to_cart(user_id: str, req: AddToCartRequest):
    cart = _carts.setdefault(user_id, [])
    cart.append(req.model_dump())
    return {"cart_size": len(cart), "items": cart}


@app.get("/api/v1/cart/{user_id}")
async def get_cart(user_id: str):
    cart = _carts.get(user_id, [])
    total = sum(item["unit_price"] * item["quantity"] for item in cart)
    return {"items": cart, "total": round(total, 2), "currency": "EUR"}


@app.delete("/api/v1/cart/{user_id}")
async def clear_cart(user_id: str):
    _carts.pop(user_id, None)
    return {"status": "cleared"}


# ──────────────────────────────────────────────────────────────
# Checkout & Orders
# ──────────────────────────────────────────────────────────────

@app.post("/api/v1/orders/checkout")
async def checkout(req: CheckoutRequest):
    """Process checkout — create order, initiate payment.

    Flow:
        1. Validate cart
        2. Create order record
        3. Reserve stock
        4. Initiate payment (Stripe / CinetPay)
        5. Emit OrderCreated event → GeoSort classification
    """
    cart = _carts.get(req.buyer_id, [])
    if not cart:
        raise HTTPException(status_code=400, detail="Cart is empty")

    order_id = str(uuid.uuid4())
    total = sum(item["unit_price"] * item["quantity"] for item in cart)

    order = {
        "id": order_id,
        "buyer_id": req.buyer_id,
        "status": OrderStatus.PENDING,
        "items": cart.copy(),
        "total_gmv": round(total, 2),
        "currency": "EUR",
        "shipping_address": {
            "name": req.shipping_name,
            "phone": req.shipping_phone,
            "street": req.shipping_street,
            "commune": req.shipping_commune,
            "city": req.shipping_city,
            "country": req.shipping_country,
            "lat": req.shipping_lat,
            "lon": req.shipping_lon,
            "raw_text": req.raw_address,
        },
        "payment_method": req.payment_method,
        "payment_status": PaymentStatus.PENDING,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _orders[order_id] = order

    # Clear cart
    _carts.pop(req.buyer_id, None)

    # In production: emit to Kafka for GeoSort classification
    logger.info("Order created: %s (GMV=%.2f %s)", order_id, total, order["currency"])

    return {
        "order_id": order_id,
        "total": total,
        "status": "pending",
        "payment_url": f"https://checkout.stripe.com/pay/{order_id}" if req.payment_method == "stripe" else None,
    }


@app.get("/api/v1/orders/{order_id}")
async def get_order(order_id: str):
    if order_id not in _orders:
        raise HTTPException(status_code=404, detail="Order not found")
    return _orders[order_id]


@app.patch("/api/v1/orders/{order_id}/status")
async def update_order_status(order_id: str, req: UpdateStatusRequest):
    if order_id not in _orders:
        raise HTTPException(status_code=404, detail="Order not found")

    _orders[order_id]["status"] = req.status
    if req.tracking_number:
        _orders[order_id]["tracking_number"] = req.tracking_number

    now = datetime.now(timezone.utc).isoformat()
    if req.status == "confirmed":
        _orders[order_id]["confirmed_at"] = now
    elif req.status == "shipped":
        _orders[order_id]["shipped_at"] = now
    elif req.status == "delivered":
        _orders[order_id]["delivered_at"] = now

    return _orders[order_id]


@app.get("/api/v1/orders")
async def list_orders(buyer_id: str | None = None, vendor_id: str | None = None):
    orders = list(_orders.values())
    if buyer_id:
        orders = [o for o in orders if o["buyer_id"] == buyer_id]
    return {"orders": orders, "total": len(orders)}
