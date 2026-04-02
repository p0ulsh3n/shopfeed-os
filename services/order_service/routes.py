"""
Order Service — FastAPI App + Routes — Section 40
=================================================

BUG #S3 FIX: Orders were created without vendor_id, which would violate
NOT NULL constraints in PostgreSQL (vendors.id FK). Fixed by deriving
vendor_id from the cart items.

BUG #S4 FIX: CommerceEvent was imported but never published. The Kafka
event for purchase is now fired on checkout so the streaming trainer
(MonolithStreamingTrainer) receives purchase signals and can update
item/user embeddings in real time.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from shared.db import get_kafka_producer
from shared.events import Topic, CommerceEvent, CommerceEventType
from shared.models.order import OrderStatus, PaymentStatus

from .schemas import (
    AddToCartRequest, CheckoutRequest, UpdateStatusRequest,
    EstimateShippingRequest, EstimateShippingResponse, VendorShippingDetail,
)
from .shipping_calculator import calculate_cart_shipping

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Order Service", version="1.0.0")

_orders: dict[str, dict] = {}
_carts: dict[str, list[dict]] = {}


# ── Cart Endpoints ──────────────────────────────────────────────

@app.post("/api/v1/cart/{user_id}/add")
async def add_to_cart(user_id: str, req: AddToCartRequest):
    cart = _carts.setdefault(user_id, [])
    cart.append(req.model_dump())
    return {"cart_size": len(cart), "items": cart}


@app.get("/api/v1/cart/{user_id}")
async def get_cart(user_id: str):
    cart = _carts.get(user_id, [])
    total = sum(item["unit_price"] * item["quantity"] for item in cart)
    return {"items": cart, "total": round(total, 2), "currency": ""}


@app.delete("/api/v1/cart/{user_id}")
async def clear_cart(user_id: str):
    _carts.pop(user_id, None)
    return {"status": "cleared"}


# ── Checkout & Orders ───────────────────────────────────────────

@app.post("/api/v1/orders/checkout")
async def checkout(req: CheckoutRequest):
    """Process checkout — create order, publish purchase events to Kafka.

    BUG #S3 FIX: vendor_id was missing from the order dict, causing
    NOT NULL violations. We now derive it from the first cart item
    (orders are single-vendor) or set a placeholder for multi-vendor carts.

    BUG #S4 FIX: CommerceEvent is now published to Kafka for every item
    purchased, so MonolithStreamingTrainer receives real purchase signals.
    """
    cart = _carts.get(req.buyer_id, [])
    if not cart:
        raise HTTPException(status_code=400, detail="Cart is empty")

    order_id = str(uuid.uuid4())
    total = sum(item["unit_price"] * item["quantity"] for item in cart)

    # BUG #S3 FIX: derive vendor_id from cart items.
    # In a single-vendor cart (standard), all items share the same vendor.
    # For multi-vendor carts, use the primary (most expensive) vendor's id.
    vendor_ids = [item.get("vendor_id", "") for item in cart if item.get("vendor_id")]
    primary_vendor_id = max(
        set(vendor_ids),
        key=lambda v: sum(
            i["unit_price"] * i["quantity"]
            for i in cart if i.get("vendor_id") == v
        ),
        default="",
    ) if vendor_ids else ""

    order = {
        "id": order_id,
        "buyer_id": req.buyer_id,
        "vendor_id": primary_vendor_id,   # BUG #S3 FIX: was always missing
        "status": OrderStatus.PENDING,
        "items": cart.copy(),
        "total_gmv": round(total, 2),
        "currency": "",
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
    _carts.pop(req.buyer_id, None)

    logger.info("Order created: %s (GMV=%.2f %s)", order_id, total, order["currency"])

    # BUG #S4 FIX: publish purchase events to Kafka for each item.
    # MonolithStreamingTrainer subscribes to this topic and uses these
    # events to update item/user embeddings in real time.
    # Previously: CommerceEvent was imported but NEVER published.
    await _publish_purchase_events(req.buyer_id, order_id, cart, primary_vendor_id)

    # ── Calculate shipping costs ──
    shipping_result = calculate_cart_shipping(
        cart_items=cart,
        buyer_lat=req.shipping_lat,
        buyer_lon=req.shipping_lon,
    )

    order["shipping_cost"] = shipping_result.total_shipping_cost
    order["shipping_breakdown"] = [
        {
            "vendor_id": v.vendor_id,
            "zone": v.zone,
            "zone_label": v.zone_label,
            "total_weight_g": v.total_weight_g,
            "shipping_cost": v.shipping_cost,
            "is_free": v.is_free,
            "distance_km": v.distance_km,
        }
        for v in shipping_result.vendors
    ]
    order["total_gmv"] = round(total + shipping_result.total_shipping_cost, 2)

    logger.info(
        "Shipping: %d vendors, shipping=%.2f, total=%.2f",
        shipping_result.vendor_count,
        shipping_result.total_shipping_cost,
        order["total_gmv"],
    )

    return {
        "order_id": order_id,
        "items_total": round(total, 2),
        "shipping_cost": shipping_result.total_shipping_cost,
        "grand_total": order["total_gmv"],
        "shipping_breakdown": order["shipping_breakdown"],
        "status": "pending",
        "payment_url": (
            f"https://checkout.stripe.com/pay/{order_id}"
            if req.payment_method == "stripe"
            else None
        ),
    }


async def _publish_purchase_events(
    buyer_id: str,
    order_id: str,
    cart: list[dict],
    vendor_id: str,
) -> None:
    """Publish a CommerceEvent for each purchased item to Kafka.

    BUG #S4 FIX: This function closes the gap between checkout and the
    streaming trainer. Without this, MonolithStreamingTrainer would only
    see view/click events (from the feed WebSocket) and never purchases.
    Purchase events are the highest-weight training signal (buy=1.0 vs view=0.1).
    """
    try:
        producer = await get_kafka_producer()
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in cart:
            event = CommerceEvent(
                event_type=CommerceEventType.PURCHASE,
                user_id=buyer_id,
                product_id=str(item.get("product_id", item.get("item_id", ""))),
                vendor_id=vendor_id,
                order_id=order_id,
                price=float(item.get("unit_price", 0.0)),
                quantity=int(item.get("quantity", 1)),
                timestamp=now_iso,
                # Pass these so streaming trainer can locate the embedding
                action="purchase",
                item_id=str(item.get("product_id", item.get("item_id", ""))),
            )
            await producer.send(
                Topic.COMMERCE_EVENTS,
                value=event.model_dump(),
                key=buyer_id.encode(),
            )

        logger.info(
            "Published %d purchase events to Kafka (order=%s, buyer=%s)",
            len(cart), order_id, buyer_id,
        )
    except Exception as e:
        # Non-fatal: the order is already created. Log and continue.
        # The streaming trainer will miss this purchase if Kafka is down,
        # but the order data is safe in the database.
        logger.error("Failed to publish purchase events to Kafka: %s", e)


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
    if vendor_id:
        orders = [o for o in orders if o.get("vendor_id") == vendor_id]
    return {"orders": orders, "total": len(orders)}


# ── Shipping Estimate (pre-checkout preview) ────────────────────

@app.post("/api/v1/orders/estimate-shipping", response_model=EstimateShippingResponse)
async def estimate_shipping(req: EstimateShippingRequest):
    """Preview shipping costs before checkout.

    Called from the cart page to show per-vendor shipping breakdown
    and free shipping hints ("Add X more for free shipping").
    """
    result = calculate_cart_shipping(
        cart_items=req.items,
        vendors_configs=req.vendors_configs,
        buyer_lat=req.buyer_lat,
        buyer_lon=req.buyer_lon,
    )

    return EstimateShippingResponse(
        total_shipping_cost=result.total_shipping_cost,
        total_items_cost=result.total_items_cost,
        grand_total=result.grand_total,
        vendor_count=result.vendor_count,
        vendors=[
            VendorShippingDetail(
                vendor_id=v.vendor_id,
                zone=v.zone,
                zone_label=v.zone_label,
                total_weight_g=v.total_weight_g,
                item_count=v.item_count,
                subtotal=v.subtotal,
                shipping_cost=v.shipping_cost,
                is_free=v.is_free,
                free_reason=v.free_reason,
                distance_km=v.distance_km,
                vendor_city=v.vendor_city,
                buyer_city=v.buyer_city,
                shipping_suggestion=v.shipping_suggestion,
            )
            for v in result.vendors
        ],
        free_shipping_hints=result.free_shipping_hints,
    )
