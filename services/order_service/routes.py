"""
Order Service — FastAPI App + Routes — Section 40.

Migration: dicts Python in-memory (_orders, _carts)
→ SQLAlchemy 2.0 ORM via OrderRepository + CartRepository.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from shared.db import get_kafka_producer
from shared.events import CommerceEvent, CommerceEventType, Topic
from shared.models.order import OrderStatus, PaymentStatus
from shared.repositories.order_repository import CartRepository, OrderRepository

from .schemas import (
    AddToCartRequest,
    CheckoutRequest,
    EstimateShippingRequest,
    EstimateShippingResponse,
    UpdateStatusRequest,
    VendorShippingDetail,
)
from .shipping_calculator import calculate_cart_shipping

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — Order Service", version="1.0.0")

_cart_repo = CartRepository()
_order_repo = OrderRepository()


def _cart_item_to_dict(item) -> dict:
    return {
        "product_id": str(item.product_id),
        "vendor_id": str(item.vendor_id),
        "variant_id": str(item.variant_id) if item.variant_id else None,
        "quantity": item.quantity,
        "unit_price": item.unit_price,
        "currency": item.currency,
        **(item.product_snapshot or {}),
    }


def _order_to_dict(order) -> dict:
    return {
        "id": str(order.id),
        "buyer_id": str(order.buyer_id) if order.buyer_id else None,
        "vendor_id": str(order.vendor_id) if order.vendor_id else None,
        "status": order.status,
        "total_amount": order.total_amount,
        "currency": order.currency,
        "shipping_cost": order.shipping_cost,
        "shipping_breakdown": order.shipping_breakdown,
        "shipping_address": order.shipping_address,
        "payment_method": order.payment_method,
        "payment_status": order.payment_status,
        "tracking_number": order.tracking_number,
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "items": [
            {
                "product_id": str(i.product_id) if i.product_id else None,
                "quantity": i.quantity,
                "unit_price": i.unit_price,
                "currency": i.currency,
            }
            for i in (order.items or [])
        ],
    }


# ── Cart Endpoints ──────────────────────────────────────────────

@app.post("/api/v1/cart/{user_id}/add")
async def add_to_cart(
    user_id: str,
    req: AddToCartRequest,
    session: AsyncSession = Depends(get_db),
):
    data = req.model_dump()
    data["user_id"] = uuid.UUID(user_id)
    data["vendor_id"] = uuid.UUID(str(data.get("vendor_id", uuid.uuid4())))
    data["product_id"] = uuid.UUID(str(data.get("product_id") or data.get("item_id", uuid.uuid4())))
    data["product_snapshot"] = {
        "name": data.get("title", ""),
        "price": data.get("unit_price", 0),
    }
    await _cart_repo.add_item(session, data)

    items = await _cart_repo.get_items(session, user_id)
    return {"cart_size": len(items), "items": [_cart_item_to_dict(i) for i in items]}


@app.get("/api/v1/cart/{user_id}")
async def get_cart(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    items = await _cart_repo.get_items(session, user_id)
    total = sum(i.unit_price * i.quantity for i in items)
    return {
        "items": [_cart_item_to_dict(i) for i in items],
        "total": round(total, 2),
        "currency": items[0].currency if items else "",
    }


@app.delete("/api/v1/cart/{user_id}")
async def clear_cart(
    user_id: str,
    session: AsyncSession = Depends(get_db),
):
    await _cart_repo.clear(session, user_id)
    return {"status": "cleared"}


# ── Checkout & Orders ───────────────────────────────────────────

@app.post("/api/v1/orders/checkout")
async def checkout(
    req: CheckoutRequest,
    session: AsyncSession = Depends(get_db),
):
    """Process checkout — create order via ORM, publish purchase events to Kafka."""
    cart_items = await _cart_repo.get_items(session, req.buyer_id)
    if not cart_items:
        raise HTTPException(status_code=400, detail="Cart is empty")

    total = sum(i.unit_price * i.quantity for i in cart_items)

    # vendor_id primaire depuis le panier
    vendor_ids = [str(i.vendor_id) for i in cart_items if i.vendor_id]
    primary_vendor_id = (
        max(
            set(vendor_ids),
            key=lambda v: sum(
                i.unit_price * i.quantity for i in cart_items if str(i.vendor_id) == v
            ),
            default="",
        )
        if vendor_ids
        else ""
    )

    cart_dicts = [_cart_item_to_dict(i) for i in cart_items]
    shipping_result = calculate_cart_shipping(
        cart_items=cart_dicts,
        buyer_lat=req.shipping_lat,
        buyer_lon=req.shipping_lon,
    )

    order = await _order_repo.create(session, {
        "buyer_id": uuid.UUID(req.buyer_id) if req.buyer_id else None,
        "vendor_id": uuid.UUID(primary_vendor_id) if primary_vendor_id else None,
        "status": OrderStatus.PENDING,
        "total_amount": round(total + shipping_result.total_shipping_cost, 2),
        "currency": cart_items[0].currency if cart_items else "",
        "shipping_cost": shipping_result.total_shipping_cost,
        "shipping_breakdown": [
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
        ],
        "shipping_address": {
            "name": req.shipping_name,
            "phone": req.shipping_phone,
            "street": req.shipping_street,
            "commune": req.shipping_commune,
            "city": req.shipping_city,
            "country": req.shipping_country,
            "lat": req.shipping_lat,
            "lon": req.shipping_lon,
        },
        "payment_method": req.payment_method,
        "payment_status": PaymentStatus.PENDING,
        "customer_id": req.buyer_id,
        "items": [
            {
                "product_id": uuid.UUID(str(i.product_id)) if i.product_id else None,
                "quantity": i.quantity,
                "unit_price": i.unit_price,
                "currency": i.currency,
                "product_snapshot": i.product_snapshot or {},
            }
            for i in cart_items
        ],
    })

    # Vider le panier
    await _cart_repo.clear(session, req.buyer_id)

    logger.info("Order created: %s (GMV=%.2f)", order.id, order.total_amount)

    # Publier les events Kafka (non-bloquant)
    await _publish_purchase_events(req.buyer_id, str(order.id), cart_dicts, primary_vendor_id)

    return {
        "order_id": str(order.id),
        "items_total": round(total, 2),
        "shipping_cost": shipping_result.total_shipping_cost,
        "grand_total": order.total_amount,
        "shipping_breakdown": order.shipping_breakdown,
        "status": "pending",
        "payment_url": (
            f"https://checkout.stripe.com/pay/{order.id}"
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
    except Exception as exc:
        logger.error("Failed to publish purchase events to Kafka: %s", exc)


@app.get("/api/v1/orders/{order_id}")
async def get_order(
    order_id: str,
    session: AsyncSession = Depends(get_db),
):
    order = await _order_repo.get_by_id(session, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return _order_to_dict(order)


@app.patch("/api/v1/orders/{order_id}/status")
async def update_order_status(
    order_id: str,
    req: UpdateStatusRequest,
    session: AsyncSession = Depends(get_db),
):
    order = await _order_repo.get_by_id(session, order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    extra = {}
    now = datetime.now(timezone.utc)
    if req.status == "confirmed":
        extra["confirmed_at"] = now
    elif req.status == "shipped":
        extra["shipped_at"] = now
    elif req.status == "delivered":
        extra["delivered_at"] = now
    if req.tracking_number:
        extra["tracking_number"] = req.tracking_number

    await _order_repo.update_status(session, order_id, req.status, **extra)
    order = await _order_repo.get_by_id(session, order_id)
    return _order_to_dict(order)


@app.get("/api/v1/orders")
async def list_orders(
    buyer_id: str | None = None,
    vendor_id: str | None = None,
    session: AsyncSession = Depends(get_db),
):
    if buyer_id:
        orders = await _order_repo.list_by_buyer(session, buyer_id)
    elif vendor_id:
        orders = await _order_repo.list_by_vendor(session, vendor_id)
    else:
        orders = []
    return {"orders": [_order_to_dict(o) for o in orders], "total": len(orders)}


# ── Shipping Estimate ────────────────────────────────────────────

@app.post("/api/v1/orders/estimate-shipping", response_model=EstimateShippingResponse)
async def estimate_shipping(req: EstimateShippingRequest):
    """Preview shipping costs before checkout — no DB needed."""
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
