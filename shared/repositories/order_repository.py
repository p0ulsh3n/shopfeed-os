"""
Repository — Orders, CartItems, Shipments
===========================================
CRUD SQLAlchemy 2.0. Zéro SQL brut.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.db.models.order import CartItemORM, OrderItemORM, OrderORM, ShipmentORM


class CartRepository:
    """Repository pour le panier persistant."""

    async def get_items(
        self, session: AsyncSession, user_id: uuid.UUID | str
    ) -> list[CartItemORM]:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        result = await session.execute(
            select(CartItemORM).where(CartItemORM.user_id == uid)
        )
        return result.scalars().all()  # type: ignore[return-value]

    async def add_item(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> CartItemORM:
        item = CartItemORM(**data)
        session.add(item)
        await session.flush()
        return item

    async def clear(
        self, session: AsyncSession, user_id: uuid.UUID | str
    ) -> None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        await session.execute(
            delete(CartItemORM).where(CartItemORM.user_id == uid)
        )
        await session.flush()


class OrderRepository:
    """Repository pour les commandes."""

    async def get_by_id(
        self, session: AsyncSession, order_id: uuid.UUID | str
    ) -> OrderORM | None:
        oid = uuid.UUID(str(order_id)) if isinstance(order_id, str) else order_id
        result = await session.execute(
            select(OrderORM)
            .options(
                selectinload(OrderORM.items),
                selectinload(OrderORM.shipment),
            )
            .where(OrderORM.id == oid, OrderORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_by_order_number(
        self,
        session: AsyncSession,
        order_number: str,
        shop_id: str | None = None,
        customer_id: str | None = None,
    ) -> OrderORM | None:
        q = (
            select(OrderORM)
            .options(
                selectinload(OrderORM.items),
                selectinload(OrderORM.shipment),
            )
            .where(
                OrderORM.order_number == order_number.upper(),
                OrderORM.deleted_at.is_(None),
            )
        )
        if customer_id:
            q = q.where(OrderORM.customer_id == customer_id)
        result = await session.execute(q)
        return result.scalar_one_or_none()

    async def list_by_buyer(
        self,
        session: AsyncSession,
        buyer_id: uuid.UUID | str,
        *,
        limit: int = 10,
    ) -> list[OrderORM]:
        uid = uuid.UUID(str(buyer_id)) if isinstance(buyer_id, str) else buyer_id
        result = await session.execute(
            select(OrderORM)
            .options(selectinload(OrderORM.items))
            .where(OrderORM.buyer_id == uid, OrderORM.deleted_at.is_(None))
            .order_by(OrderORM.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()  # type: ignore[return-value]

    async def list_by_vendor(
        self,
        session: AsyncSession,
        vendor_id: uuid.UUID | str,
        *,
        limit: int = 50,
    ) -> list[OrderORM]:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        result = await session.execute(
            select(OrderORM)
            .where(OrderORM.vendor_id == vid, OrderORM.deleted_at.is_(None))
            .order_by(OrderORM.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()  # type: ignore[return-value]

    async def list_recent_by_customer(
        self,
        session: AsyncSession,
        shop_id: str,
        customer_id: str,
        *,
        limit: int = 3,
    ) -> list[OrderORM]:
        """Fetch recent orders for a customer in a shop — used by ShopBot."""
        result = await session.execute(
            select(OrderORM)
            .options(
                selectinload(OrderORM.items),
                selectinload(OrderORM.shipment),
            )
            .where(
                OrderORM.customer_id == customer_id,
                OrderORM.deleted_at.is_(None),
            )
            .order_by(OrderORM.created_at.desc())
            .limit(min(limit, 3))
        )
        return result.scalars().all()  # type: ignore[return-value]

    async def create(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> OrderORM:
        items_data = data.pop("items", [])
        order = OrderORM(**data)
        session.add(order)
        await session.flush()

        for item_data in items_data:
            item_data["order_id"] = order.id
            session.add(OrderItemORM(**item_data))

        await session.flush()
        return order

    async def update_status(
        self,
        session: AsyncSession,
        order_id: uuid.UUID | str,
        status: str,
        **extra_fields: Any,
    ) -> None:
        from sqlalchemy import update as sa_update
        oid = uuid.UUID(str(order_id)) if isinstance(order_id, str) else order_id
        values = {"status": status, **extra_fields}
        await session.execute(
            sa_update(OrderORM).where(OrderORM.id == oid).values(**values)
        )
        await session.flush()
