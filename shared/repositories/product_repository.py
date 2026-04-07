"""
Repository — Products & ProductVariants
=========================================
CRUD SQLAlchemy 2.0. Aucun SQL brut.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.db.models.product import ProductORM, ProductVariantORM


class ProductRepository:
    """CRUD repository pour les produits."""

    async def get_by_id(
        self, session: AsyncSession, product_id: uuid.UUID | str
    ) -> ProductORM | None:
        pid = uuid.UUID(str(product_id)) if isinstance(product_id, str) else product_id
        result = await session.execute(
            select(ProductORM)
            .options(selectinload(ProductORM.variants))
            .where(ProductORM.id == pid, ProductORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def list_by_vendor(
        self,
        session: AsyncSession,
        vendor_id: uuid.UUID | str,
        *,
        status: str | None = None,
        category_id: int | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[ProductORM], int]:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id

        q = (
            select(ProductORM)
            .where(ProductORM.vendor_id == vid, ProductORM.deleted_at.is_(None))
        )
        if status:
            q = q.where(ProductORM.status == status)
        if category_id is not None:
            q = q.where(ProductORM.category_id == category_id)

        count_q = q.with_only_columns(ProductORM.id)  # type: ignore[arg-type]
        total = len((await session.execute(count_q)).fetchall())

        result = await session.execute(q.offset(offset).limit(limit))
        return result.scalars().all(), total  # type: ignore[return-value]

    async def create(self, session: AsyncSession, data: dict[str, Any]) -> ProductORM:
        product = ProductORM(**data)
        session.add(product)
        await session.flush()
        return product

    async def update(
        self,
        session: AsyncSession,
        product_id: uuid.UUID | str,
        updates: dict[str, Any],
    ) -> ProductORM | None:
        pid = uuid.UUID(str(product_id)) if isinstance(product_id, str) else product_id
        await session.execute(
            update(ProductORM)
            .where(ProductORM.id == pid)
            .values(**updates)
        )
        await session.flush()
        return await self.get_by_id(session, pid)

    async def soft_delete(
        self, session: AsyncSession, product_id: uuid.UUID | str
    ) -> None:
        pid = uuid.UUID(str(product_id)) if isinstance(product_id, str) else product_id
        await session.execute(
            update(ProductORM)
            .where(ProductORM.id == pid)
            .values(deleted_at=datetime.now(timezone.utc), status="deleted")
        )
        await session.flush()


class ProductVariantRepository:
    """CRUD pour les variantes produit."""

    async def list_by_product(
        self, session: AsyncSession, product_id: uuid.UUID | str
    ) -> list[ProductVariantORM]:
        pid = uuid.UUID(str(product_id)) if isinstance(product_id, str) else product_id
        result = await session.execute(
            select(ProductVariantORM).where(ProductVariantORM.product_id == pid)
        )
        return result.scalars().all()  # type: ignore[return-value]

    async def create(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> ProductVariantORM:
        variant = ProductVariantORM(**data)
        session.add(variant)
        await session.flush()
        return variant
