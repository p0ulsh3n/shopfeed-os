"""
Repository — Vendors
=======================
CRUD SQLAlchemy 2.0 pour les boutiques vendeurs.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.models.vendor import VendorORM


class VendorRepository:
    """CRUD repository pour les vendeurs."""

    async def get_by_id(
        self, session: AsyncSession, vendor_id: uuid.UUID | str
    ) -> VendorORM | None:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        result = await session.execute(
            select(VendorORM)
            .where(VendorORM.id == vid, VendorORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_by_user_id(
        self, session: AsyncSession, user_id: uuid.UUID | str
    ) -> VendorORM | None:
        uid = uuid.UUID(str(user_id)) if isinstance(user_id, str) else user_id
        result = await session.execute(
            select(VendorORM)
            .where(VendorORM.user_id == uid, VendorORM.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def create(self, session: AsyncSession, data: dict[str, Any]) -> VendorORM:
        vendor = VendorORM(**data)
        session.add(vendor)
        await session.flush()
        return vendor

    async def update(
        self,
        session: AsyncSession,
        vendor_id: uuid.UUID | str,
        updates: dict[str, Any],
    ) -> VendorORM | None:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        await session.execute(
            update(VendorORM)
            .where(VendorORM.id == vid)
            .values(**updates)
        )
        await session.flush()
        return await self.get_by_id(session, vid)

    async def soft_delete(
        self, session: AsyncSession, vendor_id: uuid.UUID | str
    ) -> None:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        await session.execute(
            update(VendorORM)
            .where(VendorORM.id == vid)
            .values(deleted_at=datetime.now(timezone.utc))
        )
        await session.flush()
