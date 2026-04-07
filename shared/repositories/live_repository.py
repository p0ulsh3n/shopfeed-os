"""
Repository — LiveSessions
===========================
CRUD SQLAlchemy 2.0 pour les sessions live.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.models.live_session import LiveSessionORM


class LiveSessionRepository:
    """Repository pour les sessions de live streaming."""

    async def get_by_id(
        self, session: AsyncSession, live_id: uuid.UUID | str
    ) -> LiveSessionORM | None:
        lid = uuid.UUID(str(live_id)) if isinstance(live_id, str) else live_id
        result = await session.execute(
            select(LiveSessionORM).where(LiveSessionORM.id == lid)
        )
        return result.scalar_one_or_none()

    async def list_active_by_vendor(
        self, session: AsyncSession, vendor_id: uuid.UUID | str
    ) -> list[LiveSessionORM]:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        result = await session.execute(
            select(LiveSessionORM)
            .where(
                LiveSessionORM.vendor_id == vid,
                LiveSessionORM.status.in_(["live", "scheduled"]),
            )
            .order_by(LiveSessionORM.created_at.desc())
        )
        return result.scalars().all()  # type: ignore[return-value]

    async def create(
        self, session: AsyncSession, data: dict[str, Any]
    ) -> LiveSessionORM:
        live = LiveSessionORM(**data)
        session.add(live)
        await session.flush()
        return live

    async def update(
        self,
        session: AsyncSession,
        live_id: uuid.UUID | str,
        updates: dict[str, Any],
    ) -> None:
        lid = uuid.UUID(str(live_id)) if isinstance(live_id, str) else live_id
        await session.execute(
            update(LiveSessionORM).where(LiveSessionORM.id == lid).values(**updates)
        )
        await session.flush()
