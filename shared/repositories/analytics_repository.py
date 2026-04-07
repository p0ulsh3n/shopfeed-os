"""
Repository — Analytics: VendorMetrics, ProductEventCounters, FeedVideos
=========================================================================
CRUD SQLAlchemy 2.0. Zéro SQL brut.
Utilisé par CounterSyncService pour le flush Redis → PostgreSQL.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.models.analytics import (
    FeedVideoORM,
    ProductEventCounterORM,
    VendorMetricORM,
)


class AnalyticsRepository:
    """Repository pour les métriques analytics."""

    # ───── Vendor Metrics ────────────────────────────────────────

    async def upsert_vendor_metrics(
        self,
        session: AsyncSession,
        vendor_id: uuid.UUID | str,
        data: dict[str, Any],
    ) -> None:
        """Upsert atomique des métriques vendor — PostgreSQL ON CONFLICT DO UPDATE."""
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        stmt = (
            pg_insert(VendorMetricORM)
            .values(vendor_id=vid, **data)
            .on_conflict_do_update(
                index_elements=["vendor_id"],
                set_=data,
            )
        )
        await session.execute(stmt)
        await session.flush()

    async def get_vendor_metrics(
        self, session: AsyncSession, vendor_id: uuid.UUID | str
    ) -> VendorMetricORM | None:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        result = await session.execute(
            select(VendorMetricORM).where(VendorMetricORM.vendor_id == vid)
        )
        return result.scalar_one_or_none()

    # ───── Product Event Counters ─────────────────────────────────

    async def upsert_product_counter(
        self,
        session: AsyncSession,
        product_id: uuid.UUID | str,
        vendor_id: uuid.UUID | str,
        data: dict[str, Any],
    ) -> None:
        pid = uuid.UUID(str(product_id)) if isinstance(product_id, str) else product_id
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        stmt = (
            pg_insert(ProductEventCounterORM)
            .values(product_id=pid, vendor_id=vid, **data)
            .on_conflict_do_update(
                index_elements=["product_id"],
                set_=data,
            )
        )
        await session.execute(stmt)
        await session.flush()

    async def list_product_counters(
        self,
        session: AsyncSession,
        vendor_id: uuid.UUID | str,
        *,
        limit: int = 50,
    ) -> list[ProductEventCounterORM]:
        vid = uuid.UUID(str(vendor_id)) if isinstance(vendor_id, str) else vendor_id
        result = await session.execute(
            select(ProductEventCounterORM)
            .where(ProductEventCounterORM.vendor_id == vid)
            .order_by(ProductEventCounterORM.gmv.desc())
            .limit(limit)
        )
        return result.scalars().all()  # type: ignore[return-value]

    # ───── Feed Video Counters (Redis → PostgreSQL sync) ──────────

    async def upsert_feed_video_counters(
        self,
        session: AsyncSession,
        content_id: uuid.UUID | str,
        data: dict[str, Any],
    ) -> None:
        """
        Flush des compteurs Redis vers la table feed_videos.
        Appelé par CounterSyncService toutes les 5 minutes.
        Remplace le SQL brut asyncpg $1,$2... de realtime_counters.py.
        """
        cid = uuid.UUID(str(content_id)) if isinstance(content_id, str) else content_id
        stmt = (
            pg_insert(FeedVideoORM)
            .values(id=cid, **data)
            .on_conflict_do_update(
                index_elements=["id"],
                set_=data,
            )
        )
        await session.execute(stmt)
        await session.flush()
