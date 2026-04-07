"""
ORM Models — LiveSession
========================
Sessions de live streaming — SQLAlchemy 2.0.
Les métriques temps-réel (viewers, GMV) restent dans Redis.
PostgreSQL ne garde que l'état persistant (created, ended, résumé).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class LiveSessionORM(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Table: live_sessions — sessions de diffusion live."""
    __tablename__ = "live_sessions"

    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vendors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="scheduled", index=True
    )  # scheduled | live | ended
    live_type: Mapped[str] = mapped_column(
        String(30), nullable=False, default="standard"
    )  # standard | auction | flash_sale
    pinned_product_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    scheduled_at: Mapped[datetime | None] = mapped_column(nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Métriques persistées en fin de live
    peak_viewers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_viewers: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_gmv: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    items_sold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    buy_now_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    comments_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gifts_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stream_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    replay_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[str | None] = mapped_column(Text, nullable=True)
