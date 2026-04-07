"""
ORM Models — Analytics: VendorMetric, ProductEvent, FeedVideo
==============================================================
Stockage des métriques agrégées par Postgres.
Redis est la source temps-réel ; ces tables sont le snapshot persisté.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class VendorMetricORM(Base, TimestampMixin):
    """Table: vendor_metrics — métriques agrégées par vendor (rolling 30d)."""
    __tablename__ = "vendor_metrics"

    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vendors.id", ondelete="CASCADE"),
        primary_key=True,
    )
    gmv_total: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    impressions_30d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clicks_30d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    orders_30d: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ctr_30d: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cvr_30d: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_products: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class ProductEventCounterORM(Base, TimestampMixin):
    """Table: product_event_counters — compteurs par produit et par vendor."""
    __tablename__ = "product_event_counters"

    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        primary_key=True,
    )
    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vendors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    impressions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clicks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    purchases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gmv: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    commerce_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    pool_level: Mapped[str] = mapped_column(String(5), nullable=False, default="L1")
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="")


class FeedVideoORM(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Table: feed_videos — contenu du feed TikTok-like.
    Les compteurs Redis sont flushés ici toutes les 5 minutes
    par CounterSyncService.
    """
    __tablename__ = "feed_videos"

    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vendors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content_type: Mapped[str] = mapped_column(String(30), nullable=False, default="photo")
    pool_level: Mapped[str] = mapped_column(String(5), nullable=False, default="L1", index=True)

    # Compteurs (synchonisés depuis Redis toutes les 5 min)
    total_views: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_likes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_shares: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_add_to_cart: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_purchases: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gmv_attributed: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Scores ML
    score_cvr: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    score_watch_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    score_engagement: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
