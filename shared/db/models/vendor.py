"""
ORM Models — Vendors
=====================
Modèle SQLAlchemy 2.0 pour les vendeurs (boutiques).
"""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


class VendorORM(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """Table: vendors — boutiques des vendeurs."""
    __tablename__ = "vendors"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    shop_name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    logo_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    banner_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Tier & account weight
    tier: Mapped[str] = mapped_column(String(20), nullable=False, default="bronze")
    account_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    # Performance metrics
    cvr_30d: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_rating: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_sales: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    on_time_delivery_rate: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    publication_freq: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    live_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    stripe_account_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Géolocalisation
    geo_commune: Mapped[str | None] = mapped_column(String(100), nullable=True)
    geo_city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    geo_country: Mapped[str] = mapped_column(String(5), nullable=False, default="")
    geo_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    geo_lon: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relations
    user: Mapped[UserORM] = relationship("UserORM", back_populates="vendor")
    products: Mapped[list[ProductORM]] = relationship(
        "ProductORM", back_populates="vendor", lazy="select"
    )


# Éviter les imports circulaires
from shared.db.models.user import UserORM  # noqa: E402
from shared.db.models.product import ProductORM  # noqa: E402
