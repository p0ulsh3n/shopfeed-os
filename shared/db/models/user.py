"""
ORM Models — Users, UserProfile, UserFollow
============================================
Modèles SQLAlchemy 2.0 qui correspondent aux tables PostgreSQL.
Alembic génère les migrations depuis ces modèles.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


class UserORM(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """Table: users — comptes acheteurs et vendeurs."""
    __tablename__ = "users"

    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True, index=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(30), unique=True, nullable=True)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    date_of_birth: Mapped[str | None] = mapped_column(String(20), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    commune: Mapped[str | None] = mapped_column(String(100), nullable=True)
    country: Mapped[str] = mapped_column(String(5), nullable=False, default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    persona: Mapped[str] = mapped_column(String(30), nullable=False, default="unknown")
    loyalty_points: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fcm_token: Mapped[str | None] = mapped_column(String(500), nullable=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="buyer")

    # Relations
    profile: Mapped[UserProfileORM | None] = relationship(
        "UserProfileORM", back_populates="user", uselist=False, lazy="select"
    )
    vendor: Mapped[VendorORM | None] = relationship(
        "VendorORM", back_populates="user", uselist=False, lazy="select"
    )
    follows: Mapped[list[UserFollowORM]] = relationship(
        "UserFollowORM", foreign_keys="UserFollowORM.user_id", back_populates="follower"
    )


class UserProfileORM(Base, TimestampMixin):
    """Table: user_profiles — Intent Graph ML features."""
    __tablename__ = "user_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    category_prefs: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    price_ranges: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    intent_level: Mapped[str] = mapped_column(String(20), nullable=False, default="low")
    active_categories: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    purchase_frequency: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_order_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    top_vendors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    persona: Mapped[str] = mapped_column(String(30), nullable=False, default="unknown")
    geo_cluster: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_active_at: Mapped[datetime | None] = mapped_column(nullable=True)

    user: Mapped[UserORM] = relationship("UserORM", back_populates="profile")


class UserFollowORM(Base, TimestampMixin):
    """Table: user_follows — qui suit quel vendeur."""
    __tablename__ = "user_follows"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vendors.id", ondelete="CASCADE"),
        primary_key=True,
    )

    follower: Mapped[UserORM] = relationship(
        "UserORM", foreign_keys=[user_id], back_populates="follows"
    )


# Import circulaire évité — VendorORM importé en bas
from shared.db.models.vendor import VendorORM  # noqa: E402
