"""
SQLAlchemy 2.0 — Declarative Base + Shared Mixins
===================================================
Unique source de vérité pour tous les modèles ORM du projet.
Importez toujours depuis ici : `from shared.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin`
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base ORM commune — tous les modèles héritent de cette classe."""
    pass


class UUIDPrimaryKeyMixin:
    """PK UUID v4 auto-générée côté Python (pas côté DB)."""
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )


class TimestampMixin:
    """Colonnes created_at / updated_at auto-gérées par SQLAlchemy."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """Soft delete — deleted_at IS NULL pour les requêtes normales."""
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
