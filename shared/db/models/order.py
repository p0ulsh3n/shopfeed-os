"""
ORM Models — Orders, OrderItems, Cart, Shipments
=================================================
Modèles SQLAlchemy 2.0 pour le cycle de vie des commandes.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


class CartItemORM(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Table: cart_items — panier persistant par utilisateur."""
    __tablename__ = "cart_items"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
    )
    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )
    variant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(5), nullable=False, default="")

    # Snapshot du produit au moment de l'ajout au panier
    product_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class OrderORM(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """Table: orders — commandes passées."""
    __tablename__ = "orders"

    buyer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vendors.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    order_number: Mapped[str] = mapped_column(String(50), nullable=True, unique=True, index=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending", index=True)
    total_amount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    currency: Mapped[str] = mapped_column(String(5), nullable=False, default="")
    shipping_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    shipping_breakdown: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    shipping_address: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    payment_method: Mapped[str | None] = mapped_column(String(50), nullable=True)
    payment_status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    tracking_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    customer_id: Mapped[str | None] = mapped_column(String(200), nullable=True)  # email/phone

    confirmed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    shipped_at: Mapped[datetime | None] = mapped_column(nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relations
    items: Mapped[list[OrderItemORM]] = relationship(
        "OrderItemORM", back_populates="order", lazy="select", cascade="all, delete-orphan"
    )
    shipment: Mapped[ShipmentORM | None] = relationship(
        "ShipmentORM", back_populates="order", uselist=False
    )


class OrderItemORM(Base, UUIDPrimaryKeyMixin):
    """Table: order_items — lignes de commande."""
    __tablename__ = "order_items"

    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="SET NULL"),
        nullable=True,
    )
    variant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(5), nullable=False, default="")

    # Snapshot du produit (nom, photo) conservé même si produit supprimé
    product_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    order: Mapped[OrderORM] = relationship("OrderORM", back_populates="items")


class ShipmentORM(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Table: shipments — informations de livraison d'une commande."""
    __tablename__ = "shipments"

    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orders.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    tracking_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    carrier: Mapped[str | None] = mapped_column(String(50), nullable=True)
    estimated_delivery_date: Mapped[datetime | None] = mapped_column(nullable=True)
    actual_delivery_date: Mapped[datetime | None] = mapped_column(nullable=True)
    label_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    order: Mapped[OrderORM] = relationship("OrderORM", back_populates="shipment")
