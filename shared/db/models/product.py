"""
ORM Models — Products, ProductVariants
========================================
Modèles SQLAlchemy 2.0. Les données shipping/photos sont stockées
en JSONB pour flexibilité sans sur-normalisation.
"""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDPrimaryKeyMixin


class ProductORM(Base, UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin):
    """Table: products — catalogue produits."""
    __tablename__ = "products"

    vendor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vendors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    platform_sku: Mapped[str] = mapped_column(String(100), nullable=False, default="", index=True)
    vendor_sku: Mapped[str | None] = mapped_column(String(100), nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description_short: Mapped[str] = mapped_column(String(280), nullable=False, default="")
    description_full: Mapped[str] = mapped_column(Text, nullable=False, default="")
    category_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    subcategory_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    brand: Mapped[str | None] = mapped_column(String(100), nullable=True)
    base_price: Mapped[float] = mapped_column(Float, nullable=False)
    compare_at_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(5), nullable=False, default="")
    base_stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    has_variants: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Media & ML — stockés en JSONB
    photos: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    video_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    clip_embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    cv_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Attributes
    attributes: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    tags: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Shipping config en JSONB
    weight_g: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dimensions_cm: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    processing_days: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    shipping_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    return_policy: Mapped[str] = mapped_column(String(30), nullable=False, default="free_30d")

    # Status & pool
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="draft", index=True)
    pool_level: Mapped[str] = mapped_column(String(5), nullable=False, default="L1", index=True)
    freshness_boost_until: Mapped[str | None] = mapped_column(nullable=True)

    # Flash sale
    flash_sale_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    flash_sale_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    flash_sale_ends_at: Mapped[str | None] = mapped_column(nullable=True)

    published_at: Mapped[str | None] = mapped_column(nullable=True)

    # Relations
    vendor: Mapped[VendorORM] = relationship("VendorORM", back_populates="products")
    variants: Mapped[list[ProductVariantORM]] = relationship(
        "ProductVariantORM", back_populates="product", lazy="select", cascade="all, delete-orphan"
    )


class ProductVariantORM(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """Table: product_variants — SKUs (couleur × taille, etc.)."""
    __tablename__ = "product_variants"

    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    variant_sku: Mapped[str] = mapped_column(String(100), nullable=False)
    type1_value: Mapped[str | None] = mapped_column(String(100), nullable=True)
    type2_value: Mapped[str | None] = mapped_column(String(100), nullable=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    compare_at_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    barcode: Mapped[str | None] = mapped_column(String(100), nullable=True)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    product: Mapped[ProductORM] = relationship("ProductORM", back_populates="variants")


from shared.db.models.vendor import VendorORM  # noqa: E402
