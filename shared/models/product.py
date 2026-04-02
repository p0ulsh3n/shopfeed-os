"""Product models — Tables: products, product_variants, product_content."""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ContentType(str, enum.Enum):
    PHOTO = "photo"
    CAROUSEL = "carousel"
    VIDEO = "video"
    LIVE = "live"
    LIVE_SCHEDULED = "live_scheduled"


class PoolLevel(str, enum.Enum):
    """Traffic pool level (流量池). Section 03."""
    L1 = "L1"  # Cold Start — 200-800 views
    L2 = "L2"  # Category Niche — 1K-5K
    L3 = "L3"  # Intent Similar — 5K-30K
    L4 = "L4"  # Amplification — 30K-200K
    L5 = "L5"  # Trending Viral — 200K-2M
    L6 = "L6"  # Best Seller Flash — 2M+


class ProductStatus(str, enum.Enum):
    DRAFT = "draft"
    PENDING_MODERATION = "pending_moderation"
    ACTIVE = "active"
    PAUSED = "paused"
    DELETED = "deleted"


class ReturnPolicy(str, enum.Enum):
    FREE_30D = "free_30d"
    PAID_15D = "paid_15d"
    NO_RETURNS = "no_returns"
    CUSTOM = "custom"


# ──────────────────────────────────────────────────────────────
# Product Variant
# ──────────────────────────────────────────────────────────────

class ProductVariant(BaseModel):
    """Single SKU variant (color × size, etc.)."""
    id: UUID = Field(default_factory=uuid4)
    product_id: UUID
    variant_sku: str
    type1_value: str | None = None          # "Blanc"
    type2_value: str | None = None          # "38"
    price: float
    compare_at_price: float | None = None
    stock: int = 0
    barcode: str | None = None
    image_url: str | None = None
    is_active: bool = True


# ──────────────────────────────────────────────────────────────
# Product (main table)
# ──────────────────────────────────────────────────────────────

class PhotoMeta(BaseModel):
    url: str
    cv_score: float | None = None
    clip_embedding: list[float] | None = None  # 512-dim
    caption: str | None = None


class ShippingRateTier(BaseModel):
    """Weight tier within a shipping zone rate.

    Example: 0-2000g → base_price 1500 FCFA, then +500/kg above.
    """
    min_weight_g: int = 0               # Minimum weight (grams, inclusive)
    max_weight_g: int = 2000            # Weight covered by base_price
    base_price: float = 0.0             # Base price for this tier
    price_per_extra_kg: float = 0.0     # Price per additional kg above max_weight_g


class ShippingZoneRate(BaseModel):
    """Vendor-defined rate for one geographic zone (A, B, or C).

    Zone A = same city (Livraison)
    Zone B = same country, different city (Expedition nationale)
    Zone C = different country (Expedition internationale)

    Vendors configure these in their dashboard.
    """
    zone: str                           # "A", "B", or "C"
    tiers: list[ShippingRateTier] = Field(
        default_factory=lambda: [ShippingRateTier()],
    )
    free_above: float | None = None     # Free shipping if order > this amount
    enabled: bool = True                # Does vendor ship to this zone?


class ShippingConfig(BaseModel):
    """Vendor shipping configuration.

    Each vendor defines their own rates per zone.
    If zone_rates is empty, platform defaults apply.

    Example JSON a vendor would configure:
    {
        "zone_rates": [
            {"zone": "A", "tiers": [{"max_weight_g": 2000, "base_price": 1500, "price_per_extra_kg": 500}], "free_above": 50000},
            {"zone": "B", "tiers": [{"max_weight_g": 1000, "base_price": 3000, "price_per_extra_kg": 1000}]},
            {"zone": "C", "tiers": [{"max_weight_g": 500, "base_price": 25000, "price_per_extra_kg": 5000}]}
        ],
        "package_weight_g": 100,
        "processing_days": 3
    }
    """
    zone_rates: list[ShippingZoneRate] = Field(default_factory=list)
    package_weight_g: int = 100         # Weight of packaging added to each shipment
    processing_days: int = 3            # Days to prepare the shipment


class Product(BaseModel):
    """Full product record — Section 39/40."""
    id: UUID = Field(default_factory=uuid4)
    vendor_id: UUID
    platform_sku: str = ""
    vendor_sku: str | None = None
    title: str = Field(..., min_length=10, max_length=200)
    description_short: str = Field(default="", max_length=280)
    description_full: str = ""
    category_id: int = 0
    subcategory_id: int | None = None
    brand: str | None = None
    base_price: float
    compare_at_price: float | None = None
    currency: str = ""                      # ISO 4217 — resolved from vendor GPS
    base_stock: int = 0
    has_variants: bool = False

    # Media
    photos: list[PhotoMeta] = Field(default_factory=list)
    video_url: str | None = None
    clip_embedding: list[float] | None = None  # 512-dim, main photo
    cv_score: float | None = None
    ai_description: str | None = None

    # Attributes
    attributes: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    # Shipping
    weight_g: int | None = None
    dimensions_cm: dict | None = None       # {L, l, H}
    processing_days: int = 3
    shipping: ShippingConfig = Field(default_factory=ShippingConfig)
    return_policy: ReturnPolicy = ReturnPolicy.FREE_30D

    # Compliance
    compliance_docs: dict | None = None

    # Status & Pool
    status: ProductStatus = ProductStatus.DRAFT
    pool_level: PoolLevel = PoolLevel.L1
    freshness_boost_until: datetime | None = None

    # Flash Sale
    flash_sale_active: bool = False
    flash_sale_price: float | None = None
    flash_sale_ends_at: datetime | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: datetime | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────
# Product Content (feed item — what the ML ranks)
# ──────────────────────────────────────────────────────────────

class ProductContent(BaseModel):
    """Feed-level content item — one product can have multiple content pieces."""
    id: UUID = Field(default_factory=uuid4)
    product_id: UUID
    vendor_id: UUID
    content_type: ContentType = ContentType.PHOTO
    media_urls: list[dict] = Field(default_factory=list)    # [{url, type, order}]

    # Pool & metrics
    pool_level: PoolLevel = PoolLevel.L1
    impressions: int = 0
    clicks: int = 0
    add_to_carts: int = 0
    purchases: int = 0
    buy_now_count: int = 0
    shares: int = 0
    saves: int = 0
    skip_rate: float = 0.0
    cvr: float = 0.0                       # Recalculated hourly

    # ML
    embedding: list[float] | None = None    # 512-dim content embedding

    live_session_id: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
