"""Vendor models — Section 06 (Tiers & Account Weight)."""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class VendorTier(str, enum.Enum):
    """Seller tier — determines Account Weight multiplier. Section 06."""
    BRONZE = "bronze"       # ×1.0  — new vendor (<100 sales)
    SILVER = "silver"       # ×1.3  — 100-500 sales, rating ≥ 4.0
    GOLD = "gold"           # ×1.7  — 500-5K sales, rating ≥ 4.3
    PLATINUM = "platinum"   # ×2.5  — 5K+ sales, rating ≥ 4.6


# Mapping tier → Account Weight multiplier
TIER_WEIGHTS: dict[VendorTier, float] = {
    VendorTier.BRONZE: 1.0,
    VendorTier.SILVER: 1.3,
    VendorTier.GOLD: 1.7,
    VendorTier.PLATINUM: 2.5,
}


class Vendor(BaseModel):
    """Vendor record — Section 06 / 40."""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID                           # FK → users
    shop_name: str = Field(..., max_length=100)
    description: str = ""
    logo_url: str | None = None
    banner_url: str | None = None

    # Tier system
    tier: VendorTier = VendorTier.BRONZE
    account_weight: float = 1.0             # ×1.0 to ×2.5 — recalculated weekly

    # Performance metrics (rolling windows)
    cvr_30d: float = 0.0                    # Conversion rate 30d
    avg_rating: float = 0.0
    total_sales: int = 0
    on_time_delivery_rate: float = 1.0
    publication_freq: float = 0.0           # Publications per week

    is_verified: bool = False

    # Geo (resolved from GPS coordinates at signup)
    geo_commune: str | None = None
    geo_city: str | None = None
    geo_country: str = ""                   # ISO 3166 code — set by app
    geo_lat: float | None = None
    geo_lon: float | None = None

    live_enabled: bool = False
    stripe_account_id: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def weight_multiplier(self) -> float:
        return TIER_WEIGHTS.get(self.tier, 1.0)

    def compute_account_weight(self) -> float:
        """Recompute account weight from performance features. Section 06.

        Features:
            CVR (35%) + Rating (25%) + Pub Freq (15%) +
            On-time Delivery (15%) + Content Engagement (10%)
        """
        # Normalize each feature to [0, 1]
        cvr_norm = min(self.cvr_30d / 0.10, 1.0)       # 10% CVR = max
        rating_norm = min(self.avg_rating / 5.0, 1.0)
        freq_norm = min(self.publication_freq / 7.0, 1.0)  # 7/week = max
        delivery_norm = self.on_time_delivery_rate

        raw_score = (
            cvr_norm * 0.35
            + rating_norm * 0.25
            + freq_norm * 0.15
            + delivery_norm * 0.15
            + 0.5 * 0.10   # engagement placeholder
        )

        # Map [0,1] → tier weight range [1.0, 2.5]
        self.account_weight = 1.0 + raw_score * (self.weight_multiplier - 1.0)
        return self.account_weight
