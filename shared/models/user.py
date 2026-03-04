"""User & buyer profile models — Section 05 (Intent Graph)."""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Persona(str, enum.Enum):
    """Buyer persona — inferred from behaviour. Section 05."""
    IMPULSE_BUYER = "impulse_buyer"
    RESEARCHER = "researcher"
    PRICE_HUNTER = "price_hunter"
    QUALITY_SEEKER = "quality_seeker"
    UNKNOWN = "unknown"


class IntentLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BUYING_NOW = "buying_now"
    CHECKOUT = "checkout"


class User(BaseModel):
    """Core user record — Section 40."""
    id: UUID = Field(default_factory=uuid4)
    email: str | None = None
    phone: str | None = None
    full_name: str = ""
    avatar_url: str | None = None
    date_of_birth: str | None = None        # ISO date
    gender: str | None = None               # m / f / other / prefer_not
    city: str | None = None
    commune: str | None = None
    country: str = "CI"                     # ISO 3166
    lat: float | None = None
    lon: float | None = None
    persona: Persona = Persona.UNKNOWN
    loyalty_points: int = 0
    is_verified: bool = False
    fcm_token: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserProfile(BaseModel):
    """ML feature-store representation of a buyer — Section 05 / 42.

    This is the "Intent Graph": a dynamic map of purchase intent
    across categories, price ranges, and temporal signals.
    """
    user_id: UUID

    # Long-term preference tree: {mode: {femme: 0.87}, beaute: 0.6, ...}
    category_prefs: dict[str, float] = Field(default_factory=dict)

    # Per-category price tolerance: {mode: {min: 15, max: 80}, ...}
    price_ranges: dict[str, dict[str, float]] = Field(default_factory=dict)

    intent_level: IntentLevel = IntentLevel.LOW

    # Short-term interests (48h window)
    active_categories: list[str] = Field(default_factory=list)

    # Two-Tower user embedding (256-dim) — batch-computed daily
    embedding: list[float] | None = None

    # Purchase patterns
    purchase_frequency: float = 0.0         # purchases / 30d
    avg_order_value: float = 0.0

    # Top vendors by affinity (rolling 90d)
    top_vendors: list[UUID] = Field(default_factory=list)

    persona: Persona = Persona.UNKNOWN

    # Geo
    geo_cluster: int = 0                    # K-Means cluster

    last_active_at: datetime | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
