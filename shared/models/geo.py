"""Geo classification models — Section 43 (GeoSort)."""

from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GeoLevel(str, enum.Enum):
    """Geographic classification level — Section 43."""
    L1 = "L1"   # Same commune         🟢
    L2 = "L2"   # Same city            🟡
    L3 = "L3"   # Same country         🟠
    L4 = "L4"   # International        🔴


class ClassificationMethod(str, enum.Enum):
    GEO_HIERARCHY = "geo_hierarchy"     # PostGIS lookup — <5ms
    KMEANS = "kmeans"                   # GPS clustering — batch
    NLP = "nlp"                         # Informal address parsing — 50-200ms
    MANUAL = "manual"                   # Human override




class OrderGeoClassification(BaseModel):
    """Result of geo classification for an order — Section 43."""
    id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    vendor_id: UUID

    buyer_country: str = ""
    buyer_city: str = ""
    buyer_commune: str = ""
    buyer_lat: float | None = None
    buyer_lon: float | None = None

    vendor_commune: str = ""

    geo_level: GeoLevel = GeoLevel.L4
    geo_cluster_id: int | None = None
    distance_km: float = 0.0

    classification_method: ClassificationMethod = ClassificationMethod.GEO_HIERARCHY
    confidence: float = 1.0
    is_outlier: bool = False

    shipping_type_suggested: str = ""
    classified_at: datetime = Field(default_factory=datetime.utcnow)
