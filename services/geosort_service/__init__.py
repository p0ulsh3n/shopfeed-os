"""GeoSort Service — Geographic Order Classification — Section 43."""

from .classifier import (
    GEO_ZONES,
    GeoZoneEntry,
    haversine_km,
    nlp_geocode_address,
    suggest_shipping,
)
from .routes import app
from .schemas import ClassifyRequest, ClassifyResponse

__all__ = [
    "app",
    "ClassifyRequest",
    "ClassifyResponse",
    "GeoZoneEntry",
    "GEO_ZONES",
    "haversine_km",
    "nlp_geocode_address",
    "suggest_shipping",
]
