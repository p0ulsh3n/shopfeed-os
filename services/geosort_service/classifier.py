"""
Geo Classification — Global Order Zone Classification (reverse_geocode + haversine).

Replaces the old hardcoded 30-zone system with a 100% global offline solution.
Works for ANY country in the world, no hardcoded zones needed.

How it works:
    1. App sends vendor lat/lon + buyer lat/lon (already tracked in React Native)
    2. reverse_geocode resolves coordinates → city + country (offline, <1ms)
    3. haversine calculates exact distance in km
    4. Simple logic determines zone:

    Zone A (Livraison)  → same city OR distance < 50km
    Zone B (Expedition) → same country, different city
    Zone C (Expedition) → different country (international)

Dependencies:
    pip install reverse_geocode

Architecture:
    - reverse_geocode: 100% offline after install, embedded GeoNames database
    - Covers 100,000+ cities worldwide
    - Resolution: ~1ms per lookup
    - No API calls, no internet needed at runtime
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── Earth distance (haversine) ────────────────────────────────────

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates on Earth.

    Args:
        lat1, lon1: Point A coordinates (degrees)
        lat2, lon2: Point B coordinates (degrees)

    Returns:
        Distance in kilometers (float)
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── GeoLocation result ───────────────────────────────────────────

@dataclass
class GeoLocation:
    """Resolved location from GPS coordinates."""
    city: str
    country: str
    country_code: str
    latitude: float
    longitude: float


# ── Reverse geocoding (offline, global) ──────────────────────────

def resolve_location(lat: float, lon: float) -> GeoLocation:
    """Resolve GPS coordinates to city + country using offline database.

    Uses reverse_geocode library (100,000+ cities worldwide, offline).
    If reverse_geocode is not installed, returns a minimal fallback.

    Performance: <1ms per lookup (pre-loaded in memory).
    """
    try:
        import reverse_geocode
        results = reverse_geocode.search([(lat, lon)])
        if results:
            r = results[0]
            return GeoLocation(
                city=r.get("city", ""),
                country=r.get("country", ""),
                country_code=r.get("country_code", ""),
                latitude=lat,
                longitude=lon,
            )
    except ImportError:
        logger.warning(
            "reverse_geocode not installed — pip install reverse_geocode. "
            "Using coordinate-only classification."
        )
    except Exception as e:
        logger.error("Reverse geocode failed for (%.4f, %.4f): %s", lat, lon, e)

    return GeoLocation(
        city="",
        country="",
        country_code="",
        latitude=lat,
        longitude=lon,
    )


# ── Order classification ─────────────────────────────────────────

@dataclass
class OrderClassification:
    """Result of order geo-classification."""
    zone: str                       # "A", "B", or "C"
    zone_label: str                 # "Livraison" or "Expedition"
    geo_level: str                  # "L1", "L2", "L3", "L4" (backcompat)
    distance_km: float
    buyer_city: str
    buyer_country: str
    buyer_country_code: str
    vendor_city: str
    vendor_country: str
    vendor_country_code: str
    shipping_suggestion: str
    confidence: float


# Zone A threshold — same city OR distance less than this
SAME_ZONE_RADIUS_KM = 50.0


def classify_order(
    vendor_lat: float,
    vendor_lon: float,
    buyer_lat: float,
    buyer_lon: float,
    same_zone_radius_km: float = SAME_ZONE_RADIUS_KM,
) -> OrderClassification:
    """Classify an order into Zone A, B, or C based on vendor/buyer coordinates.

    This is the main entry point. 100% global, works for any country.

    Zone logic:
        Zone A (Livraison)  → same city OR distance < same_zone_radius_km
            = delivery within the same area
        Zone B (Expedition) → different city, same country
            = national shipping
        Zone C (Expedition) → different country
            = international shipping

    Args:
        vendor_lat, vendor_lon: Vendor GPS coordinates
        buyer_lat, buyer_lon: Buyer GPS coordinates
        same_zone_radius_km: Maximum distance for Zone A (default 50km)

    Returns:
        OrderClassification with zone, distance, cities, and shipping suggestion
    """
    # 1. Resolve both locations (offline, <1ms each)
    vendor_loc = resolve_location(vendor_lat, vendor_lon)
    buyer_loc = resolve_location(buyer_lat, buyer_lon)

    # 2. Calculate exact distance
    distance = haversine_km(vendor_lat, vendor_lon, buyer_lat, buyer_lon)
    distance = round(distance, 1)

    # 3. Classify zone
    same_country = _same_country(vendor_loc, buyer_loc)
    same_city = _same_city(vendor_loc, buyer_loc)

    if same_city or distance <= same_zone_radius_km:
        # Zone A — Livraison (same city or very close)
        zone = "A"
        zone_label = "Livraison"
        geo_level = "L1" if distance < 10 else "L2"
        confidence = 0.99 if same_city and distance < 20 else 0.95
        shipping = _suggest_shipping(distance, "local")

    elif same_country:
        # Zone B — Expedition nationale (different city, same country)
        zone = "B"
        zone_label = "Expedition"
        geo_level = "L3"
        confidence = 0.98
        shipping = _suggest_shipping(distance, "national")

    else:
        # Zone C — Expedition internationale (different country)
        zone = "C"
        zone_label = "Expedition"
        geo_level = "L4"
        confidence = 0.99
        shipping = _suggest_shipping(distance, "international")

    return OrderClassification(
        zone=zone,
        zone_label=zone_label,
        geo_level=geo_level,
        distance_km=distance,
        buyer_city=buyer_loc.city,
        buyer_country=buyer_loc.country,
        buyer_country_code=buyer_loc.country_code,
        vendor_city=vendor_loc.city,
        vendor_country=vendor_loc.country,
        vendor_country_code=vendor_loc.country_code,
        shipping_suggestion=shipping,
        confidence=confidence,
    )


# ── Comparison helpers ───────────────────────────────────────────

def _same_country(a: GeoLocation, b: GeoLocation) -> bool:
    """Check if two locations are in the same country."""
    if a.country_code and b.country_code:
        return a.country_code.upper() == b.country_code.upper()
    if a.country and b.country:
        return a.country.lower() == b.country.lower()
    return False


def _same_city(a: GeoLocation, b: GeoLocation) -> bool:
    """Check if two locations are in the same city."""
    if not a.city or not b.city:
        return False
    return a.city.lower().strip() == b.city.lower().strip()


# ── Shipping suggestion (100% distance-based, no hardcoded countries) ──

# Estimated delivery speed thresholds (km)
_SPEED_TIERS = {
    "express":  5,      # < 5km → express
    "fast":     20,     # < 20km → fast local
    "local":    50,     # < 50km → standard local
    "regional": 200,    # < 200km → regional
    "national": 800,    # < 800km → standard national
    "distant":  3000,   # < 3000km → continental
}


def _suggest_shipping(distance_km: float, scope: str) -> str:
    """Suggest shipping method based purely on distance and scope.

    scope: "local", "national", "international"

    No country codes. Works for any location in the world.
    The estimated delivery windows are based on distance alone.
    """
    d = distance_km

    if scope == "local":
        # Zone A — same city or < 50km
        if d < _SPEED_TIERS["express"]:
            return f"Express delivery ({d:.1f}km, < 1h)"
        if d < _SPEED_TIERS["fast"]:
            return f"Fast delivery ({d:.1f}km, 1-3h)"
        return f"Standard local delivery ({d:.1f}km, same day)"

    if scope == "national":
        # Zone B — same country, different city
        if d < _SPEED_TIERS["regional"]:
            return f"Regional shipping ({d:.0f}km, 1-2 days)"
        if d < _SPEED_TIERS["national"]:
            return f"National shipping ({d:.0f}km, 2-3 days)"
        return f"Long-distance national shipping ({d:.0f}km, 3-5 days)"

    # scope == "international" — Zone C
    if d < _SPEED_TIERS["distant"]:
        return f"International shipping ({d:.0f}km, 3-5 days)"
    return f"International shipping ({d:.0f}km, 5-10 days)"
