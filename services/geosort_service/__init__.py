"""GeoSort Service — Geographic Order Classification — Section 43.

Classifies orders into geo levels (L1-L4) in <500ms via Kafka consumer.
Pushes classification to vendor dashboard via WebSocket in real-time.

4 Classification Methods:
    1. Geo Hierarchy Lookup — PostGIS, <5ms
    2. K-Means on GPS — batch clustering
    3. NLP Geocoder — informal African addresses, 50-200ms
    4. DBSCAN — outlier/fraud detection, batch
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from shared.models.geo import ClassificationMethod, GeoLevel

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — GeoSort Service", version="1.0.0")


# ──────────────────────────────────────────────────────────────
# Haversine distance
# ──────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth."""
    R = 6371.0  # Earth radius km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ──────────────────────────────────────────────────────────────
# Geo Zone Registry (pre-loaded at startup)
# ──────────────────────────────────────────────────────────────

@dataclass
class GeoZoneEntry:
    id: int
    country_code: str
    city: str
    commune: str
    aliases: list[str]
    center_lat: float
    center_lon: float


# Pre-loaded zones — Section 43 table
_GEO_ZONES: list[GeoZoneEntry] = [
    # Côte d'Ivoire — Abidjan communes
    GeoZoneEntry(1, "CI", "Abidjan", "Cocody", ["cocodie", "cocody angré", "2 plateaux", "riviera"], 5.3411, -3.9810),
    GeoZoneEntry(2, "CI", "Abidjan", "Marcory", ["marcory zone 4", "marcory résidentiel"], 5.3050, -3.9860),
    GeoZoneEntry(3, "CI", "Abidjan", "Plateau", ["le plateau", "plateau centre"], 5.3200, -4.0200),
    GeoZoneEntry(4, "CI", "Abidjan", "Abobo", ["abobo gare", "abobo pk18"], 5.4300, -4.0200),
    GeoZoneEntry(5, "CI", "Abidjan", "Yopougon", ["yop", "yopougon maroc", "yopougon wassakara"], 5.3590, -4.0780),
    GeoZoneEntry(6, "CI", "Abidjan", "Treichville", ["treich", "treichville marché"], 5.2940, -3.9940),
    GeoZoneEntry(7, "CI", "Abidjan", "Adjamé", ["adjamé gare"], 5.3530, -4.0200),
    GeoZoneEntry(8, "CI", "Abidjan", "Koumassi", ["koumassi nord", "koumassi remblais"], 5.2950, -3.9570),
    GeoZoneEntry(9, "CI", "Abidjan", "Port-Bouët", ["port bouet", "vridi"], 5.2560, -3.9610),
    GeoZoneEntry(10, "CI", "Abidjan", "Attécoubé", ["attecoube", "atté"], 5.3350, -4.0400),
    # Other CI cities
    GeoZoneEntry(11, "CI", "Bouaké", "Bouaké Centre", ["bouake"], 7.6900, -5.0300),
    GeoZoneEntry(12, "CI", "Yamoussoukro", "Yamoussoukro", ["yakro"], 6.8200, -5.2800),
    # France
    GeoZoneEntry(20, "FR", "Paris", "1er", ["paris 1", "les halles", "chatelet"], 48.8600, 2.3470),
    GeoZoneEntry(21, "FR", "Paris", "11ème", ["paris 11", "oberkampf", "bastille"], 48.8590, 2.3780),
    GeoZoneEntry(22, "FR", "Paris", "18ème", ["paris 18", "montmartre"], 48.8930, 2.3440),
    GeoZoneEntry(23, "FR", "Lyon", "Lyon 1er", ["presqu'île", "lyon centre"], 45.7676, 4.8344),
    GeoZoneEntry(24, "FR", "Marseille", "1er", ["vieux port", "marseille centre"], 43.2965, 5.3698),
    # Sénégal
    GeoZoneEntry(30, "SN", "Dakar", "Plateau", ["plateau dakar"], 14.6937, -17.4441),
    GeoZoneEntry(31, "SN", "Dakar", "Almadies", ["almadies"], 14.7306, -17.5097),
    GeoZoneEntry(32, "SN", "Dakar", "Parcelles Assainies", ["PA", "parcelles"], 14.7620, -17.4320),
    # Cameroun
    GeoZoneEntry(40, "CM", "Douala", "Douala 1er", ["akwa", "bonanjo"], 4.0511, 9.7679),
    GeoZoneEntry(41, "CM", "Yaoundé", "Yaoundé 1er", ["centre ville yaounde"], 3.8480, 11.5023),
    # Belgique
    GeoZoneEntry(50, "BE", "Bruxelles", "Ixelles", ["ixelles"], 50.8275, 4.3745),
    GeoZoneEntry(51, "BE", "Bruxelles", "Molenbeek", ["molenbeek saint jean"], 50.8561, 4.3318),
    # Maroc
    GeoZoneEntry(60, "MA", "Casablanca", "Anfa", ["anfa", "ain diab"], 33.5731, -7.6295),
    GeoZoneEntry(61, "MA", "Casablanca", "Mâarif", ["maarif"], 33.5785, -7.6380),
]

# Build lookup indices
_COMMUNE_LOOKUP: dict[tuple[str, str], GeoZoneEntry] = {}
_ALIAS_LOOKUP: dict[str, GeoZoneEntry] = {}

for zone in _GEO_ZONES:
    _COMMUNE_LOOKUP[(zone.city.lower(), zone.commune.lower())] = zone
    _ALIAS_LOOKUP[zone.commune.lower()] = zone
    for alias in zone.aliases:
        _ALIAS_LOOKUP[alias.lower()] = zone


# ──────────────────────────────────────────────────────────────
# NLP Geocoder for informal African addresses
# ──────────────────────────────────────────────────────────────

def nlp_geocode_address(raw_text: str) -> tuple[GeoZoneEntry | None, float]:
    """Parse informal addresses like "Cocody derrière Total" → commune=Cocody.

    Returns (geo_zone, confidence_score).
    """
    text = raw_text.lower().strip()

    # Direct alias match
    for alias, zone in _ALIAS_LOOKUP.items():
        if alias in text:
            return zone, 0.85

    # Pattern matching for common formats
    # "Quartier X, Commune Y" or "Commune - Quartier"
    for zone in _GEO_ZONES:
        commune_lower = zone.commune.lower()
        if commune_lower in text:
            return zone, 0.80

    # City-level fallback
    for zone in _GEO_ZONES:
        if zone.city.lower() in text:
            return zone, 0.50

    return None, 0.0


# ──────────────────────────────────────────────────────────────
# Main Classification Pipeline
# ──────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    order_id: str
    vendor_id: str
    buyer_country: str = "CI"
    buyer_city: str = ""
    buyer_commune: str = ""
    buyer_lat: float | None = None
    buyer_lon: float | None = None
    raw_address: str | None = None
    vendor_commune: str = ""
    vendor_city: str = ""
    vendor_country: str = "CI"
    vendor_lat: float | None = None
    vendor_lon: float | None = None


class ClassifyResponse(BaseModel):
    order_id: str
    geo_level: str
    commune: str = ""
    city: str = ""
    country: str = ""
    distance_km: float = 0.0
    confidence: float = 1.0
    method: str = "geo_hierarchy"
    shipping_suggestion: str = ""
    needs_verification: bool = False


def _suggest_shipping(level: str, country: str, distance_km: float) -> str:
    """Suggest shipping method based on geo level."""
    if level == "L1":
        return "Livraison moto express (même commune)"
    elif level == "L2":
        return "Livraison moto / Yango" if country == "CI" else "Colissimo standard"
    elif level == "L3":
        return "Wafio / Transport inter-ville" if country == "CI" else "Colissimo 48h"
    else:
        return "DHL Express / Colissimo International"


@app.post("/api/v1/geosort/classify", response_model=ClassifyResponse)
async def classify_order(req: ClassifyRequest):
    """Classify order geographically in <500ms — Section 43.

    Priority:
        1. GPS coordinates → geo zone lookup
        2. Structured address (commune field)
        3. NLP geocoding (informal text address)
    """
    buyer_zone: GeoZoneEntry | None = None
    confidence = 1.0
    method = ClassificationMethod.GEO_HIERARCHY

    # ── Method 1: GPS lookup ──
    if req.buyer_lat and req.buyer_lon:
        # Find nearest zone by distance
        min_dist = float("inf")
        for zone in _GEO_ZONES:
            if zone.country_code == req.buyer_country:
                dist = haversine_km(req.buyer_lat, req.buyer_lon, zone.center_lat, zone.center_lon)
                if dist < min_dist:
                    min_dist = dist
                    buyer_zone = zone
        confidence = 0.99 if min_dist < 10 else 0.80
        method = ClassificationMethod.GEO_HIERARCHY

    # ── Method 2: Structured address ──
    elif req.buyer_commune:
        key = (req.buyer_city.lower(), req.buyer_commune.lower())
        buyer_zone = _COMMUNE_LOOKUP.get(key) or _ALIAS_LOOKUP.get(req.buyer_commune.lower())
        confidence = 0.95 if buyer_zone else 0.40
        method = ClassificationMethod.GEO_HIERARCHY

    # ── Method 3: NLP for informal addresses ──
    elif req.raw_address:
        buyer_zone, confidence = nlp_geocode_address(req.raw_address)
        method = ClassificationMethod.NLP

    # ── Determine geo level ──
    if buyer_zone is None:
        geo_level = GeoLevel.L4
        distance_km = 0.0
    elif req.vendor_country != buyer_zone.country_code:
        geo_level = GeoLevel.L4
        distance_km = haversine_km(
            buyer_zone.center_lat, buyer_zone.center_lon,
            req.vendor_lat or 0, req.vendor_lon or 0,
        ) if req.vendor_lat else 5000.0
    elif req.vendor_city.lower() != buyer_zone.city.lower():
        geo_level = GeoLevel.L3
        distance_km = haversine_km(
            buyer_zone.center_lat, buyer_zone.center_lon,
            req.vendor_lat or 0, req.vendor_lon or 0,
        ) if req.vendor_lat else 200.0
    elif req.vendor_commune.lower() != buyer_zone.commune.lower():
        geo_level = GeoLevel.L2
        distance_km = haversine_km(
            buyer_zone.center_lat, buyer_zone.center_lon,
            req.vendor_lat or buyer_zone.center_lat, req.vendor_lon or buyer_zone.center_lon,
        )
    else:
        geo_level = GeoLevel.L1
        distance_km = haversine_km(
            buyer_zone.center_lat, buyer_zone.center_lon,
            req.vendor_lat or buyer_zone.center_lat, req.vendor_lon or buyer_zone.center_lon,
        ) if req.vendor_lat else 2.0

    shipping = _suggest_shipping(geo_level, buyer_zone.country_code if buyer_zone else req.buyer_country, distance_km)

    return ClassifyResponse(
        order_id=req.order_id,
        geo_level=geo_level,
        commune=buyer_zone.commune if buyer_zone else "",
        city=buyer_zone.city if buyer_zone else "",
        country=buyer_zone.country_code if buyer_zone else req.buyer_country,
        distance_km=round(distance_km, 1),
        confidence=round(confidence, 2),
        method=method,
        shipping_suggestion=shipping,
        needs_verification=confidence < 0.6,
    )
