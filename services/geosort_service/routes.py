"""GeoSort Service — FastAPI App + Routes — Section 43."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from shared.models.geo import ClassificationMethod, GeoLevel

from .classifier import (
    ALIAS_LOOKUP,
    COMMUNE_LOOKUP,
    GEO_ZONES,
    GeoZoneEntry,
    haversine_km,
    nlp_geocode_address,
    suggest_shipping,
)
from .schemas import ClassifyRequest, ClassifyResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — GeoSort Service", version="1.0.0")


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
        min_dist = float("inf")
        for zone in GEO_ZONES:
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
        buyer_zone = COMMUNE_LOOKUP.get(key) or ALIAS_LOOKUP.get(req.buyer_commune.lower())
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

    shipping = suggest_shipping(geo_level, buyer_zone.country_code if buyer_zone else req.buyer_country, distance_km)

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
