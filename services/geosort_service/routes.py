"""GeoSort Service — FastAPI App + Routes.

POST /api/v1/geosort/classify → classify an order into Zone A/B/C.
100% global, offline. Uses reverse_geocode + haversine.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI

from .classifier import classify_order, haversine_km, resolve_location
from .schemas import ClassifyRequest, ClassifyResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="ShopFeed OS — GeoSort Service", version="2.0.0")


@app.post("/api/v1/geosort/classify", response_model=ClassifyResponse)
async def classify_order_route(req: ClassifyRequest) -> ClassifyResponse:
    """Classify order geographically — Zone A (Livraison), B or C (Expedition).

    Input: vendor lat/lon + buyer lat/lon (from React Native geolocation).
    Output: zone, distance, resolved cities/countries, shipping suggestion.

    Performance: <5ms (offline reverse geocoding + haversine math).
    """
    t_start = time.perf_counter()

    result = classify_order(
        vendor_lat=req.vendor_lat,
        vendor_lon=req.vendor_lon,
        buyer_lat=req.buyer_lat,
        buyer_lon=req.buyer_lon,
        same_zone_radius_km=req.same_zone_radius_km,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "GeoSort: order=%s zone=%s (%s) %s→%s %.1fkm in %.1fms",
        req.order_id, result.zone, result.zone_label,
        result.vendor_city, result.buyer_city,
        result.distance_km, elapsed_ms,
    )

    return ClassifyResponse(
        order_id=req.order_id,
        zone=result.zone,
        zone_label=result.zone_label,
        geo_level=result.geo_level,
        distance_km=result.distance_km,
        buyer_city=result.buyer_city,
        buyer_country=result.buyer_country,
        buyer_country_code=result.buyer_country_code,
        vendor_city=result.vendor_city,
        vendor_country=result.vendor_country,
        vendor_country_code=result.vendor_country_code,
        shipping_suggestion=result.shipping_suggestion,
        confidence=result.confidence,
    )


@app.get("/api/v1/geosort/health")
async def health():
    """Health check — verify reverse_geocode is loaded."""
    try:
        loc = resolve_location(5.3599, -4.0083)  # Abidjan
        return {
            "status": "ok",
            "test_city": loc.city,
            "test_country": loc.country,
            "reverse_geocode": "loaded",
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}
