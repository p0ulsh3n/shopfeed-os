"""
Geo Classifier — Wrapper ML pour le service de classification géographique.
============================================================================

CE N'EST PAS DU MACHINE LEARNING.

La classification géographique est un problème 100% déterministe :
  - distance < 50km → livraison locale (Zone A)
  - même pays        → national (Zone B)
  - pays différent   → international (Zone C)

L'implémentation réelle est dans:
  services/geosort_service/classifier.py

Ce fichier est un WRAPPER qui:
  1. Réexporte classify_order() pour backward-compat avec le pipeline ML
  2. Fournit rule_based_geo_level() utilisé comme fallback dans le ranking
  3. Expose haversine_km() pour le calcul de distance dans le feature store

Le MLP nn.Module qui existait ici avant a été supprimé car:
  - Un réseau de neurones pour classifier "même pays ou pas" est absurde
  - Il nécessitait un modèle entraîné pour un problème déterministe
  - Il pouvait SE TROMPER (softmax P=0.51) là où les règles sont à 100%
  - Le service classifier.py est plus rapide (<1ms), plus fiable, et sans GPU
"""

from __future__ import annotations

# ── Re-export depuis le service (source de vérité) ───────────────
# Tout le code réel est dans services/geosort_service/classifier.py

try:
    from services.geosort_service.classifier import (
        classify_order,
        OrderClassification,
        GeoLocation,
        resolve_location,
        haversine_km,
    )
except ImportError:
    # Fallback si le service n'est pas dans le PYTHONPATH
    # (ex: exécution depuis ml/ directement)
    import math
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(
        "services.geosort_service.classifier not found — "
        "using minimal fallback. Add project root to PYTHONPATH."
    )

    EARTH_RADIUS_KM = 6371.0

    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two GPS coordinates."""
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Labels géographiques (backward-compat) ───────────────────────

GEO_LEVELS = ["L1", "L2", "L3", "L4"]


def rule_based_geo_level(
    buyer_country: str,
    vendor_country: str,
    buyer_city: str,
    vendor_city: str,
    buyer_commune: str = "",
    vendor_commune: str = "",
    distance_km: float | None = None,
) -> str:
    """Classification par règles déterministes (pas de ML).

    Utilisé comme feature dans le ranking pipeline pour le geo-boosting:
      - L1/L2 → boost produits locaux dans le feed
      - L3    → boost national
      - L4    → pas de boost géo

    Returns: "L1", "L2", "L3", or "L4"
    """
    # International
    if buyer_country.lower().strip() != vendor_country.lower().strip():
        return "L4"

    # National — villes différentes
    if buyer_city.lower().strip() != vendor_city.lower().strip():
        return "L3"

    # Local — même commune
    if (
        buyer_commune and vendor_commune
        and buyer_commune.lower().strip() == vendor_commune.lower().strip()
    ):
        return "L1"

    # Distance-based si disponible
    if distance_km is not None:
        if distance_km <= 5:
            return "L1"
        elif distance_km <= 30:
            return "L2"

    return "L2"


# ── GeoClassifier stub (backward-compat pour health.py/schemas.py) ──

class GeoClassifier:
    """Stub — backward-compat pour ModelRegistry et health checks.

    Ce n'est PAS un nn.Module. C'est un wrapper autour des règles
    déterministes de services/geosort_service/classifier.py.
    """

    def predict_level(
        self,
        buyer_lat: float, buyer_lon: float,
        vendor_lat: float, vendor_lon: float,
        buyer_country: str = "", vendor_country: str = "",
    ) -> str:
        """Classify geo level using rules (not ML)."""
        dist = haversine_km(buyer_lat, buyer_lon, vendor_lat, vendor_lon)
        if dist <= 50:
            return "L1" if dist <= 5 else "L2"
        if buyer_country and vendor_country:
            if buyer_country.lower() == vendor_country.lower():
                return "L3"
            return "L4"
        return "L3"
