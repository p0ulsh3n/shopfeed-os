"""Geo classification logic — zones, haversine, NLP geocoding — Section 43."""

from __future__ import annotations

import math
from dataclasses import dataclass


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth."""
    R = 6371.0  # Earth radius km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


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
GEO_ZONES: list[GeoZoneEntry] = [
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
COMMUNE_LOOKUP: dict[tuple[str, str], GeoZoneEntry] = {}
ALIAS_LOOKUP: dict[str, GeoZoneEntry] = {}

for zone in GEO_ZONES:
    COMMUNE_LOOKUP[(zone.city.lower(), zone.commune.lower())] = zone
    ALIAS_LOOKUP[zone.commune.lower()] = zone
    for alias in zone.aliases:
        ALIAS_LOOKUP[alias.lower()] = zone


def nlp_geocode_address(raw_text: str) -> tuple[GeoZoneEntry | None, float]:
    """Parse informal addresses like "Cocody derrière Total" → commune=Cocody.

    Returns (geo_zone, confidence_score).
    """
    text = raw_text.lower().strip()

    # Direct alias match
    for alias, zone in ALIAS_LOOKUP.items():
        if alias in text:
            return zone, 0.85

    # Pattern matching for common formats
    for zone in GEO_ZONES:
        commune_lower = zone.commune.lower()
        if commune_lower in text:
            return zone, 0.80

    # City-level fallback
    for zone in GEO_ZONES:
        if zone.city.lower() in text:
            return zone, 0.50

    return None, 0.0


def suggest_shipping(level: str, country: str, distance_km: float) -> str:
    """Suggest shipping method based on geo level."""
    if level == "L1":
        return "Livraison moto express (même commune)"
    elif level == "L2":
        return "Livraison moto / Yango" if country == "CI" else "Colissimo standard"
    elif level == "L3":
        return "Wafio / Transport inter-ville" if country == "CI" else "Colissimo 48h"
    else:
        return "DHL Express / Colissimo International"
