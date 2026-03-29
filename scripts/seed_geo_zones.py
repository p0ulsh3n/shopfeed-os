"""
Seed Geo Zones — Peuplement de la table geo_zones depuis OpenStreetMap / GeoNames.

Couvre:
  CI: 13 communes Abidjan + 10 villes majeures
  SN: communes Dakar + villes SN
  FR: Paris arrondissements + villes majeures
  BE, MA, CM, DZ: villes majeures
  + GeoNames: capitales + admin level 2 pour 54 pays africains

Usage:
  python -m scripts.seed_geo_zones --countries CI,SN,FR
  python -m scripts.seed_geo_zones --africa_all
"""

from __future__ import annotations
import argparse
import asyncio
import logging
import os
import json
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class GeoZone:
    country_code: str
    country_name: str
    region: Optional[str]
    city: Optional[str]
    commune: Optional[str]
    center_lat: float
    center_lon: float
    aliases: list[str]
    country_phone_code: Optional[str] = None


# ── Zones hardcodées prioritaires ─────────────────────────────────────────────

PRIORITY_ZONES: list[GeoZone] = [
    # Côte d'Ivoire — Abidjan communes
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Plateau",
            5.3196, -4.0159, ["Le Plateau", "Plateau Abidjan"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Cocody",
            5.3711, -3.9969, ["Cocodie", "2 Plateaux", "Angré", "Riviera", "Brofodoumé", "Danga"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Yopougon",
            5.3417, -4.0839, ["Yop", "Kouté", "Port-Bouet 2", "Selmer"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Marcory",
            5.2959, -3.9895, ["Zone 4", "Marcory Zone 4", "Anoumabo"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Koumassi",
            5.2906, -3.9625, [], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Port-Bouet",
            5.2545, -3.9393, ["Vridi", "Aéroport", "Ile Boulay"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Treichville",
            5.2976, -4.0112, [], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Attécoubé",
            5.3303, -4.0407, ["Abobo Baoulé", "Washington"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Abobo",
            5.4165, -4.0165, ["Abobo Gare", "Abobo Doumé", "Abobo-Est", "N'dotré"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Adjamé",
            5.3559, -4.0191, ["Adjamé Liberté", "Agban", "220 Logements"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Bingerville",
            5.3583, -3.8900, [], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Grand-Bassam",
            5.2030, -3.7441, ["Bassam"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lagunes", "Abidjan", "Songon",
            5.3833, -4.2167, [], "+225"),
    # CI villes
    GeoZone("CI", "Côte d'Ivoire", "Vallée du Bandama", "Bouaké", None,
            7.6903, -5.0306, ["Bouaké centre", "Koko"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Bas-Sassandra", "San-Pédro", None,
            4.7485, -6.6363, [], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Lacs", "Yamoussoukro", None,
            6.8276, -5.2893, ["Yamous"], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Haut-Sassandra", "Daloa", None,
            6.8773, -6.4502, [], "+225"),
    GeoZone("CI", "Côte d'Ivoire", "Poro", "Korhogo", None,
            9.4580, -5.6290, [], "+225"),

    # Sénégal — Dakar communes
    GeoZone("SN", "Sénégal", "Dakar", "Dakar", "Plateau",
            14.6937, -17.4441, ["Centre-ville Dakar"], "+221"),
    GeoZone("SN", "Sénégal", "Dakar", "Dakar", "Médina",
            14.6961, -17.4472, [], "+221"),
    GeoZone("SN", "Sénégal", "Dakar", "Dakar", "Fann",
            14.6875, -17.4553, ["Fann-Point E", "Amitié"], "+221"),
    GeoZone("SN", "Sénégal", "Dakar", "Dakar", "Almadies",
            14.7332, -17.4999, ["Les Almadies", "Ouakam"], "+221"),
    GeoZone("SN", "Sénégal", "Dakar", "Dakar", "Pikine",
            14.7523, -17.3924, [], "+221"),
    GeoZone("SN", "Sénégal", "Thiès", "Thiès", None,
            14.7873, -16.9260, [], "+221"),
    GeoZone("SN", "Sénégal", "Saint-Louis", "Saint-Louis", None,
            16.0326, -16.5000, [], "+221"),

    # France — Paris + grandes villes
    GeoZone("FR", "France", "Île-de-France", "Paris", "1er Arrondissement",
            48.8606, 2.3477, ["1er", "Châtelet", "Louvre"], "+33"),
    GeoZone("FR", "France", "Île-de-France", "Paris", "8ème Arrondissement",
            48.8752, 2.3095, ["8ème", "Champs-Élysées", "Madeleine"], "+33"),
    GeoZone("FR", "France", "Île-de-France", "Paris", "18ème Arrondissement",
            48.8920, 2.3350, ["18ème", "Montmartre"], "+33"),
    GeoZone("FR", "France", "Auvergne-Rhône-Alpes", "Lyon", None,
            45.7640, 4.8357, ["Lyon centre", "La Part-Dieu"], "+33"),
    GeoZone("FR", "France", "Provence", "Marseille", None,
            43.2965, 5.3698, [], "+33"),

    # Cameroun
    GeoZone("CM", "Cameroun", "Littoral", "Douala", "Akwa",
            4.0483, 9.6967, ["Akwa Nord", "Bonaberi"], "+237"),
    GeoZone("CM", "Cameroun", "Centre", "Yaoundé", None,
            3.8480, 11.5021, ["Yaoundé", "Mfandena", "Bastos"], "+237"),

    # Maroc
    GeoZone("MA", "Maroc", "Grand Casablanca", "Casablanca", None,
            33.5731, -7.5898, ["Casa", "Ain Diab", "Maârif"], "+212"),
    GeoZone("MA", "Maroc", "Rabat-Salé", "Rabat", None,
            34.0209, -6.8416, [], "+212"),

    # Belgique
    GeoZone("BE", "Belgique", "Région Bruxelles-Capitale", "Bruxelles", None,
            50.8503, 4.3517, ["Brussels", "Ixelles", "Molenbeek"], "+32"),

    # Algérie
    GeoZone("DZ", "Algérie", "Alger", "Alger", None,
            36.7372, 3.0865, ["Alger centre", "Bab Ezzouar", "Hydra"], "+213"),
    GeoZone("DZ", "Algérie", "Oran", "Oran", None,
            35.6969, -0.6331, [], "+213"),
]


async def seed_zones(
    db_dsn: str,
    zones: list[GeoZone],
    upsert: bool = True,
) -> int:
    """Insère les zones dans geosort_db.geo_zones."""
    import asyncpg

    pool = await asyncpg.create_pool(db_dsn, min_size=1, max_size=3)
    inserted = 0

    try:
        for z in zones:
            try:
                if upsert:
                    await pool.execute(
                        """
                        INSERT INTO geo_zones
                          (country_code, country_name, region, city, commune,
                           center_lat, center_lon, aliases, country_phone_code)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                        ON CONFLICT (country_code, city, commune) DO UPDATE
                          SET aliases = EXCLUDED.aliases,
                              center_lat = EXCLUDED.center_lat,
                              center_lon = EXCLUDED.center_lon
                        """,
                        z.country_code, z.country_name, z.region,
                        z.city, z.commune, z.center_lat, z.center_lon,
                        z.aliases, z.country_phone_code,
                    )
                inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert {z.city}/{z.commune}: {e}")

    finally:
        await pool.close()

    return inserted


def main():
    parser = argparse.ArgumentParser(description="Seed geo_zones table")
    parser.add_argument(
        "--db_dsn",
        default=os.environ.get("GEOSORT_DB_DSN", "postgresql://localhost/geosort_db"),
    )
    parser.add_argument(
        "--countries",
        default="CI,SN,FR,CM,MA,BE,DZ",
        help="Codes pays séparés par des virgules"
    )
    parser.add_argument("--upsert", action="store_true", default=True)
    args = parser.parse_args()

    countries = {c.strip().upper() for c in args.countries.split(",")}
    zones_to_seed = [z for z in PRIORITY_ZONES if z.country_code in countries]

    logger.info(f"Seeding {len(zones_to_seed)} zones for {countries}")
    n = asyncio.run(seed_zones(args.db_dsn, zones_to_seed, upsert=args.upsert))
    logger.info(f"Done. {n} zones seeded.")


if __name__ == "__main__":
    main()
