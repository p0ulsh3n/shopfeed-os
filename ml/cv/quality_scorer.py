"""
Quality Scorer — Évaluation de la qualité des photos produits via SightEngine.

Critères évalués:
  - Éclairage (lighting)
  - Netteté (sharpness)
  - Cadrage (framing)
  - Fond propre (background)
  - Présence humaine (face)
  - Overlays texte (text_overlay)

Le score composite [0,1] est stocké dans products.cv_score.
"""

from __future__ import annotations
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SIGHTENGINE_API = "https://api.sightengine.com/1.0"
SIGHTENGINE_USER = os.environ.get("SIGHTENGINE_USER", "")
SIGHTENGINE_SECRET = os.environ.get("SIGHTENGINE_SECRET", "")

# Poids des critères pour le score composite
QUALITY_WEIGHTS = {
    "sharpness": 0.30,
    "lighting": 0.25,
    "framing": 0.20,
    "background": 0.15,
    "no_text_overlay": 0.10,
}

# Recommandations par critère
RECOMMENDATIONS = {
    "sharpness": "Image floue — utilise un fond stable ou un trépied",
    "lighting": "Éclairage insuffisant — utilise de la lumière naturelle ou un ring light",
    "framing": "Cadrage à améliorer — centre le produit et garde des marges",
    "background": "Fond trop chargé — utilise un fond blanc ou uni",
    "text_overlay": "Texte sur l'image détecté — évite les watermarks visibles",
}


async def score_product_photo(image_url: str) -> dict[str, Any]:
    """
    Évalue la qualité d'une photo produit via SightEngine.

    Args:
        image_url: URL accessible publiquement de l'image

    Returns:
        {
            score: float [0,1],
            lighting: float,
            sharpness: float,
            framing: float,
            background_clean: bool,
            text_overlay_detected: bool,
            face_detected: bool,
            recommendations: [str],
        }
    """
    if not SIGHTENGINE_USER or not SIGHTENGINE_SECRET:
        logger.warning("SightEngine credentials not set. Returning default score.")
        return _default_score()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{SIGHTENGINE_API}/check.json",
                params={
                    "url": image_url,
                    "models": "properties,quality,type,faces,text-content",
                    "api_user": SIGHTENGINE_USER,
                    "api_secret": SIGHTENGINE_SECRET,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return _parse_sightengine_response(data)

    except httpx.TimeoutException:
        logger.warning(f"SightEngine timeout for {image_url}")
    except Exception as e:
        logger.error(f"SightEngine quality check failed for {image_url}: {e}")

    return _default_score()


def _parse_sightengine_response(data: dict) -> dict[str, Any]:
    """Parse la réponse SightEngine en score structuré."""
    recommendations = []

    # Qualité technique
    quality_raw = data.get("quality", {})
    sharpness = float(quality_raw.get("sharpness", 0.5))
    lighting = float(quality_raw.get("natural", 0.5))  # SightEngine "natural" ~ bon éclairage

    # Framing / composition
    framing_ok = data.get("type", {}).get("photo", 0) > 0.5
    framing = float(data.get("type", {}).get("photo", 0.5))

    # Fond
    background_raw = data.get("properties", {}).get("colors", {})
    # Heuristique: fond propre si la couleur principale est blanche/neutre
    main_colors = background_raw.get("dominant", [])
    background_clean = _is_clean_background(main_colors)

    # Text overlay
    text_data = data.get("text", {})
    text_overlay = text_data.get("has_artificial", False)

    # Visage
    faces = data.get("faces", {}).get("faces", [])
    face_detected = len(faces) > 0

    # Recommandations
    if sharpness < 0.5:
        recommendations.append(RECOMMENDATIONS["sharpness"])
    if lighting < 0.5:
        recommendations.append(RECOMMENDATIONS["lighting"])
    if framing < 0.5:
        recommendations.append(RECOMMENDATIONS["framing"])
    if not background_clean:
        recommendations.append(RECOMMENDATIONS["background"])
    if text_overlay:
        recommendations.append(RECOMMENDATIONS["text_overlay"])

    # Score composite
    score = (
        QUALITY_WEIGHTS["sharpness"] * sharpness
        + QUALITY_WEIGHTS["lighting"] * lighting
        + QUALITY_WEIGHTS["framing"] * framing
        + QUALITY_WEIGHTS["background"] * (1.0 if background_clean else 0.0)
        + QUALITY_WEIGHTS["no_text_overlay"] * (0.0 if text_overlay else 1.0)
    )

    return {
        "score": round(min(max(score, 0.0), 1.0), 4),
        "sharpness": round(sharpness, 4),
        "lighting": round(lighting, 4),
        "framing": round(framing, 4),
        "background_clean": background_clean,
        "text_overlay_detected": text_overlay,
        "face_detected": face_detected,
        "recommendations": recommendations,
    }


def _is_clean_background(colors: list) -> bool:
    """Heuristique: fond propre si couleur dominante est claire/unie."""
    if not colors:
        return True  # pas d'info → on assume OK
    for c in colors[:2]:
        hex_color = c.get("hex", "").lstrip("#")
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            if brightness > 200:  # fond clair = propre
                return True
    return False


def _default_score() -> dict:
    """Valeurs par défaut si SightEngine non disponible."""
    return {
        "score": 0.5,
        "sharpness": 0.5,
        "lighting": 0.5,
        "framing": 0.5,
        "background_clean": True,
        "text_overlay_detected": False,
        "face_detected": False,
        "recommendations": [],
    }
