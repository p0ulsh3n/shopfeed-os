"""
Quality Scorer — Evaluation de la qualite des photos produits via Llama 4 Scout.

Pipeline de scoring (2026):
    Llama 4 Scout 17B-16E (Meta)
        → 17B params actifs / 109B total (MoE 16 experts)
        → Multimodal natif (texte + image)
        → 1 seul A100 80GB
        → Remplace SightEngine + Qwen2.5-VL-7B

Criteres evalues:
  - Eclairage (lighting)
  - Nettete (sharpness)
  - Cadrage (framing)
  - Fond propre (background)
  - Presence humaine (face)
  - Overlays texte (text_overlay)
  - Composition (Scout vision)
  - Coherence de marque (Scout vision)
  - Conseils styling (Scout vision)
  - Adequation categorie (Scout vision)

Le score composite [0,1] est stocke dans products.cv_score.
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Poids des criteres pour le score composite
QUALITY_WEIGHTS = {
    "sharpness": 0.25,
    "lighting": 0.20,
    "framing": 0.15,
    "background": 0.10,
    "no_text_overlay": 0.05,
    "composition": 0.15,
    "brand_coherence": 0.10,
}

# Recommandations par critere
RECOMMENDATIONS = {
    "sharpness": "Image floue — utilise un fond stable ou un trepied",
    "lighting": "Eclairage insuffisant — utilise de la lumiere naturelle ou un ring light",
    "framing": "Cadrage a ameliorer — centre le produit et garde des marges",
    "background": "Fond trop charge — utilise un fond blanc ou uni",
    "text_overlay": "Texte sur l'image detecte — evite les watermarks visibles",
}


async def score_product_photo(
    image_url: str,
    product_title: str = "",
    category: str = "",
) -> dict[str, Any]:
    """Evalue la qualite d'une photo produit via Llama 4 Scout.

    Args:
        image_url: URL accessible publiquement de l'image
        product_title: Titre du produit (enrichit l'analyse)
        category: Categorie du produit

    Returns:
        {
            score: float [0,1],
            sharpness, lighting, framing: float,
            background_clean, text_overlay_detected, face_detected: bool,
            composition_score, brand_coherence: float,
            styling_advice, category_fit: str,
            recommendations: [str],
            scored_by: "scout" | "default",
        }
    """
    # ── Llama 4 Scout multimodal scoring ─────────────────────────
    try:
        from ml.llm.llm_enrichment import score_photo_quality
        result = await score_photo_quality(image_url, product_title, category)

        if result and isinstance(result, dict) and "score" in result:
            # Enrich recommendations from thresholds
            recs = list(result.get("recommendations", []))
            if result.get("sharpness", 1) < 0.5:
                recs.append(RECOMMENDATIONS["sharpness"])
            if result.get("lighting", 1) < 0.5:
                recs.append(RECOMMENDATIONS["lighting"])
            if result.get("framing", 1) < 0.5:
                recs.append(RECOMMENDATIONS["framing"])
            if not result.get("background_clean", True):
                recs.append(RECOMMENDATIONS["background"])
            if result.get("text_overlay_detected", False):
                recs.append(RECOMMENDATIONS["text_overlay"])

            # Deduplicate
            result["recommendations"] = list(dict.fromkeys(recs))
            result["scored_by"] = "scout"

            logger.info(
                "Scout quality score: %.3f for %s (composition=%.2f, brand=%.2f)",
                result["score"], image_url[:60],
                result.get("composition_score", 0),
                result.get("brand_coherence", 0),
            )
            return result

    except ImportError:
        logger.debug("ml.llm not available — using default score")
    except Exception as e:
        logger.warning("Scout quality scoring failed: %s", e)

    # ── Default scores (graceful degradation) ────────────────────
    result = _default_score()
    result["scored_by"] = "default"
    return result


def _default_score() -> dict:
    """Valeurs par defaut si Scout n'est pas disponible."""
    return {
        "score": 0.5,
        "sharpness": 0.5,
        "lighting": 0.5,
        "framing": 0.5,
        "background_clean": True,
        "text_overlay_detected": False,
        "face_detected": False,
        "composition_score": 0.5,
        "brand_coherence": 0.5,
        "styling_advice": "",
        "category_fit": "",
        "recommendations": [],
    }
