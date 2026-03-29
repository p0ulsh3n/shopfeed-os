"""
Whisper Transcriber — Transcription audio automatique + extraction d'entités.

Modèle: openai/whisper-large-v3
Déclenché asynchrone après upload vidéo vendeur.
Résultats stockés dans:
  - feed_content.transcript
  - session ASR Index (Redis session:{session_id}:asr_index)

Entités extraites:
  - Produits mentionnés (robe, chaussures, sac...)
  - Marques
  - Prix (regex "(\d+) euros?", "gratuit", "promo")
  - Urgence ("maintenant", "dernière chance", "stock limité")
"""

from __future__ import annotations
import logging
import os
import re
import asyncio
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Modèle Whisper
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")

_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info("Whisper model loaded.")
    except ImportError:
        logger.warning("openai-whisper not installed. Transcription unavailable.")
    except Exception as e:
        logger.error(f"Failed to load Whisper: {e}")
    return _whisper_model


# ── Extraction d'entités ────────────────────────────────────────────────────

# Mots de produits courants
PRODUCT_KEYWORDS = {
    # Mode
    "robe", "chemise", "pantalon", "jupe", "veste", "manteau", "pull",
    "chaussures", "sac", "ceinture", "bijoux", "collier", "bracelet", "montre",
    # Beauté
    "crème", "sérum", "lotion", "parfum", "maquillage", "rouge", "fond de teint",
    # Alimentation
    "attiéké", "kedjenou", "thiéboudienne", "mafé", "poulet", "poisson",
    "jus", "sauce", "pâte", "riz", "haricot",
    # Tech
    "téléphone", "smartphone", "casque", "chargeur", "câble", "écouteur",
    # Maison
    "cuisine", "lit", "table", "chaise", "canapé", "rideau", "drap",
}

# Urgence
URGENCE_KEYWORDS = {
    "maintenant", "dernière chance", "stock limité", "quantité limitée",
    "aujourd'hui seulement", "offre limitée", "promo flash", "vite",
    "dépêche", "avant que ça se termine", "plus que",
}

# Prix patterns (FR + FCFA)
PRICE_PATTERNS = [
    r'\b(\d+(?:\s\d{3})*)\s*(?:euros?|€|FCFA|XOF|CFA|francs?)\b',
    r'\bgratuit\b',
    r'\boffert\b',
    r'\bentre\s+(\d+)\s+et\s+(\d+)\b',
    r'\bpromo\b', r'\bpromotion\b', r'\bréduction\b', r'\bremise\b',
]


def extract_entities(transcript: str) -> list[dict]:
    """
    Extrait des entités du transcript :
      - product: mots-clés produits
      - price: montants + promos
      - urgency: mots d'urgence
      - brand: dans la liste brand_dictionary (simplifiée ici)

    Returns:
        [{text, type: product|brand|price|urgency, confidence}]
    """
    entities = []
    text_lower = transcript.lower()

    # Produits
    for kw in PRODUCT_KEYWORDS:
        if kw in text_lower:
            entities.append({
                "text": kw,
                "type": "product",
                "confidence": 0.80,
            })

    # Prix
    for pattern in PRICE_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for m in matches:
            text = m if isinstance(m, str) else " ".join(m)
            entities.append({
                "text": text.strip(),
                "type": "price",
                "confidence": 0.90,
            })

    # Urgence
    for kw in URGENCE_KEYWORDS:
        if kw in text_lower:
            entities.append({
                "text": kw,
                "type": "urgency",
                "confidence": 0.85,
            })

    # Dédoublonnage sur le texte
    seen = set()
    unique = []
    for e in entities:
        if e["text"] not in seen:
            seen.add(e["text"])
            unique.append(e)

    return unique


async def transcribe(audio_url: str, language: str = "fr") -> dict:
    """
    Transcrit un audio depuis son URL.

    Args:
        audio_url: URL de l'audio ou vidéo (S3, CDN)
        language:  langue principale (défaut: 'fr')

    Returns:
        {
            text: str,
            segments: [{start, end, text}],
            language: str,
            entities: [{text, type, confidence}],
        }
    """
    model = _load_whisper()
    if model is None:
        return _empty_result()

    loop = asyncio.get_event_loop()

    # Télécharger l'audio en local
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
    except Exception as e:
        logger.error(f"Failed to download audio from {audio_url}: {e}")
        return _empty_result()

    # Transcription dans un thread
    result = await loop.run_in_executor(
        None,
        lambda: _run_whisper(model, tmp_path, language),
    )

    # Cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    # Extraction d'entités
    full_text = result.get("text", "")
    entities = extract_entities(full_text)
    result["entities"] = entities

    return result


def _run_whisper(model, audio_path: str, language: str) -> dict:
    """Exécuté dans un thread executor (opération GPU bloquante)."""
    try:
        result = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            verbose=False,
        )
        segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", language),
        }
    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}")
        return _empty_result()


def _empty_result() -> dict:
    return {"text": "", "segments": [], "language": "fr", "entities": []}
