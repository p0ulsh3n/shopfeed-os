"""
BLIP-2 Describer — Auto-génération de description produit depuis une image.

Modèle: Salesforce/blip2-opt-2.7b ou blip2-flan-t5-xl
Déclenché asynchrone après upload vidéo/photo vendeur.
Résultat stocké dans products.auto_description.
"""

from __future__ import annotations
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_blip2_processor = None
_blip2_model = None


def _load_blip2():
    global _blip2_processor, _blip2_model
    if _blip2_model is not None:
        return _blip2_processor, _blip2_model

    model_name = os.environ.get("BLIP2_MODEL", "Salesforce/blip2-opt-2.7b")
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _blip2_processor = Blip2Processor.from_pretrained(model_name)
        _blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        logger.info(f"BLIP-2 model loaded: {model_name} on {device}")
    except ImportError:
        logger.warning("transformers not installed. BLIP-2 unavailable.")
    except Exception as e:
        logger.error(f"Failed to load BLIP-2: {e}")

    return _blip2_processor, _blip2_model


async def describe_product(image_url: str, product_title: str = "") -> str:
    """
    Génère une description textuelle d'un produit depuis sa première image.

    Args:
        image_url:     URL de l'image principale du produit
        product_title: titre du produit pour contextualiser le prompt

    Returns:
        str — description auto-générée, max 300 chars
        Stocké dans products.auto_description
    """
    import asyncio

    processor, model = _load_blip2()
    if model is None:
        return _fallback_description(product_title)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_blip2, image_url, product_title, processor, model)


def _run_blip2(
    image_url: str,
    product_title: str,
    processor,
    model,
) -> str:
    """Exécuté dans un thread (opération GPU bloquante)."""
    try:
        from PIL import Image
        import httpx
        import io
        import torch

        resp = httpx.get(image_url, timeout=10)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        prompt_text = (
            f"Describe this product for sale: {product_title}. "
            "Be specific about material, color, style, and key features."
        ) if product_title else "Describe this product in detail."

        inputs = processor(images=image, text=prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
                temperature=0.7,
            )

        description = processor.decode(output[0], skip_special_tokens=True)
        # Nettoyer le prompt du début
        if prompt_text in description:
            description = description.replace(prompt_text, "").strip()

        return description[:300]

    except Exception as e:
        logger.error(f"BLIP-2 inference failed for {image_url}: {e}")
        return _fallback_description(product_title)


def _fallback_description(title: str) -> str:
    """Description de fallback si BLIP-2 non disponible."""
    return title[:200] if title else ""
