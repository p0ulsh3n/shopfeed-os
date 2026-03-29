"""
CLIP Encoder — embeddings visuels 512d pour produits et textes.

Stratégie:
  Mode (catégorie 1): Marqo/marqo-fashionSigLIP (+22% recall@1 vs CLIP générique)
  Autres catégories:   ViT-B/32 OpenAI CLIP (gelé)

Ces encodeurs sont GELÉS — jamais re-entraînés.
Les embeddings sont stockés dans catalog_db.products.clip_embedding (vector 512).
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Catégorie ID de la Mode
FASHION_CATEGORY_ID = 1

# Modèles disponibles
FASHION_MODEL_NAME = "Marqo/marqo-fashionSigLIP"
GENERIC_MODEL_NAME = "ViT-B/32"

_fashion_model = None
_generic_model = None
_text_model = None  # sentence-transformers


def _get_fashion_model():
    global _fashion_model
    if _fashion_model is None:
        try:
            from ml.feature_store.encoders import get_visual_encoder
            _fashion_model = get_visual_encoder(model_name=FASHION_MODEL_NAME)
            logger.info(f"FashionSigLIP encoder loaded: {FASHION_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load FashionSigLIP: {e}")
    return _fashion_model


def _get_generic_model():
    global _generic_model
    if _generic_model is None:
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
            )
            model.eval()
            _generic_model = (model, preprocess)
            logger.info("Generic CLIP (ViT-B/32) loaded.")
        except ImportError:
            logger.warning("open_clip not installed. Using zero vectors for CLIP.")
        except Exception as e:
            logger.error(f"Failed to load generic CLIP: {e}")
    return _generic_model


def _get_text_encoder():
    global _text_model
    if _text_model is None:
        try:
            from ml.feature_store.encoders import get_text_encoder
            _text_model = get_text_encoder()
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
    return _text_model


def _load_image_from_url(image_url: str):
    """Charge une image depuis une URL en PIL Image."""
    from PIL import Image
    import httpx
    resp = httpx.get(image_url, timeout=10)
    resp.raise_for_status()
    import io
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def encode_product_image(image_url: str, category_id: int = 0) -> np.ndarray:
    """
    Encode une image produit en vecteur 512d.

    Args:
        image_url:   URL de l'image (S3 ou CDN)
        category_id: ID catégorie — détermine le modèle utilisé

    Returns:
        np.ndarray[512] — L2-normalized embedding
    """
    import torch

    try:
        if category_id == FASHION_CATEGORY_ID:
            model = _get_fashion_model()
            if model is not None:
                image = _load_image_from_url(image_url)
                with torch.no_grad():
                    emb = model.encode_image(image)
                    emb_np = emb.cpu().numpy().flatten()
                    # L2 normalize
                    norm = np.linalg.norm(emb_np)
                    return emb_np / max(norm, 1e-8)
        else:
            generic = _get_generic_model()
            if generic is not None:
                model, preprocess = generic
                image = _load_image_from_url(image_url)
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    emb = model.encode_image(img_tensor)
                    emb_np = emb.cpu().numpy().flatten()
                    norm = np.linalg.norm(emb_np)
                    return emb_np / max(norm, 1e-8)
    except Exception as e:
        logger.error(f"CLIP encode failed for {image_url}: {e}")

    return np.zeros(512, dtype=np.float32)


def encode_text(text: str) -> np.ndarray:
    """
    Encode un texte en vecteur 768d.
    Utilise paraphrase-multilingual-mpnet-base-v2 (sentence-transformers).

    Returns:
        np.ndarray[768]
    """
    encoder = _get_text_encoder()
    if encoder is None:
        return np.zeros(768, dtype=np.float32)
    try:
        emb = encoder.encode(text, normalize_embeddings=True)
        if hasattr(emb, "numpy"):
            return emb.numpy()
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        logger.error(f"Text encode failed: {e}")
        return np.zeros(768, dtype=np.float32)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity entre deux embeddings normalisés."""
    return float(np.dot(emb1, emb2))


def find_similar_products(
    query_emb: np.ndarray,
    faiss_index,
    k: int = 20,
) -> list[tuple[str, float]]:
    """
    ANN search FAISS → [(product_id, score)] triés par score décroissant.
    Utilisé pour la recherche visuelle (/api/v1/search/visual).
    """
    ids, scores = faiss_index.search(query_emb.reshape(1, -1), k=k)
    return list(zip(ids, scores))
