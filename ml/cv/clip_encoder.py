"""
CLIP Encoder — Embeddings visuels produits (adapter LoRA 2026)
==============================================================
Ce module est la facade légère du feature store visuel.
Toute la logique réelle est dans ml.feature_store.encoders (API production)
et ml.feature_store.multi_domain_encoder (EcommerceEncoder).

AVANT (v1)  : category_id == 1 → FashionSigLIP, sinon ViT-B/32
MAINTENANT  : encode_product_image() → ecommerce-L + LoRA par catégorie

Backward compat : l'ancienne signature encode_product_image(url, category_id)
est conservée mais délègue à la nouvelle API.

Modèle de base : Marqo/marqo-ecommerce-embeddings-L (652M, 1024d → proj → 512d)
Config         : configs/encoders.yaml
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Mapping category_id (entier legacy) → nom de domaine (str YAML)
# Permet la compat avec le code qui utilise category_id=1 pour la mode
CATEGORY_ID_TO_DOMAIN: dict[int, str] = {
    1:  "fashion",
    2:  "electronics",
    3:  "food",
    4:  "beauty",
    5:  "home",
    6:  "sports",
    7:  "auto",
    8:  "baby",
    9:  "health",
}


def encode_product_image(image_url: str, category_id: int = 0) -> np.ndarray:
    """Encode une image produit en vecteur 512d normalisé L2.

    Délègue à ml.feature_store.encoders.encode_product_image() qui utilise
    EcommerceEncoder (Marqo ecommerce-L + LoRA adapter selon catégorie).

    Args:
        image_url:   URL de l'image (S3, CDN) ou chemin local
        category_id: ID catégorie produit (legacy int) — converti en domaine texte

    Returns:
        np.ndarray[512] — L2-normalized embedding, dtype=float32
    """
    domain = CATEGORY_ID_TO_DOMAIN.get(category_id, "default")

    try:
        from ml.feature_store.encoders import encode_product_image as _enc
        tensor = _enc(image_url, category=domain)
        return tensor.numpy().astype(np.float32)
    except Exception as e:
        logger.error("encode_product_image(%s, cat=%d): %s", image_url, category_id, e)
        return np.zeros(512, dtype=np.float32)


def encode_query_image(image_url: str) -> np.ndarray:
    """Encode une image de requête utilisateur (ONLINE, pas d'adapter, <10ms GPU).

    Args:
        image_url: URL ou chemin local

    Returns:
        np.ndarray[512] — même espace d'embedding que encode_product_image()
    """
    try:
        from ml.feature_store.encoders import encode_query_image as _enc
        tensor = _enc(image_url)
        return tensor.numpy().astype(np.float32)
    except Exception as e:
        logger.error("encode_query_image(%s): %s", image_url, e)
        return np.zeros(512, dtype=np.float32)


def encode_text(text: str) -> np.ndarray:
    """Encode un texte en vecteur 768d (sentence-transformers multilingue 🔒).

    Returns:
        np.ndarray[768]
    """
    try:
        from ml.feature_store.encoders import get_text_encoder
        encoder = get_text_encoder()
        if encoder is None:
            return np.zeros(768, dtype=np.float32)
        emb = encoder.encode(text, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)
    except Exception as e:
        logger.error("encode_text: %s", e)
        return np.zeros(768, dtype=np.float32)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity entre deux embeddings normalisés L2."""
    return float(np.dot(emb1, emb2))


def find_similar_products(
    query_emb: np.ndarray,
    faiss_index,
    k: int = 20,
) -> list[tuple[str, float]]:
    """ANN search FAISS → [(product_id, score)] triés par score décroissant.

    Utilisé pour la recherche visuelle (/api/v1/search/visual).
    """
    ids, scores = faiss_index.search(query_emb.reshape(1, -1), k=k)
    return list(zip(ids[0], scores[0]))
