"""
Lazy-loaded Encoders — API publique du Feature Store (Section 41)
=================================================================
Point d'entrée unique pour tous les encodeurs visuels et texte.
Délègue à ml.feature_store.multi_domain_encoder.EcommerceEncoder.

Architecture (validée production avril 2026) :
  - Base model : Marqo/marqo-ecommerce-embeddings-L (652M, 1024d)
  - Adapters   : LoRA par catégorie (PEFT, r=8, zéro latence après merge)
  - Projection : 1024d → 512d (couche apprise, compatible pipeline existant)
  - Fallback   : FashionSigLIP (mode), CLIP DataComp (générique)
  - Config     : configs/encoders.yaml + env vars

API BACKWARD-COMPATIBLE :
  get_visual_encoder()                    → (model, preprocess) — legacy
  get_visual_encoder_for_category(cat)    → torch.Tensor [512]  — nouveau
  encode_product_image(image, cat)        → torch.Tensor [512]  — production
  encode_query_image(image)              → torch.Tensor [512]  — online
  get_text_encoder()                      → SentenceTransformer
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ─── Singletons — conservés pour compatibilité avec le code existant ──────────
_legacy_visual_encoder = None
_text_encoder          = None


# ══════════════════════════════════════════════════════════════
# API PRINCIPALE (production 2026)
# ══════════════════════════════════════════════════════════════

def encode_product_image(image: Any, category: str = "default") -> torch.Tensor:
    """Encode une image produit avec le bon adapter LoRA selon la catégorie.

    OFFLINE — appeler lors de l'indexation des produits, pas à chaque requête.

    Args:
        image:    PIL.Image, chemin fichier (str), ou URL (str)
        category: Catégorie du produit ("electronics", "fashion", "food"...)

    Returns:
        Tensor [512] normalisé L2, prêt pour FAISS / comparaison cosine.
    """
    try:
        from ml.feature_store.multi_domain_encoder import EcommerceEncoder
        return EcommerceEncoder.get_instance().encode_product(image, category)
    except Exception as e:
        logger.error("encode_product_image échoué: %s — retourne zéros", e)
        return torch.zeros(512)


def encode_query_image(image: Any) -> torch.Tensor:
    """Encode une image de requête utilisateur (sans adapter, <10ms GPU).

    ONLINE — appeler à chaque requête de recherche visuelle.
    Même espace d'embedding que encode_product_image().

    Args:
        image: PIL.Image, chemin fichier (str), ou URL (str)

    Returns:
        Tensor [512] normalisé L2.
    """
    try:
        from ml.feature_store.multi_domain_encoder import EcommerceEncoder
        return EcommerceEncoder.get_instance().encode_query(image)
    except Exception as e:
        logger.error("encode_query_image échoué: %s — retourne zéros", e)
        return torch.zeros(512)


def encode_product_batch(
    images: list[Any],
    categories: list[str] | None = None,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode un batch de produits en parallèle (optimisé GPU).

    OFFLINE — pour l'indexation initiale ou la mise à jour en masse.

    Args:
        images:     Liste de PIL.Image ou chemins
        categories: Catégories correspondantes (None = "default" pour tous)
        batch_size: Taille de batch GPU

    Returns:
        Tensor [N, 512] normalisé L2.
    """
    try:
        from ml.feature_store.multi_domain_encoder import EcommerceEncoder
        return EcommerceEncoder.get_instance().encode_batch(images, categories, batch_size)
    except Exception as e:
        logger.error("encode_product_batch échoué: %s — retourne zéros", e)
        return torch.zeros(len(images) if images else 0, 512)


# ══════════════════════════════════════════════════════════════
# API LEGACY — compatibilité avec le code existant dans le pipeline
# ══════════════════════════════════════════════════════════════

def get_visual_encoder() -> Any:
    """[LEGACY] Retourne (model, preprocess) pour compatibilité avec l'ancien code.

    Le nouveau code doit utiliser encode_product_image() ou encode_query_image()
    qui sont plus simples et intègrent déjà la projection + le LoRA.

    Returns:
        (model, preprocess) si le modèle de base est disponible,
        None sinon.
    """
    global _legacy_visual_encoder
    if _legacy_visual_encoder is not None:
        return _legacy_visual_encoder

    try:
        from ml.feature_store.multi_domain_encoder import EcommerceEncoder
        enc = EcommerceEncoder.get_instance()
        base = enc._ensure_base_model()
        if base is not None:
            _legacy_visual_encoder = (base.model, base.preprocess)
            logger.info(
                "get_visual_encoder() [legacy] → %s (dim=%d). "
                "Préférer encode_product_image() pour le nouveau code.",
                base.model_id, base.output_dim,
            )
            return _legacy_visual_encoder
    except Exception as e:
        logger.error("get_visual_encoder() échoué: %s", e)

    # Fallback legacy (open_clip direct)
    logger.warning("EcommerceEncoder indisponible — fallback open_clip legacy")
    return _get_visual_encoder_legacy()


def _get_visual_encoder_legacy() -> Any:
    """Fallback open_clip pur (si transformers / multi_domain_encoder indispoble)."""
    global _legacy_visual_encoder
    if _legacy_visual_encoder is not None:
        return _legacy_visual_encoder

    try:
        from ml.config_loader import get_visual_encoder_model
        model_cfg  = get_visual_encoder_model()
    except Exception:
        model_cfg = {}

    primary  = model_cfg.get("primary", "")
    fallback = model_cfg.get("fallback", "ViT-B-16-SigLIP")
    fallback_pt = model_cfg.get("fallback_pretrained", "webli")

    for model_id in [primary, fallback]:
        if not model_id:
            continue
        result = _try_load_openclip(model_id)
        if result:
            _legacy_visual_encoder = result
            logger.info("Legacy encoder chargé: %s", model_id)
            return _legacy_visual_encoder

    result = _try_load_openclip_arch("ViT-B-16-SigLIP", "webli")
    if result:
        _legacy_visual_encoder = result
        return _legacy_visual_encoder

    logger.error("Tous les encodeurs visuels ont échoué.")
    return None


def _try_load_openclip(model_id: str) -> Any | None:
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_id)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return (model, preprocess)
    except Exception as e:
        logger.debug("_try_load_openclip(%s): %s", model_id, e)
    return None


def _try_load_openclip_arch(arch: str, pretrained: str) -> Any | None:
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return (model, preprocess)
    except Exception as e:
        logger.debug("_try_load_openclip_arch(%s, %s): %s", arch, pretrained, e)
    return None


# ══════════════════════════════════════════════════════════════
# Text Encoder
# ══════════════════════════════════════════════════════════════

def get_text_encoder() -> Any:
    """Charge le sentence-transformer multilingue (🔒 gelé).

    Modèle configuré dans configs/encoders.yaml (text_encoder.model)
    ou via TEXT_ENCODER_MODEL env var.

    Supports: FR, EN, AR, et 50+ langues.
    Output: 768d embeddings.
    """
    global _text_encoder
    if _text_encoder is not None:
        return _text_encoder

    try:
        from ml.config_loader import get_text_encoder_model
        model_name = get_text_encoder_model()
    except Exception:
        model_name = os.environ.get(
            "TEXT_ENCODER_MODEL",
            "paraphrase-multilingual-mpnet-base-v2",
        )

    try:
        from sentence_transformers import SentenceTransformer
        _text_encoder = SentenceTransformer(model_name)
        for param in _text_encoder.parameters():
            param.requires_grad = False  # 🔒 FROZEN
        logger.info("Text encoder chargé: %s (🔒 gelé)", model_name)
    except Exception as e:
        logger.warning("sentence-transformers indisponible (%s): %s", model_name, e)

    return _text_encoder


# ══════════════════════════════════════════════════════════════
# Utilitaires
# ══════════════════════════════════════════════════════════════

def reset_encoders() -> None:
    """Reset tous les singletons (tests, changement de domaine à chaud)."""
    global _legacy_visual_encoder, _text_encoder
    _legacy_visual_encoder = None
    _text_encoder          = None
    try:
        from ml.feature_store.multi_domain_encoder import EcommerceEncoder
        EcommerceEncoder.reset()
    except Exception:
        pass
    logger.info("Tous les encodeurs réinitialisés.")


def encoder_status() -> dict:
    """Status de l'encodeur — utile pour les health checks et le monitoring."""
    status: dict = {
        "text_encoder":    "loaded" if _text_encoder else "not_loaded",
        "legacy_encoder":  "loaded" if _legacy_visual_encoder else "not_loaded",
        "ecommerce_ready": False,
        "adapters":        [],
        "output_dim":      512,
    }
    try:
        from ml.feature_store.multi_domain_encoder import EcommerceEncoder
        enc = EcommerceEncoder.get_instance()
        status["ecommerce_ready"] = enc.is_ready
        status["adapters"]        = enc.available_adapters()
        status["output_dim"]      = enc.output_dim
    except Exception:
        pass
    return status
