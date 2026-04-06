"""
Lazy-loaded Encoders — Visual & Text (Section 41)
===================================================
Visual encoder choice is now driven by configs/encoders.yaml (or the
VISUAL_ENCODER_DOMAIN env var). This makes ShopFeed a TRUE GENERALIST
marketplace — not fashion-only.

Domain        Model (default)                        Best use-case
─────────── ──────────────────────────────────────── ─────────────────
generic     CLIP ViT-L-14 DataComp (LAION)            Mixed catalogue
fashion     Marqo/marqo-fashionSigLIP                 >70% mode/apparel
electronics CLIP ViT-L-14 (OpenAI)                   Tech / périphériques
food        CLIP ViT-B-32 (OpenAI)                   FMCG / épicerie
beauty      CLIP ViT-B-32 LAION-2B                   Beauté / cosmétiques
auto        CLIP ViT-L-14 (OpenAI)                   Automobile

Text encoder: sentence-transformers multilingual (configurable via
TEXT_ENCODER_MODEL env var or configs/encoders.yaml).

API note (open_clip >= 2.26):
    create_model_and_transforms() returns (model, train_preprocess, val_preprocess).
    Use val_preprocess for inference (no augmentations).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ─── Lazy-loaded singleton instances ──────────────────────────────────
_visual_encoder = None
_text_encoder = None


def get_visual_encoder() -> Any:
    """Load the configured visual encoder (domain-aware, not fashion-only).

    Reads the model to load from configs/encoders.yaml via
    ml.config_loader.get_visual_encoder_model(). Falls back through a
    chain of alternatives if the primary model is unavailable.

    Returns:
        (model, preprocess) tuple, or None if all encoders fail.
    """
    global _visual_encoder
    if _visual_encoder is not None:
        return _visual_encoder

    # ── Determine which model to load ───────────────────────────────
    try:
        from ml.config_loader import get_visual_encoder_model
        model_cfg = get_visual_encoder_model()   # reads domain from env/yaml
    except Exception:
        model_cfg = {}

    primary   = model_cfg.get("primary", "")
    fallback  = model_cfg.get("fallback", "ViT-B-16-SigLIP")
    pretrained_fallback = model_cfg.get("pretrained_fallback", "webli")
    domain    = os.environ.get("VISUAL_ENCODER_DOMAIN", "generic")

    logger.info(
        "Loading visual encoder — domain=%s, primary=%s",
        domain, primary or "(config not found, using SigLIP generic)",
    )

    # ── Attempt 1: Primary model from config ─────────────────────────
    if primary:
        result = _try_load_openclip(primary)
        if result is not None:
            _visual_encoder = result
            logger.info("Visual encoder loaded: %s (domain=%s, 🔒 frozen)", primary, domain)
            return _visual_encoder

    # ── Attempt 2: Fallback model from config ──────────────────────
    if fallback and fallback != primary:
        # Distinguish hf-hub models from architecture+pretrained models
        if fallback.startswith("hf-hub:") or "/" in fallback:
            result = _try_load_openclip(fallback)
        else:
            result = _try_load_openclip_arch(fallback, pretrained_fallback)
        if result is not None:
            _visual_encoder = result
            logger.warning(
                "Primary visual encoder unavailable — using fallback: %s", fallback
            )
            return _visual_encoder

    # ── Attempt 3: Generic SigLIP (last resort) ────────────────────
    result = _try_load_openclip_arch("ViT-B-16-SigLIP", "webli")
    if result is not None:
        _visual_encoder = result
        logger.warning(
            "All domain-specific encoders failed — using generic SigLIP (webli). "
            "Visual quality may be reduced. Check VISUAL_ENCODER_DOMAIN and pip install open-clip-torch>=2.29"
        )
        return _visual_encoder

    logger.error("All visual encoders failed — returning None. Image features will be zero-padded.")
    return None


def _try_load_openclip(model_id: str) -> Any | None:
    """Try to load an open_clip model by hf-hub or model name string.

    Handles both:
      - "hf-hub:Marqo/marqo-fashionSigLIP"
      - "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"

    Returns (model, val_preprocess) or None on failure.
    """
    try:
        import open_clip
        model, _train_preprocess, val_preprocess = (
            open_clip.create_model_and_transforms(model_id)
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False  # 🔒 FROZEN — never re-train
        return (model, val_preprocess)
    except Exception as e:
        logger.debug("_try_load_openclip(%s) failed: %s", model_id, e)
        return None


def _try_load_openclip_arch(arch: str, pretrained: str) -> Any | None:
    """Try to load an open_clip model by architecture name + pretrained tag.

    Examples:
        _try_load_openclip_arch("ViT-L-14", "openai")
        _try_load_openclip_arch("ViT-B-32", "openai")
        _try_load_openclip_arch("ViT-B-16-SigLIP", "webli")

    Returns (model, val_preprocess) or None on failure.
    """
    try:
        import open_clip
        model, _train_preprocess, val_preprocess = (
            open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False  # 🔒 FROZEN
        return (model, val_preprocess)
    except Exception as e:
        logger.debug("_try_load_openclip_arch(%s, %s) failed: %s", arch, pretrained, e)
        return None


def get_text_encoder() -> Any:
    """Load the configured sentence-transformer text encoder.

    Model is read from configs/encoders.yaml (text_encoder.model) or
    the TEXT_ENCODER_MODEL env var. Defaults to
    paraphrase-multilingual-mpnet-base-v2 (50+ languages, 768D, 🔒 frozen).

    For marketplaces with non-Latin languages (Arabic, Bengali, Hindi...)
    consider setting TEXT_ENCODER_MODEL=intfloat/multilingual-e5-large.
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
        # 🔒 FROZEN — never re-train the pre-trained text encoder
        for param in _text_encoder.parameters():
            param.requires_grad = False
        logger.info("Text encoder loaded: %s (🔒 frozen)", model_name)
    except Exception as e:
        logger.warning("sentence-transformers unavailable or model not found (%s): %s", model_name, e)

    return _text_encoder


def reset_encoders() -> None:
    """Reset singleton instances (useful for testing or domain switching)."""
    global _visual_encoder, _text_encoder
    _visual_encoder = None
    _text_encoder = None
    logger.info("Encoder singletons reset.")
