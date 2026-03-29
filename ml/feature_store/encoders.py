"""
Lazy-loaded Encoders — Visual & Text (Section 41)
===================================================
Marqo-FashionSigLIP (🔒 FROZEN) for visual embeddings.
sentence-transformers multilingual (🔒 FROZEN) for text embeddings.

API note (open_clip >= 2.26):
    open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
    returns (model, train_preprocess, val_preprocess) — 3 values.
    Use val_preprocess for inference.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Lazy-loaded Encoders ───────────────────────────────────────────
_visual_encoder = None
_text_encoder = None


def get_visual_encoder() -> Any:
    """Load Marqo-FashionSigLIP via HuggingFace Hub (Section 35, 41).

    Model: Marqo/marqo-fashionSigLIP
    Architecture: ViT-B-16-SigLIP fine-tuned with GCL on fashion data.
    Performance: +22% recall@1 text-to-image vs FashionCLIP 2.0.

    BUG #4 FIX: Previously loaded `pretrained="webli"` (generic SigLIP, WebLI
    dataset). Corrected to load Marqo fashion-specific weights via HF Hub.

    API FIX: open_clip.create_model_and_transforms() returns 3 values
    (model, train_preprocess, val_preprocess). We use val_preprocess for
    inference. Previously used create_model_from_pretrained which returns 2.

    Fallback chain:
        1. Marqo/marqo-fashionSigLIP  (best — fashion-specific, GCL-trained)
        2. patrickjohncyh/fashion-clip (FashionCLIP 2.0 fallback)
        3. ViT-B-16-SigLIP / webli    (generic fallback — last resort)
    """
    global _visual_encoder
    if _visual_encoder is None:
        try:
            import open_clip
            # BUG #4 FIX: correct API is create_model_and_transforms with hf-hub: prefix.
            # Returns (model, train_preprocess, val_preprocess) — 3 values.
            # Use val_preprocess (no augmentations) for inference.
            model, _train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
                "hf-hub:Marqo/marqo-fashionSigLIP"
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad = False  # 🔒 FROZEN — never re-train
            _visual_encoder = (model, val_preprocess)
            logger.info("Loaded Marqo-FashionSigLIP visual encoder (🔒 frozen, hf-hub)")
        except Exception as e:
            logger.warning("Marqo-FashionSigLIP unavailable (%s), trying FashionCLIP 2.0", e)
            _visual_encoder = _load_fashionclip_fallback()
    return _visual_encoder


def _load_fashionclip_fallback() -> Any:
    """FashionCLIP 2.0 fallback, then generic SigLIP."""
    try:
        import open_clip
        model, _train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
            "hf-hub:patrickjohncyh/fashion-clip"
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Loaded FashionCLIP 2.0 fallback encoder (🔒 frozen)")
        return (model, val_preprocess)
    except Exception as e:
        logger.warning("FashionCLIP 2.0 unavailable (%s), falling back to generic SigLIP", e)

    try:
        import open_clip
        model, _train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP", pretrained="webli"
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        logger.warning(
            "Using generic SigLIP (webli) — fashion specificity lost. "
            "Install: pip install open-clip-torch>=2.29"
        )
        return (model, val_preprocess)
    except Exception as e:
        logger.error("All visual encoders failed (%s)", e)
        return None


def get_text_encoder() -> Any:
    """Load sentence-transformers multilingual encoder (Section 41).

    Model: paraphrase-multilingual-mpnet-base-v2
    Supports: French, English, Arabic, and 50+ languages.
    768D output embeddings. 🔒 FROZEN.
    """
    global _text_encoder
    if _text_encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _text_encoder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            # 🔒 FROZEN — never re-train
            for param in _text_encoder.parameters():
                param.requires_grad = False
            logger.info("Loaded text encoder: paraphrase-multilingual-mpnet-base-v2 (🔒 frozen)")
        except Exception as e:
            logger.warning("sentence-transformers unavailable (%s)", e)
    return _text_encoder
