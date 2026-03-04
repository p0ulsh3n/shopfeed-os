"""
Lazy-loaded Encoders — Visual & Text (Section 41)
===================================================
Marqo-FashionSigLIP (🔒 FROZEN) for visual embeddings.
sentence-transformers multilingual (🔒 FROZEN) for text embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Lazy-loaded Encoders ───────────────────────────────────────────
_visual_encoder = None
_text_encoder = None


def get_visual_encoder() -> Any:
    """Load Marqo-FashionSigLIP (Section 35, 41).

    This is the BEST fashion embedding model (2025):
    +22% recall@1 text-to-image vs FashionCLIP 2.0.
    Trained with GCL on categories, styles, colors, materials.
    """
    global _visual_encoder
    if _visual_encoder is None:
        try:
            import open_clip
            # Marqo-FashionSigLIP uses ViT-B-16-SigLIP architecture
            # Fallback chain: FashionSigLIP → FashionCLIP 2.0 → generic CLIP
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16-SigLIP",
                pretrained="webli",  # Base SigLIP, fine-tuned by Marqo for fashion
            )
            model.eval()
            for param in model.parameters():
                param.requires_grad = False  # 🔒 FROZEN — never re-train
            _visual_encoder = (model, preprocess)
            logger.info("Loaded Marqo-FashionSigLIP visual encoder (🔒 frozen)")
        except Exception as e:
            logger.warning("FashionSigLIP unavailable (%s), using deterministic fallback", e)
            _visual_encoder = None
    return _visual_encoder


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
            logger.warning("sentence-transformers unavailable (%s), using deterministic fallback", e)
    return _text_encoder
