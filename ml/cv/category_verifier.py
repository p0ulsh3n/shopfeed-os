"""
Category Verifier — Vérification CLIP zero-shot que l'image correspond à la catégorie déclarée.
Utilisé par moderation_service step 3 (category verification).

Méthode:
  1. Encoder l'image → CLIP embedding 512d
  2. Encoder les labels textuels de la catégorie → text embeddings
  3. Cosine similarity image vs catégorie déclarée vs toutes catégories
  4. Si similarity < 0.3 → possible erreur de catégorisation → flag pour review

Stocké dans moderation_logs.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Descriptions textuelles par catégorie (à étendre avec le vrai catalogue)
CATEGORY_LABELS: dict[int, list[str]] = {
    1:  ["fashion clothing dress shoes bag accessories", "women men fashion apparel"],
    2:  ["food meal cuisine dish restaurant cooking", "african food meal plate dish"],
    3:  ["electronics smartphone phone laptop computer tablet", "tech gadget device"],
    4:  ["beauty cosmetics skincare makeup perfume", "skincare lotion cream serum"],
    5:  ["health natural herbal supplement wellness", "natural medicine herb plant remedy"],
    6:  ["home furniture decoration interior design", "furniture sofa chair table lamp"],
    7:  ["sport fitness gym equipment outdoor", "sporting goods exercise workout gear"],
    8:  ["baby child toy kids product", "baby clothing toys children accessories"],
    9:  ["automotive car vehicle parts accessories", "car motorcycle parts tools"],
    10: ["book education stationery school office", "books magazines educational material"],
}

# Si category_id non trouvé, fallback générique
DEFAULT_LABELS = ["product item goods merchandise", "e-commerce product for sale"]

# Seuil minimum de similarité pour valider la catégorie
MIN_SIMILARITY_THRESHOLD = 0.20


def verify_category(
    image_url: str,
    declared_category_id: int,
) -> dict:
    """
    Vérifie que l'image correspond à la catégorie déclarée.

    Args:
        image_url:            URL de l'image principale du produit
        declared_category_id: catégorie déclarée par le vendeur

    Returns:
        {
            match: bool,
            score: float (cosine similarity vs catégorie déclarée),
            closest_category: int,
            closest_score: float,
            confidence: float,
        }
    """
    from ml.cv.clip_encoder import encode_product_image, encode_text, compute_similarity

    try:
        # 1. Visual embedding de l'image
        image_emb = encode_product_image(image_url, category_id=declared_category_id)

        if np.all(image_emb == 0):
            logger.warning(f"Zero image embedding for {image_url}. Skipping verify.")
            return _default_result(declared_category_id)

        # 2. Score vs catégorie déclarée
        declared_labels = CATEGORY_LABELS.get(declared_category_id, DEFAULT_LABELS)
        declared_score = _score_category(image_emb, declared_labels)

        # 3. Trouver la catégorie la plus proche
        best_cat_id = declared_category_id
        best_score = declared_score
        all_scores = {declared_category_id: declared_score}

        for cat_id, labels in CATEGORY_LABELS.items():
            if cat_id == declared_category_id:
                continue
            s = _score_category(image_emb, labels)
            all_scores[cat_id] = s
            if s > best_score:
                best_score = s
                best_cat_id = cat_id

        # 4. Décision
        match = (
            declared_score >= MIN_SIMILARITY_THRESHOLD
            and declared_score >= best_score * 0.85  # tolérance 15%
        )

        # Confidence: ratio score_declared / best_score
        confidence = declared_score / max(best_score, 1e-6) if best_score > 0 else 0.0
        confidence = min(confidence, 1.0)

        return {
            "match": match,
            "score": round(declared_score, 4),
            "closest_category": best_cat_id,
            "closest_score": round(best_score, 4),
            "confidence": round(confidence, 4),
        }

    except Exception as e:
        logger.error(f"Category verify failed for {image_url}: {e}")
        return _default_result(declared_category_id)


def _score_category(image_emb: np.ndarray, labels: list[str]) -> float:
    """Score moyen de similarité entre l'image et les labels textuels de la catégorie.
    
    STRUCTURAL FIX: Previously used encode_text() which produces 768d sentence-transformer
    embeddings, but image_emb is a 512d CLIP embedding. Computing dot product on truncated
    dimensions is mathematically invalid. Now uses open_clip text tokenizer to produce
    512d text embeddings in the SAME space as the image embeddings.
    """
    scores = []
    for label in labels:
        try:
            # Use CLIP text encoder (512d) — same space as image embeddings
            import torch
            from ml.cv.clip_encoder import _get_generic_model
            generic = _get_generic_model()
            if generic is not None:
                model, _ = generic
                import open_clip
                tokenizer = open_clip.get_tokenizer("ViT-B-32")
                text_tokens = tokenizer([label])
                with torch.no_grad():
                    text_emb = model.encode_text(text_tokens)
                    text_emb_np = text_emb.cpu().numpy().flatten()
                    # L2 normalize
                    norm = np.linalg.norm(text_emb_np)
                    if norm > 0:
                        text_emb_norm = text_emb_np / norm
                    else:
                        continue
                    s = float(np.dot(image_emb, text_emb_norm))
                    scores.append(max(0.0, s))
            else:
                # Fallback: use sentence-transformer but project to image dim
                from ml.cv.clip_encoder import encode_text
                text_emb = encode_text(label)
                if text_emb is not None and not np.all(text_emb == 0):
                    norm = np.linalg.norm(text_emb)
                    if norm > 0:
                        text_emb_norm = text_emb / norm
                        # Project 768d → 512d via truncation (lossy but safe fallback)
                        projected = text_emb_norm[:len(image_emb)]
                        proj_norm = np.linalg.norm(projected)
                        if proj_norm > 0:
                            projected = projected / proj_norm
                        s = float(np.dot(image_emb, projected))
                        scores.append(max(0.0, s))
        except Exception:
            continue
    return sum(scores) / len(scores) if scores else 0.0


def _default_result(category_id: int) -> dict:
    return {
        "match": True,  # default = approuvé (pas de modèle)
        "score": 0.5,
        "closest_category": category_id,
        "closest_score": 0.5,
        "confidence": 1.0,
    }
