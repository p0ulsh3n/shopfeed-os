"""
Category Router — Gap 5: Category-based partition routing (Pailitao pattern)
=============================================================================
Before searching the entire vector index, predict the product category from
the query (image or text) and restrict the search to the relevant Milvus
partition. This reduces the search space by 10-100x (Alibaba Pailitao pattern).

Architecture:
    Query (image or text)
      → CLIP zero-shot classification → top-K category predictions
      → Build Milvus filter expression: category_id IN [predicted_cats]
      → Pass to vector search (pre-filtered)

Best practices 2026:
    - Milvus partition_key on category_id for automatic partition routing
    - CLIP zero-shot as category predictor (no training needed)
    - Fallback to full-index search if confidence < threshold
    - Cache category predictions per query hash (Redis, TTL 300s)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Top-level product categories (L1) — matches your N_CATEGORIES=200
# In production, loaded from catalog DB. Here we keep the most common ones
# for CLIP zero-shot classification prompts.
DEFAULT_CATEGORY_LABELS = {
    1: "clothing and fashion",
    2: "shoes and footwear",
    3: "bags and accessories",
    4: "jewelry and watches",
    5: "beauty and cosmetics",
    6: "electronics and gadgets",
    7: "home and furniture",
    8: "sports and fitness",
    9: "toys and games",
    10: "food and beverages",
    11: "books and stationery",
    12: "automotive parts",
    13: "health and wellness",
    14: "baby and kids",
    15: "art and crafts",
}

# Confidence threshold — below this, search the full index
CATEGORY_CONFIDENCE_THRESHOLD = 0.25

# Max categories to include in the partition filter
MAX_PREDICTED_CATEGORIES = 3


class CategoryRouter:
    """Predict product category from query to narrow Milvus search scope.

    This implements the Alibaba Pailitao pattern: before searching
    billions of vectors, predict the category and only search within
    that category's partition. Reduces search space by 10-100x.

    Usage:
        router = CategoryRouter()
        filter_expr = router.predict_from_image(image_url)
        # → 'category_id IN [1, 3]'
        # Pass this to Milvus search as filter_expr
    """

    def __init__(
        self,
        category_labels: dict[int, str] | None = None,
        confidence_threshold: float = CATEGORY_CONFIDENCE_THRESHOLD,
        max_categories: int = MAX_PREDICTED_CATEGORIES,
    ):
        self.category_labels = category_labels or DEFAULT_CATEGORY_LABELS
        self.confidence_threshold = confidence_threshold
        self.max_categories = max_categories
        self._clip_model = None
        self._tokenizer = None
        self._text_features = None  # Pre-computed text embeddings for categories
        self._cache: dict[str, str] = {}  # query_hash → filter_expr

    def _load_clip(self):
        """Lazy-load CLIP for zero-shot classification."""
        if self._clip_model is not None:
            return

        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._clip_model = (model, preprocess)
            self._tokenizer = tokenizer

            # Pre-compute text features for all category labels
            prompts = [
                f"a photo of {label}" for label in self.category_labels.values()
            ]
            tokens = tokenizer(prompts)
            with torch.no_grad():
                self._text_features = model.encode_text(tokens)
                self._text_features /= self._text_features.norm(dim=-1, keepdim=True)

            logger.info(
                "CategoryRouter: CLIP loaded, %d categories indexed",
                len(self.category_labels),
            )
        except Exception as e:
            logger.warning("CategoryRouter: CLIP load failed: %s", e)

    def _get_cache_key(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def predict_from_image(self, image_url: str) -> Optional[str]:
        """Predict category from image → Milvus filter expression.

        Returns:
            Filter expression like 'category_id IN [1, 3, 5]'
            or None if confidence is too low (search full index).
        """
        cache_key = self._get_cache_key(f"img:{image_url}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._load_clip()
        if self._clip_model is None or self._text_features is None:
            return None

        try:
            import torch
            from PIL import Image
            import httpx
            import io

            model, preprocess = self._clip_model

            resp = httpx.get(image_url, timeout=10)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0)

            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Cosine similarity with all category text features
                similarities = (image_features @ self._text_features.T).squeeze(0)
                probs = similarities.softmax(dim=-1)

            # Get top categories above threshold
            cat_ids = list(self.category_labels.keys())
            top_indices = probs.argsort(descending=True)[: self.max_categories]

            predicted = []
            for idx in top_indices:
                if probs[idx].item() >= self.confidence_threshold:
                    predicted.append(cat_ids[idx])

            if not predicted:
                return None  # Low confidence → search full index

            filter_expr = f"category_id in {predicted}"
            self._cache[cache_key] = filter_expr
            logger.info(
                "CategoryRouter: image → categories %s (conf=%.2f)",
                predicted,
                probs[top_indices[0]].item(),
            )
            return filter_expr

        except Exception as e:
            logger.warning("CategoryRouter: image prediction failed: %s", e)
            return None

    def predict_from_text(self, query: str) -> Optional[str]:
        """Predict category from text query → Milvus filter expression.

        Returns:
            Filter expression like 'category_id IN [1, 3, 5]'
            or None if confidence is too low.
        """
        cache_key = self._get_cache_key(f"txt:{query}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._load_clip()
        if self._clip_model is None or self._text_features is None:
            return None

        try:
            import torch

            model, _ = self._clip_model
            tokens = self._tokenizer([query])

            with torch.no_grad():
                query_features = model.encode_text(tokens)
                query_features /= query_features.norm(dim=-1, keepdim=True)

                similarities = (query_features @ self._text_features.T).squeeze(0)
                probs = similarities.softmax(dim=-1)

            cat_ids = list(self.category_labels.keys())
            top_indices = probs.argsort(descending=True)[: self.max_categories]

            predicted = []
            for idx in top_indices:
                if probs[idx].item() >= self.confidence_threshold:
                    predicted.append(cat_ids[idx])

            if not predicted:
                return None

            filter_expr = f"category_id in {predicted}"
            self._cache[cache_key] = filter_expr
            logger.info(
                "CategoryRouter: text '%s' → categories %s", query[:50], predicted
            )
            return filter_expr

        except Exception as e:
            logger.warning("CategoryRouter: text prediction failed: %s", e)
            return None

    def build_filter(
        self,
        category_filter: Optional[int] = None,
        predicted_filter: Optional[str] = None,
    ) -> Optional[str]:
        """Combine explicit category filter with predicted filter.

        Priority: explicit > predicted > none (full index).
        """
        if category_filter is not None:
            return f"category_id == {category_filter}"
        return predicted_filter
