"""
Feature Transforms — Product & User Feature Engineering (Section 41)
=====================================================================
Converts raw PostgreSQL product/user rows into ML-ready tensors
using the exact blueprint specification.

Encoders (Section 41 — Layer Architecture):
  🔒 FROZEN: Marqo-FashionSigLIP (ViT-B-16) → 512D visual embeddings
  🔒 FROZEN: sentence-transformers (paraphrase-multilingual) → 768D text
  ✏️ TRAINABLE: Vendor embeddings → 64D (learned from interactions)
  Deterministic: price norm, freshness decay, cv_score, stock signal

Blueprint reference (Section 41):
  product_to_features() code is specified explicitly in the blueprint.
  This implementation follows that specification exactly.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import torch

from .encoders import get_text_encoder, get_visual_encoder
from .vendor_embeddings import VendorEmbeddingTable

# ─── Module-level state ───────────────────────────────────────────

# Singleton vendor embedding table
_vendor_embeddings = VendorEmbeddingTable(embed_dim=64)

# Category average prices (updated daily by batch job)
_category_avg_prices: dict[int, float] = {}


def set_category_avg_prices(prices: dict[int, float]) -> None:
    """Set category average prices for log-normalization."""
    global _category_avg_prices
    _category_avg_prices = prices


# ─── Product Feature Engineering (Section 41 — exact blueprint) ─────

def product_to_features(
    product: dict[str, Any],
    n_categories: int = 500,
) -> torch.Tensor:
    """Convert a product row into an ML feature vector.

    This follows the EXACT specification from blueprint Section 41:
    [visual_512, text_768, price_1, category_N, freshness_1, cv_score_1, vendor_64, stock_1]

    Args:
        product: Dict from PostgreSQL products table or Pydantic model.
        n_categories: Total number of product categories.

    Returns:
        Feature tensor of dimension: 512 + 768 + 1 + N_cat + 1 + 1 + 64 + 1
    """
    features = []

    # 1. Visual embedding (512D) — Marqo-FashionSigLIP (🔒 FROZEN)
    clip_emb = product.get("clip_embedding")
    if clip_emb is not None:
        visual_feat = torch.tensor(clip_emb, dtype=torch.float32)[:512]
    else:
        encoder = get_visual_encoder()
        if encoder is not None:
            model, preprocess = encoder
            try:
                from PIL import Image
                import httpx
                photos = product.get("photos", [])
                if photos:
                    url = photos[0].get("url", "") if isinstance(photos[0], dict) else str(photos[0])
                    if url:
                        resp = httpx.get(url, timeout=10)
                        img = Image.open(resp)
                        img_tensor = preprocess(img).unsqueeze(0)
                        with torch.no_grad():
                            visual_feat = model.encode_image(img_tensor).squeeze(0)
                    else:
                        visual_feat = torch.zeros(512)
                else:
                    visual_feat = torch.zeros(512)
            except Exception:
                visual_feat = torch.zeros(512)
        else:
            visual_feat = torch.zeros(512)
    features.append(visual_feat)

    # 2. Text embedding (768D) — sentence-transformers multilingual (🔒 FROZEN)
    title = product.get("title", "")
    desc = product.get("description_short", product.get("description", ""))
    text_input = f"{title} {desc}".strip()

    text_encoder = get_text_encoder()
    if text_encoder is not None and text_input:
        with torch.no_grad():
            text_feat = torch.tensor(text_encoder.encode(text_input), dtype=torch.float32)[:768]
    else:
        text_feat = torch.zeros(768)
    features.append(text_feat)

    # 3. Price normalized (1D) — log(price / avg_category_price) (Section 41)
    price = float(product.get("base_price", product.get("price", 0)))
    cat_id = int(product.get("category_id", 0))
    avg_price = _category_avg_prices.get(cat_id, 50.0)
    price_norm = math.log1p(price) - math.log1p(avg_price)
    features.append(torch.tensor([price_norm], dtype=torch.float32))

    # 4. Category one-hot (N_categories D)
    cat_vec = torch.zeros(n_categories)
    if 0 <= cat_id < n_categories:
        cat_vec[cat_id] = 1.0
    features.append(cat_vec)

    # 5. Freshness decay (1D) — exp(-age_hours / 168) — 7-day decay (Section 41)
    created = product.get("created_at")
    if created:
        if isinstance(created, str):
            try:
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            except ValueError:
                created = None
        if created and hasattr(created, "timestamp"):
            age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        else:
            age_hours = 48.0  # default 2 days
    else:
        age_hours = 48.0
    freshness = math.exp(-age_hours / 168.0)  # 7-day decay constant
    features.append(torch.tensor([freshness], dtype=torch.float32))

    # 6. CV Score (1D) — SightEngine quality score (Section 15)
    cv_score = float(product.get("cv_score", 0.5))
    features.append(torch.tensor([cv_score], dtype=torch.float32))

    # 7. Vendor embedding (64D) — ✏️ TRAINABLE from scratch (Section 41)
    vendor_id = str(product.get("vendor_id", "unknown"))
    vendor_feat = _vendor_embeddings.get(vendor_id)
    features.append(vendor_feat)

    # 8. Stock signal (1D) — [0,1] normalized (Section 41)
    stock = min(int(product.get("base_stock", product.get("stock", 100))), 100)
    features.append(torch.tensor([stock / 100.0], dtype=torch.float32))

    # Concatenate: 512 + 768 + 1 + N_cat + 1 + 1 + 64 + 1 = ~1348 + N_cat
    return torch.cat(features)


# ─── User Feature Engineering ───────────────────────────────────────

def user_to_features(
    user: dict[str, Any],
    n_categories: int = 500,
) -> torch.Tensor:
    """Convert a user profile into feature vector.

    Components:
      - User embedding (256D) from Two-Tower model
      - Interest categories (N_cat weights)
      - Price range preference (3D: min, max, avg)
      - RFM features (3D: recency, frequency, monetary)
      - Device/context (2D)
    """
    features = []

    # User embedding (256D) — from Two-Tower model
    user_emb = user.get("user_embedding")
    if user_emb is not None:
        features.append(torch.tensor(user_emb, dtype=torch.float32)[:256])
    else:
        features.append(torch.zeros(256))

    # Interest categories (N_cat)
    interest_vec = torch.zeros(n_categories)
    interests = user.get("interest_categories", {})
    if isinstance(interests, dict):
        for cat_id_str, score in interests.items():
            try:
                cat_id = int(cat_id_str)
                if 0 <= cat_id < n_categories:
                    interest_vec[cat_id] = float(score)
            except (ValueError, TypeError):
                pass
    features.append(interest_vec)

    # Price range (3D)
    price_pref = user.get("price_range_pref", {})
    price_min = float(price_pref.get("min", 0))
    price_max = float(price_pref.get("max", 100))
    price_avg = float(price_pref.get("avg", 50))
    features.append(torch.tensor([
        math.log1p(price_min), math.log1p(price_max), math.log1p(price_avg),
    ], dtype=torch.float32))

    # RFM features (3D) — Recency, Frequency, Monetary
    rfm_recency = min(int(user.get("rfm_recency", 30)), 365) / 365.0
    rfm_frequency = min(int(user.get("rfm_frequency", 0)), 100) / 100.0
    rfm_monetary = math.log1p(float(user.get("rfm_monetary", 0)))
    features.append(torch.tensor([rfm_recency, rfm_frequency, rfm_monetary], dtype=torch.float32))

    # Device context (2D: is_ios, is_android)
    device = user.get("device_os", "android")
    features.append(torch.tensor([
        1.0 if device == "ios" else 0.0,
        1.0 if device == "android" else 0.0,
    ], dtype=torch.float32))

    return torch.cat(features)
