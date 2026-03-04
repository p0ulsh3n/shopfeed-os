"""
Feature Store — Product & User Feature Engineering (Section 41)
================================================================
Converts raw PostgreSQL product/user rows into ML-ready tensors.

Modules:
  - encoders:          Lazy-loaded visual & text encoders (🔒 FROZEN)
  - vendor_embeddings: Trainable vendor embedding table (✏️ TRAINABLE)
  - transforms:        product_to_features() & user_to_features()
"""

from .encoders import get_text_encoder, get_visual_encoder
from .transforms import product_to_features, set_category_avg_prices, user_to_features
from .vendor_embeddings import VendorEmbeddingTable

__all__ = [
    # Encoders
    "get_visual_encoder",
    "get_text_encoder",
    # Transforms
    "product_to_features",
    "user_to_features",
    "set_category_avg_prices",
    # Vendor
    "VendorEmbeddingTable",
]
