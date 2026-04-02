"""
Feature Store — Product & User Feature Engineering (Section 41)
================================================================
Converts raw PostgreSQL product/user rows into ML-ready tensors.

Modules:
  - encoders:          Lazy-loaded visual & text encoders (🔒 FROZEN)
  - vendor_embeddings: Trainable vendor embedding table (✏️ TRAINABLE)
  - transforms:        product_to_features() & user_to_features()
  - temporal:          Circadian temporal features (t.md §2)
  - desire_graph:      6D desire vector (t.md §5)
"""

from .encoders import get_text_encoder, get_visual_encoder
from .transforms import product_to_features, set_category_avg_prices, user_to_features
from .vendor_embeddings import VendorEmbeddingTable
from .temporal import compute_temporal_features, get_vulnerability_multiplier
from .desire_graph import DesireGraph

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
    # t.md: Temporal + Desire Graph
    "compute_temporal_features",
    "get_vulnerability_multiplier",
    "DesireGraph",
]
