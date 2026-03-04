"""
ShopFeed OS — ML Pipeline
===========================
Machine learning pipeline for e-commerce recommendation.

Packages:
  - feature_store: Product & user feature engineering (Section 41)
  - datasets:      HuggingFace / Kaggle dataset loaders
  - training:      GPU training orchestrator (DIN, DIEN, BST, DeepFM, MTL, Two-Tower)
  - serving:       Model registry and inference serving
  - monolith:      Online streaming training (Vitesse 2, Section 14)
"""

__all__ = [
    "feature_store",
    "datasets",
    "training",
    "serving",
    "monolith",
]
