"""
Dataset Loaders — HuggingFace + Kaggle Integration
====================================================
Provides standardized loaders for all blueprint datasets.

Modules:
  - configs: Dataset configurations and registry
  - loaders: HuggingFace loaders, Dataset classes, convenience functions
"""

from .configs import (
    ALL_DATASETS,
    DatasetConfig,
    DatasetSource,
    get_dataset_config,
    list_datasets_by_section,
)
from .loaders import (
    BehaviorSequenceDataset,
    FashionEmbeddingDataset,
    get_fashion_clip_fallback_id,
    get_fashion_siglip_model_id,
    load_alibaba_userbehavior,
    load_amazon_reviews,
    load_deepfashion,
    load_food101,
    load_hf_dataset,
    load_plantnet,
)

__all__ = [
    # Configs
    "ALL_DATASETS",
    "DatasetConfig",
    "DatasetSource",
    "get_dataset_config",
    "list_datasets_by_section",
    # Loaders
    "load_hf_dataset",
    "load_alibaba_userbehavior",
    "load_amazon_reviews",
    "load_food101",
    "load_deepfashion",
    "load_plantnet",
    # Dataset classes
    "BehaviorSequenceDataset",
    "FashionEmbeddingDataset",
    # Model IDs
    "get_fashion_siglip_model_id",
    "get_fashion_clip_fallback_id",
]
