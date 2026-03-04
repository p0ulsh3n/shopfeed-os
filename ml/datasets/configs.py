"""
Dataset Configurations — All Blueprint Datasets (Section 32, 35)
=================================================================
Centralized config for every dataset referenced in the blueprint.
Provides HuggingFace Hub IDs, Kaggle slugs, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DatasetSource(str, Enum):
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    URL = "url"
    INTERNAL = "internal"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    source: DatasetSource
    identifier: str                   # HF repo_id, Kaggle slug, or URL
    description: str
    section: str                      # Blueprint section reference
    usage: str                        # What we use it for
    size: str = ""                    # Human-readable size
    license: str = ""
    subset: str | None = None         # HF subset if needed
    split_map: dict[str, str] = field(default_factory=dict)


# ─── Section 32 — General E-Commerce Datasets ──────────────────────

ALIBABA_USER_BEHAVIOR = DatasetConfig(
    name="Alibaba UserBehavior",
    source=DatasetSource.HUGGINGFACE,
    identifier="kk15507/alibaba_user_behavior",
    description="100M+ behaviors from Taobao: click, fav, cart, buy. "
                "9-day sequences. Reference for DIN/DIEN/BST training.",
    section="Section 32",
    usage="Pre-train DIN/DIEN/BST behavior models. "
          "Structure: {user_id, item_id, category_id, behavior_type, timestamp}",
    size="100M+ behaviors, 987K users, 4.1M items",
    license="CC BY-NC-SA 4.0",
)

AMAZON_REVIEWS = DatasetConfig(
    name="Amazon Product Reviews 2023",
    source=DatasetSource.HUGGINGFACE,
    identifier="McAuley-Lab/Amazon-Reviews-2023",
    description="233M+ reviews across 29 categories. "
                "Content-based filtering and CF matrix.",
    section="Section 32",
    usage="NLP text embeddings from review text. "
          "User-item rating matrix for collaborative filtering.",
    size="233M+ reviews, 43M users, 9.9M items",
    license="Apache 2.0",
)

RETAILROCKET = DatasetConfig(
    name="Retailrocket",
    source=DatasetSource.KAGGLE,
    identifier="retailrocket/ecommerce-dataset",
    description="2.7M e-commerce events: views, carts, purchases. "
                "Transformable into behavioral sequences for DIEN.",
    section="Section 32",
    usage="Behavioral sequence training for DIEN. "
          "Events: {timestamp, visitorid, event, itemid, transactionid}",
    size="2.7M events, 1.4M visitors, 417K items",
    license="Public",
)


# ─── Section 35 — Fashion Datasets ─────────────────────────────────

DEEPFASHION = DatasetConfig(
    name="DeepFashion",
    source=DatasetSource.HUGGINGFACE,
    identifier="detection-datasets/deepfashion",
    description="800K+ images, 50 categories, 1000 attributes. "
                "Reference for fashion CLIP pre-training.",
    section="Section 35",
    usage="Pre-train CLIP fashion embeddings and category classifier. "
          "Consumer-to-shop retrieval.",
    size="800K+ images, 50 categories, 1000 attributes",
    license="Non-commercial",
)

DEEPFASHION2 = DatasetConfig(
    name="DeepFashion2",
    source=DatasetSource.HUGGINGFACE,
    identifier="Yufei-Gao/DeepFashion2",
    description="491K images with segmentation, keypoints, mask annotations. "
                "13 categories, 801K items.",
    section="Section 35",
    usage="Detect and segment clothing in vendor videos. "
          "Auto-tag products in feed content.",
    size="491K images, 13 categories, 801K annotated items",
    license="Non-commercial",
)

MARQO_FASHION_SIGLIP = DatasetConfig(
    name="Marqo-FashionSigLIP",
    source=DatasetSource.HUGGINGFACE,
    identifier="Marqo/marqo-fashionSigLIP",
    description="Best fashion embedding model 2025. +22% recall@1 text-to-image. "
                "GCL-trained on categories, styles, colors, materials.",
    section="Section 35",
    usage="BASE for all fashion visual embeddings. "
          "Replaces generic CLIP everywhere in Mode category. Apache 2.0.",
    size="ViT-B-16-SigLIP architecture",
    license="Apache 2.0",
)

FASHION_CLIP_2 = DatasetConfig(
    name="FashionCLIP 2.0",
    source=DatasetSource.HUGGINGFACE,
    identifier="patrickjohncyh/fashion-clip",
    description="Farfetch-trained, LAION-2B base (5x OpenAI CLIP). "
                "Fallback if FashionSigLIP unavailable.",
    section="Section 35",
    usage="Backup embedding model for fashion. "
          "Best for product photos on white background.",
    size="ViT-B/32 architecture, 800K+ fashion products",
    license="Public",
)

IMATERIALIST = DatasetConfig(
    name="iMaterialist Fashion Attribute",
    source=DatasetSource.KAGGLE,
    identifier="c/imaterialist-fashion-2019-FGVC6",
    description="1M+ images, 228 fine-grained attributes across 8 groups. "
                "Google Brain + Cornell Tech, CVPR 2019.",
    section="Section 35",
    usage="Fine-grained attribute classifier: material, color, pattern, "
          "style, occasion, construction details. Auto-enrich product pages.",
    size="1M+ images, 228 attributes, 8 attribute groups",
    license="Public",
)

FASHIONPEDIA = DatasetConfig(
    name="Fashionpedia",
    source=DatasetSource.KAGGLE,
    identifier="c/fashionpedia2020",
    description="48K images with pixel-level attribute segmentation. "
                "294 attributes. Cornell Tech, ECCV 2020.",
    section="Section 35",
    usage="Fine-grained segmentation for virtual try-on. "
          "Identify each component of clothing in photos/videos.",
    size="48K annotated images, 27 categories, 294 attributes",
    license="Public",
)

POLYVORE = DatasetConfig(
    name="Polyvore Outfits",
    source=DatasetSource.URL,
    identifier="https://github.com/xthan/polyvore-dataset",
    description="21K complete outfits with style coherence scoring. "
                "365K items paired as outfits.",
    section="Section 35",
    usage="Cross-sell 'Complete the Look' recommendations. "
          "Suggest matching shoes + bag when user buys a dress.",
    size="21K outfits, 365K items",
    license="Academic",
)

FASHION_IQ = DatasetConfig(
    name="Fashion IQ",
    source=DatasetSource.URL,
    identifier="https://github.com/XiaoxiaoGuo/fashion-iq",
    description="77K triplets with natural language feedback. "
                "CVPR 2021. Composed Image Retrieval.",
    section="Section 35",
    usage="Conversational search: 'Like this but with long sleeves and navy'. "
          "Fine-tune CLIP with NL feedback for nuanced recommendations.",
    size="77K triplets, 3 categories",
    license="Academic",
)

FASHION200K = DatasetConfig(
    name="Fashion200K",
    source=DatasetSource.URL,
    identifier="https://github.com/xthan/fashion-200k",
    description="200K+ fashion images from Lyst.com with product descriptions. "
                "9 detection classes.",
    section="Section 35",
    usage="Visual-semantic embedding for attribute-based retrieval.",
    size="200K+ images, 9 detection classes",
    license="Academic",
)


# ─── Section 35 — Food Datasets ────────────────────────────────────

FOOD_101 = DatasetConfig(
    name="Food-101",
    source=DatasetSource.HUGGINGFACE,
    identifier="ethz/food101",
    description="101K images across 101 food categories. "
                "EPFL/ETH Zürich standard benchmark.",
    section="Section 35",
    usage="Auto-classify food in vendor videos/photos. "
          "Auto-tag food product listings.",
    size="101K images, 101 categories, 1000 images/class",
    license="Public",
)

RECIPE1M_PLUS = DatasetConfig(
    name="Recipe1M+",
    source=DatasetSource.URL,
    identifier="http://pic2recipe.csail.mit.edu/",
    description="1M+ recipes with 13M images. Cross-modal image↔recipe. "
                "MIT CSAIL 2019.",
    section="Section 35",
    usage="Enrich food product descriptions from photos. "
          "Cross-sell ingredients based on recipe match.",
    size="1M+ recipes, 13M images",
    license="Academic",
)

VIREO_FOOD_172 = DatasetConfig(
    name="VIREO Food-172",
    source=DatasetSource.URL,
    identifier="http://vireo.cs.cityu.edu.hk/VireoFood172/",
    description="110K images, 172 food categories (mainly Asian), "
                "353 identifiable ingredients.",
    section="Section 35",
    usage="Asian cuisine specialization. Essential for target markets "
          "with strong Asian food community.",
    size="110K images, 172 categories, 353 ingredients",
    license="Academic",
)


# ─── Section 35 — Wellness/Plant Datasets ──────────────────────────

PLANTNET = DatasetConfig(
    name="PlantNet",
    source=DatasetSource.HUGGINGFACE,
    identifier="bertiqwerty/plantnet",
    description="8M+ images, 300K species worldwide. "
                "INRIA/CIRAD France. Global plant recognition.",
    section="Section 35",
    usage="Auto-identify medicinal plants in product photos. "
          "Verify product matches announced plant species.",
    size="8M+ images, 300K species",
    license="Public",
)


# ─── Registry ──────────────────────────────────────────────────────

ALL_DATASETS: dict[str, DatasetConfig] = {
    # E-Commerce General (Section 32)
    "alibaba_userbehavior": ALIBABA_USER_BEHAVIOR,
    "amazon_reviews": AMAZON_REVIEWS,
    "retailrocket": RETAILROCKET,
    # Fashion (Section 35)
    "deepfashion": DEEPFASHION,
    "deepfashion2": DEEPFASHION2,
    "marqo_fashion_siglip": MARQO_FASHION_SIGLIP,
    "fashion_clip_2": FASHION_CLIP_2,
    "imaterialist": IMATERIALIST,
    "fashionpedia": FASHIONPEDIA,
    "polyvore": POLYVORE,
    "fashion_iq": FASHION_IQ,
    "fashion200k": FASHION200K,
    # Food (Section 35)
    "food101": FOOD_101,
    "recipe1m": RECIPE1M_PLUS,
    "vireo_food_172": VIREO_FOOD_172,
    # Wellness
    "plantnet": PLANTNET,
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset config by name. Raises KeyError if not found."""
    if name not in ALL_DATASETS:
        available = ", ".join(sorted(ALL_DATASETS.keys()))
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")
    return ALL_DATASETS[name]


def list_datasets_by_section(section: str) -> list[DatasetConfig]:
    """List all datasets for a given blueprint section."""
    return [d for d in ALL_DATASETS.values() if d.section == section]
