"""
Dataset Loaders — HuggingFace + Kaggle Integration
====================================================
Provides standardized loaders for all blueprint datasets.
Each loader returns data in a format ready for the training pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset

from .configs import (
    ALL_DATASETS,
    DatasetConfig,
    DatasetSource,
    get_dataset_config,
)

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/datasets")


# ─── HuggingFace Loader ────────────────────────────────────────────

def load_hf_dataset(
    name: str,
    split: str = "train",
    streaming: bool = True,
    cache_dir: str | None = None,
) -> Any:
    """Load a HuggingFace dataset by config name.

    Args:
        name:      Key from ALL_DATASETS (e.g. 'alibaba_userbehavior')
        split:     Dataset split ('train', 'test', 'validation')
        streaming: If True, use streaming mode (no full download)
        cache_dir: Custom cache directory

    Returns:
        HuggingFace Dataset or IterableDataset
    """
    from datasets import load_dataset

    config = get_dataset_config(name)
    if config.source != DatasetSource.HUGGINGFACE:
        raise ValueError(
            f"Dataset '{name}' is not on HuggingFace (source={config.source}). "
            f"Use the appropriate loader."
        )

    logger.info(
        "Loading HuggingFace dataset: %s (%s) split=%s streaming=%s",
        config.name, config.identifier, split, streaming,
    )

    kwargs: dict[str, Any] = {
        "path": config.identifier,
        "split": split,
        "streaming": streaming,
    }
    if config.subset:
        kwargs["name"] = config.subset
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    return load_dataset(**kwargs)


# ─── Behavior Sequence Dataset (for DIN/DIEN/BST) ──────────────────

class BehaviorSequenceDataset(Dataset):
    """Converts raw interaction data into behavior sequences for DIN/DIEN/BST.

    Each sample: (behavior_item_ids, candidate_id, candidate_cat, dense_features, labels)
    """

    def __init__(
        self,
        interactions: list[dict[str, Any]],
        max_seq_len: int = 200,
        n_tasks: int = 3,
    ):
        self.interactions = interactions
        self.max_seq_len = max_seq_len
        self.n_tasks = n_tasks
        self._preprocess()

    def _preprocess(self) -> None:
        """Group interactions by user and build sequences."""
        from collections import defaultdict

        user_sequences: dict[Any, list[dict]] = defaultdict(list)
        for interaction in self.interactions:
            uid = interaction.get("user_id", interaction.get("visitorid", 0))
            user_sequences[uid].append(interaction)

        # Sort each user's interactions by timestamp
        self.samples: list[dict[str, Any]] = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x.get("timestamp", 0))

            # For each interaction after the first few, create a training sample
            for i in range(min(5, len(seq)), len(seq)):
                history = seq[max(0, i - self.max_seq_len):i]
                target = seq[i]

                self.samples.append({
                    "behavior_ids": [s.get("item_id", s.get("itemid", 0)) for s in history],
                    "candidate_id": target.get("item_id", target.get("itemid", 0)),
                    "candidate_cat": target.get("category_id", target.get("item_category", 0)),
                    "label": 1 if target.get("behavior_type", target.get("event", "")) in ("buy", "purchase", "transaction") else 0,
                })

        logger.info("Built %d training samples from %d interactions", len(self.samples), len(self.interactions))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Pad behavior sequence
        behavior = sample["behavior_ids"][-self.max_seq_len:]
        padded = [0] * (self.max_seq_len - len(behavior)) + behavior
        mask = [False] * (self.max_seq_len - len(behavior)) + [True] * len(behavior)

        return {
            "behavior_ids": torch.tensor(padded, dtype=torch.long),
            "candidate_id": torch.tensor(sample["candidate_id"], dtype=torch.long),
            "candidate_cat": torch.tensor(sample["candidate_cat"], dtype=torch.long),
            "behavior_mask": torch.tensor(mask, dtype=torch.bool),
            "dense_features": torch.zeros(5),  # placeholder — filled by feature store
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


# ─── Vision Dataset (for Fashion CLIP fine-tuning) ──────────────────

class FashionEmbeddingDataset(Dataset):
    """Wraps a HuggingFace fashion dataset for embedding extraction.

    Returns (image, text_description, category_label) tuples.
    """

    def __init__(
        self,
        hf_dataset: Any,
        image_key: str = "image",
        label_key: str = "label",
        max_samples: int | None = None,
    ):
        self.dataset = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        return {
            "image": item[self.image_key],
            "label": item.get(self.label_key, -1),
        }


# ─── Convenience Functions ─────────────────────────────────────────

def load_alibaba_userbehavior(
    split: str = "train",
    streaming: bool = True,
) -> Any:
    """Load Alibaba UserBehavior dataset (100M+ behaviors)."""
    return load_hf_dataset("alibaba_userbehavior", split=split, streaming=streaming)


def load_amazon_reviews(
    split: str = "train",
    streaming: bool = True,
) -> Any:
    """Load Amazon Reviews 2023 (233M+ reviews)."""
    return load_hf_dataset("amazon_reviews", split=split, streaming=streaming)


def load_food101(
    split: str = "train",
    streaming: bool = False,
) -> Any:
    """Load Food-101 dataset (101K images)."""
    return load_hf_dataset("food101", split=split, streaming=streaming)


def load_deepfashion(
    split: str = "train",
    streaming: bool = False,
) -> Any:
    """Load DeepFashion dataset."""
    return load_hf_dataset("deepfashion", split=split, streaming=streaming)


def load_plantnet(
    split: str = "train",
    streaming: bool = True,
) -> Any:
    """Load PlantNet dataset (8M+ plant images)."""
    return load_hf_dataset("plantnet", split=split, streaming=streaming)


def get_fashion_siglip_model_id() -> str:
    """Get the HuggingFace model ID for Marqo-FashionSigLIP.

    This is the BEST fashion embedding model (2025):
    +22% recall@1 text-to-image vs FashionCLIP 2.0
    """
    return "Marqo/marqo-fashionSigLIP"


def get_fashion_clip_fallback_id() -> str:
    """Get the fallback FashionCLIP 2.0 model ID."""
    return "patrickjohncyh/fashion-clip"
