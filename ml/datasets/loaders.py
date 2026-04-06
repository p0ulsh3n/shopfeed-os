"""
Dataset Loaders — HuggingFace + Kaggle Integration
====================================================
Provides standardized loaders for all blueprint datasets.
Each loader returns data in a format ready for the training pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
from torch.utils.data import Dataset

from .configs import (
    ALL_DATASETS,
    DatasetConfig,
    DatasetSource,
    get_dataset_config,
)

logger = logging.getLogger(__name__)


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
        # BUG #10 FIX: lowered default from 5 to 1 so cold-start users (<5
        # interactions) are included in training rather than silently excluded.
        # The padding mask already handles short histories correctly.
        min_seq_len: int = 1,
    ):
        self.interactions = interactions
        self.max_seq_len = max_seq_len
        self.n_tasks = n_tasks
        self.min_seq_len = min_seq_len
        self._preprocess()

    def _preprocess(self) -> None:
        """Group interactions by user and build sequences."""
        from collections import defaultdict

        user_sequences: dict[Any, list[dict]] = defaultdict(list)
        for interaction in self.interactions:
            uid = interaction.get("user_id", interaction.get("visitorid", 0))
            user_sequences[uid].append(interaction)

        # Sort each user's interactions by timestamp
        now_ts = time.time()
        self.samples: list[dict[str, Any]] = []
        for uid, seq in user_sequences.items():
            seq.sort(key=lambda x: x.get("timestamp", 0))

            # BUG #10 FIX: use min_seq_len (default=1) instead of hardcoded 5.
            # Cold-start users with 1–4 interactions are now included; their
            # histories will be short but the padding mask handles this correctly.
            min_start = min(self.min_seq_len, len(seq))
            for i in range(min_start, len(seq)):
                history = seq[max(0, i - self.max_seq_len):i]
                target = seq[i]

                # BUG #7 FIX: previously always torch.zeros(5).
                # Compute real dense features from data available at dataset build time.
                # These match what the serving feature store provides at inference:
                #   [0] log-normalized price       (log1p(price) / 10)
                #   [1] freshness decay            (exp(-age_hours / 168))
                #   [2] CV quality score           (0.0–1.0)
                #   [3] normalized stock ratio     (stock / 100, capped at 1.0)
                #   [4] vendor account_weight      (0.0–1.0, max=5.0)
                price = float(target.get("price", target.get("base_price", 0.0)))
                ts    = float(target.get("timestamp", now_ts))
                age_hours = max((now_ts - ts) / 3600.0, 0.0)
                freshness = math.exp(-age_hours / 168.0)  # half-life = 1 week
                cv_score  = float(target.get("cv_score", 0.5))
                stock     = float(target.get("stock", target.get("base_stock", 10)))
                weight    = float(target.get("account_weight", target.get("seller_weight", 1.0)))

                self.samples.append({
                    "behavior_ids": [s.get("item_id", s.get("itemid", 0)) for s in history],
                    "candidate_id": target.get("item_id", target.get("itemid", 0)),
                    "candidate_cat": target.get("category_id", target.get("item_category", 0)),
                    "label": 1 if target.get("behavior_type", target.get("event", "")) in ("buy", "purchase", "transaction") else 0,
                    # Store raw values so __getitem__ can build the tensor
                    "_price": math.log1p(price) / 10.0,
                    "_freshness": freshness,
                    "_cv_score": cv_score,
                    "_stock_ratio": min(stock / 100.0, 1.0),
                    "_weight": min(weight / 5.0, 1.0),
                })

        logger.info(
            "Built %d training samples from %d interactions "
            "(min_seq_len=%d, pos_rate=%.2f%%)",
            len(self.samples), len(self.interactions),
            self.min_seq_len,
            100.0 * sum(1 for s in self.samples if s["label"] == 1) / max(len(self.samples), 1),
        )

    @property
    def pos_weight(self) -> float:
        """BUG #9 FIX: compute positive class weight for BCELoss.

        With ~1.2% purchase rate (Alibaba UserBehavior) the model converges
        to predicting 0 everywhere without pos_weight. This property returns
        the neg/pos ratio so callers can pass it to loss functions:

            loss = DINLoss(pos_weight=torch.tensor([dataset.pos_weight]))
        """
        n_pos = sum(1 for s in self.samples if s["label"] == 1)
        n_neg = len(self.samples) - n_pos
        if n_pos == 0:
            return 1.0  # Degenerate case: no positives, no correction needed
        return n_neg / n_pos

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
            # BUG #7 FIX: real dense features instead of torch.zeros(5)
            "dense_features": torch.tensor([
                sample["_price"],
                sample["_freshness"],
                sample["_cv_score"],
                sample["_stock_ratio"],
                sample["_weight"],
            ], dtype=torch.float32),
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


# ─── Batch Loaders for pretrain.py & finetune_run.py ────────────


class AlibabaUserBehaviorLoader:
    """Wraps Alibaba UserBehavior HuggingFace dataset into batches
    of (user_features, item_features, labels) for Two-Tower pre-training.

    This loader streams REAL data from HuggingFace — zero simulation.
    """

    def __init__(self, batch_size: int = 256, max_samples: int | None = None):
        self.batch_size = batch_size
        self.max_samples = max_samples
        self._dataset = None

    def _load(self) -> None:
        if self._dataset is not None:
            return
        self._dataset = load_hf_dataset(
            "alibaba_userbehavior", split="train", streaming=True,
        )

    def get_batches(self):
        """Yields batches of {user_features, item_features, labels}."""
        self._load()
        batch_user: list = []
        batch_item: list = []
        batch_labels: list = []
        n_seen = 0

        for row in self._dataset:
            if self.max_samples and n_seen >= self.max_samples:
                break

            # Map raw Alibaba fields to feature vectors
            user_id = int(row.get("user_id", row.get("userid", 0)))
            item_id = int(row.get("item_id", row.get("itemid", 0)))
            cat_id = int(row.get("category_id", row.get("item_category", 0)))
            btype = row.get("behavior_type", row.get("event", "pv"))
            ts = float(row.get("timestamp", 0))

            # Build minimal dense features (real data, no simulation)
            # user_features: [user_id_hash, activity_hour, day_of_week, ...]
            import math
            hour = (ts % 86400) / 3600.0 if ts > 0 else 12.0
            day = (ts % 604800) / 86400.0 if ts > 0 else 3.0
            user_feat = torch.zeros(764)
            user_feat[0] = (user_id % 1000) / 1000.0  # hashed user signal
            user_feat[1] = hour / 24.0
            user_feat[2] = day / 7.0
            # Sparse category preference in user vector
            cat_idx = min(cat_id % 100, 99)
            user_feat[10 + cat_idx] = 1.0

            # item_features: [item_id_hash, category, ...]
            item_feat = torch.zeros(1348)
            item_feat[0] = (item_id % 10000) / 10000.0
            item_feat[1] = (cat_id % 500) / 500.0

            # Labels: 1 for purchase, 0 for view
            label = 1.0 if btype in ("buy", "purchase", "transaction") else 0.0

            batch_user.append(user_feat)
            batch_item.append(item_feat)
            batch_labels.append(label)
            n_seen += 1

            if len(batch_user) >= self.batch_size:
                yield {
                    "user_features": torch.stack(batch_user),
                    "item_features": torch.stack(batch_item),
                    "labels": torch.tensor(batch_labels, dtype=torch.float32),
                }
                batch_user, batch_item, batch_labels = [], [], []

        # Last partial batch
        if batch_user:
            yield {
                "user_features": torch.stack(batch_user),
                "item_features": torch.stack(batch_item),
                "labels": torch.tensor(batch_labels, dtype=torch.float32),
            }


class AlibabaSequenceLoader:
    """Wraps Alibaba UserBehavior into behavior sequences for DIN/DIEN/BST.

    Groups interactions by user, sorts by timestamp, builds sequences.
    This loader streams REAL data from HuggingFace — zero simulation.
    """

    def __init__(
        self,
        batch_size: int = 128,
        max_seq_len: int = 200,
        max_users: int | None = None,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_users = max_users
        self._built = False
        self._dataset: BehaviorSequenceDataset | None = None

    def _build(self) -> None:
        if self._built:
            return

        # Stream raw interactions from HuggingFace
        raw_ds = load_hf_dataset(
            "alibaba_userbehavior", split="train", streaming=True,
        )

        interactions: list[dict] = []
        user_counts: dict[int, int] = {}
        for row in raw_ds:
            uid = int(row.get("user_id", row.get("userid", 0)))

            # Limit users if specified
            if self.max_users:
                if uid not in user_counts and len(user_counts) >= self.max_users:
                    continue
                user_counts.setdefault(uid, 0)
                user_counts[uid] += 1

            interactions.append({
                "user_id": uid,
                "item_id": int(row.get("item_id", row.get("itemid", 0))),
                "category_id": int(row.get("category_id", row.get("item_category", 0))),
                "behavior_type": row.get("behavior_type", row.get("event", "pv")),
                "timestamp": float(row.get("timestamp", 0)),
            })

            # Stop after reasonable size for pre-training
            if len(interactions) >= 5_000_000:
                break

        self._dataset = BehaviorSequenceDataset(
            interactions, max_seq_len=self.max_seq_len, min_seq_len=1,
        )
        self._built = True
        logger.info(
            "AlibabaSequenceLoader built: %d samples from %d interactions",
            len(self._dataset), len(interactions),
        )

    def get_batches(self):
        """Yields batches of behavior sequence dicts."""
        self._build()
        loader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        yield from loader


class ProprietaryInteractionsLoader:
    """Loads proprietary ShopFeed interaction data for fine-tuning.

    Expects pre-processed data at data/finetune/{model_type}/ as:
      - train.parquet  (training set)
      - val.parquet    (validation set — MANDATORY for checkpoint selection)

    If data files don't exist, raises FileNotFoundError to prevent
    silent failures / mock data training.
    """

    def __init__(
        self,
        batch_size: int = 128,
        model_type: str = "din",
        data_dir: str = "data/finetune",
    ):
        import os
        self.batch_size = batch_size
        self.model_type = model_type
        self.data_dir = os.path.join(data_dir, model_type)
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._load()

    def _load(self) -> None:
        import os
        import pandas as pd

        train_path = os.path.join(self.data_dir, "train.parquet")
        val_path = os.path.join(self.data_dir, "val.parquet")

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Fine-tuning training data not found: {train_path}\n"
                f"Run `python -m scripts.prepare_data --finetune` first."
            )

        # Load training data
        train_df = pd.read_parquet(train_path)
        train_interactions = train_df.to_dict("records")

        if self.model_type in ("din", "dien", "bst"):
            self._train_ds = BehaviorSequenceDataset(
                train_interactions, max_seq_len=200, min_seq_len=1,
            )
        else:
            from ml.training.train import InteractionDataset
            self._train_ds = InteractionDataset(train_path)

        # Load validation data
        if os.path.exists(val_path):
            val_df = pd.read_parquet(val_path)
            val_interactions = val_df.to_dict("records")
            if self.model_type in ("din", "dien", "bst"):
                self._val_ds = BehaviorSequenceDataset(
                    val_interactions, max_seq_len=200, min_seq_len=1,
                )
            else:
                from ml.training.train import InteractionDataset
                self._val_ds = InteractionDataset(val_path)
        else:
            # Split 85/15 from training if no separate val set
            total = len(self._train_ds)
            val_size = int(total * 0.15)
            train_size = total - val_size
            from torch.utils.data import Subset
            indices = list(range(total))
            self._val_ds = Subset(self._train_ds, indices[train_size:])
            self._train_ds = Subset(self._train_ds, indices[:train_size])

        logger.info(
            "ProprietaryInteractionsLoader: train=%d val=%d (model=%s)",
            len(self._train_ds), len(self._val_ds), self.model_type,
        )

    def get_batches(self):
        """Yields training batches."""
        loader = torch.utils.data.DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        yield from loader

    def get_val_batches(self):
        """Yields validation batches — used for checkpoint selection."""
        loader = torch.utils.data.DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        yield from loader

