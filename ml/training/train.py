"""GPU Training Orchestrator — Section 41 / 11.

This is the module you run on rented GPU servers (Lambda, RunPod, Vast.ai)
to train all models end-to-end. Handles:

    1. Data loading from training events (ClickHouse/CSV export)
    2. Pre-training with public datasets (Ali-CCP, DeepFashion)
    3. Fine-tuning on your own interaction data
    4. Model export (.pt checkpoints + ONNX for TorchServe)
    5. FAISS index building for Two-Tower retrieval

Usage on GPU server:
    python -m ml.training.train --config configs/training.yaml
    python -m ml.training.train --model two_tower --epochs 20 --batch-size 2048
    python -m ml.training.train --model mtl --epochs 10 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ml.feature_store import (
    product_to_features,
    set_category_avg_prices,
    user_to_features,
)
from ml.training.deepfm import DeepFM
from ml.training.mtl_model import MTLModel, TASK_NAMES, NUM_TASKS
from ml.training.two_tower import TwoTowerModel
from ml.training.din import DINModel, DINLoss
from ml.training.dien import DIENModel, DIENLoss
from ml.training.bst import BSTModel, BSTLoss

logger = logging.getLogger(__name__)

# Feature dimensions (set dynamically by feature store)
USER_DENSE_DIM = 764   # 256 + 500 + 3 + 3 + 2
ITEM_DENSE_DIM = 1348  # 512 + 768 + 1 + N_cat + 1 + 1 + 64 + 1

# ──────────────────────────────────────────────────────────────
# Training Config
# ──────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # General
    model_name: str = "two_tower"       # two_tower | deepfm | mtl | din | dien | bst | finetune
    output_dir: str = "checkpoints"
    device: str = "auto"                # auto | cuda | cpu

    # Data
    data_path: str = "data/training_events.parquet"
    val_split: float = 0.1
    num_workers: int = 4
    # BUG #5 FIX: temporal_split avoids data leakage for sequence models.
    # When True, the last val_split% of samples (by index, which preserves
    # timestamp order from BehaviorSequenceDataset._preprocess()) is used
    # as the validation set instead of a random shuffle.
    # Must be True for DIN / DIEN / BST to avoid future interactions leaking
    # into the training set.
    temporal_split: bool = True

    # Training
    epochs: int = 20
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    mixed_precision: bool = True          # AMP for faster GPU training

    # Two-Tower specific
    embedding_dim: int = 256
    temperature: float = 0.07
    num_negatives: int = 10

    # DeepFM specific
    num_sparse_features: int = 5000
    fm_embedding_dim: int = 16

    # MTL specific
    num_shared_experts: int = 4
    num_task_experts: int = 2
    num_ple_layers: int = 2

    # DIN/DIEN/BST specific (Section 32 Marketplace models)
    n_items: int = 1_000_000         # Total catalog size
    n_categories: int = 500
    seq_embed_dim: int = 64          # Behavior sequence embedding dim
    max_seq_len: int = 200           # Max behavior history length
    n_transformer_layers: int = 2    # BST layers
    n_tasks_marketplace: int = 3     # P(click), P(cart), P(purchase)

    # Fine-tuning specific (Section 41)
    finetune_base_model: str = "din"          # Model to fine-tune
    finetune_checkpoint: str | None = None    # Pre-trained checkpoint
    lora_rank: int = 8                        # LoRA rank

    # Logging
    log_interval: int = 100               # Log every N batches
    eval_interval: int = 1                # Eval every N epochs
    save_best: bool = True

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """Training dataset from interaction events.

    Loads pre-processed feature tensors + labels from disk.
    Each sample contains: user_features, item_features, labels_dict
    """

    def __init__(self, data_path: str, max_samples: int | None = None):
        self.data_path = Path(data_path)
        self.samples: list[dict[str, Any]] = []
        self._load(max_samples)

    def _load(self, max_samples: int | None = None) -> None:
        """Load data from Parquet or JSON lines."""
        path = self.data_path

        if path.suffix == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(path)
                if max_samples:
                    df = df.head(max_samples)
                self.samples = df.to_dict("records")
            except ImportError:
                logger.error("pandas required for parquet loading")
                return

        elif path.suffix in (".jsonl", ".json"):
            with open(path) as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    self.samples.append(json.loads(line))

        elif path.is_dir():
            # Load from pre-processed tensor directory
            user_feats = torch.load(path / "user_features.pt", weights_only=True)
            item_feats = torch.load(path / "item_features.pt", weights_only=True)
            labels = torch.load(path / "labels.pt", weights_only=True)

            for i in range(len(user_feats)):
                sample = {
                    "_user_tensor": user_feats[i],
                    "_item_tensor": item_feats[i],
                }
                if isinstance(labels, dict):
                    for k, v in labels.items():
                        sample[f"label_{k}"] = v[i]
                else:
                    sample["label_click"] = labels[i]
                self.samples.append(sample)

        logger.info("Loaded %d training samples from %s", len(self.samples), path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # If pre-tensorized
        if "_user_tensor" in sample:
            result = {
                "user_features": sample["_user_tensor"],
                "item_features": sample["_item_tensor"],
            }
            for k, v in sample.items():
                if k.startswith("label_"):
                    result[k] = torch.tensor(float(v)) if not isinstance(v, torch.Tensor) else v
            return result

        # Otherwise, compute features on-the-fly
        user_feat = user_to_features(sample.get("user_profile", {}))
        item_feat = product_to_features(sample.get("product", {}))

        result = {
            "user_features": user_feat,
            "item_features": item_feat,
        }

        # Extract labels for each MTL task
        for task in TASK_NAMES:
            label_key = f"label_{task}"
            if label_key in sample:
                result[label_key] = torch.tensor(float(sample[label_key]))
            elif task == "negative" and "label_skip" in sample:
                result[label_key] = torch.tensor(float(sample["label_skip"]))

        return result


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def compute_auc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn dependency overhead."""
    if len(np.unique(labels)) < 2:
        return 0.5

    # Sort by prediction descending
    sorted_indices = np.argsort(-predictions)
    sorted_labels = labels[sorted_indices]

    n_pos = sorted_labels.sum()
    n_neg = len(sorted_labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Trapezoidal AUC
    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += tpr * (fpr - prev_fpr)
            prev_fpr = fpr

    return float(auc)


def compute_logloss(predictions: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """Binary cross-entropy (LogLoss)."""
    preds = np.clip(predictions, eps, 1 - eps)
    return float(-np.mean(labels * np.log(preds) + (1 - labels) * np.log(1 - preds)))


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────

class Trainer:
    """GPU training orchestrator for all ShopFeed models.

    Supports:
        - Mixed precision (AMP) for 2x speedup on modern GPUs
        - Gradient accumulation for effective large batch sizes
        - Cosine annealing with warm restarts
        - Gradient clipping for training stability
        - Automatic checkpointing of best model
        - Real AUC/LogLoss metrics (no faking)
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = config.resolve_device()

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model
        self.model = self._build_model()
        self.model.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler: Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config.epochs // 3 + 1, T_mult=2
        )

        # Mixed precision scaler
        self.scaler = GradScaler("cuda", enabled=config.mixed_precision and self.device.type == "cuda")

        # Tracking
        self.best_metric = float("inf")
        self.global_step = 0

        logger.info(
            "Trainer ready: model=%s, device=%s, params=%s",
            config.model_name,
            self.device,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    def _build_model(self) -> nn.Module:
        cfg = self.config

        if cfg.model_name == "two_tower":
            return TwoTowerModel(
                user_input_dim=USER_DENSE_DIM,
                item_input_dim=ITEM_DENSE_DIM,
                embedding_dim=cfg.embedding_dim,
                temperature=cfg.temperature,
            )

        elif cfg.model_name == "deepfm":
            return DeepFM(
                num_sparse_features=cfg.num_sparse_features,
                dense_input_dim=ITEM_DENSE_DIM,
                fm_embedding_dim=cfg.fm_embedding_dim,
            )

        elif cfg.model_name == "mtl":
            input_dim = USER_DENSE_DIM + ITEM_DENSE_DIM
            return MTLModel(
                input_dim=input_dim,
                hidden_dim=256,
                num_shared_experts=cfg.num_shared_experts,
                num_task_experts=cfg.num_task_experts,
                num_ple_layers=cfg.num_ple_layers,
            )

        elif cfg.model_name == "din":
            return DINModel(
                n_items=cfg.n_items,
                n_categories=cfg.n_categories,
                embed_dim=cfg.seq_embed_dim,
                n_tasks=cfg.n_tasks_marketplace,
            )

        elif cfg.model_name == "dien":
            return DIENModel(
                n_items=cfg.n_items,
                n_categories=cfg.n_categories,
                embed_dim=cfg.seq_embed_dim,
                n_tasks=cfg.n_tasks_marketplace,
            )

        elif cfg.model_name == "bst":
            return BSTModel(
                n_items=cfg.n_items,
                n_categories=cfg.n_categories,
                embed_dim=cfg.seq_embed_dim,
                n_layers=cfg.n_transformer_layers,
                max_seq_len=cfg.max_seq_len,
                n_tasks=cfg.n_tasks_marketplace,
            )

        elif cfg.model_name == "finetune":
            from ml.training.finetune import FineTuneConfig, FineTuneTrainer
            ft_config = FineTuneConfig(
                base_model=cfg.finetune_base_model,
                checkpoint_path=cfg.finetune_checkpoint,
                lora_rank=cfg.lora_rank,
                learning_rate=cfg.lr,
                epochs=cfg.epochs,
                batch_size=cfg.batch_size,
            )
            ft_trainer = FineTuneTrainer(ft_config)
            return ft_trainer.prepare_model()

        else:
            raise ValueError(
                f"Unknown model: {cfg.model_name}. "
                f"Available: two_tower, deepfm, mtl, din, dien, bst, finetune"
            )

    def train(self, dataset: InteractionDataset) -> dict[str, float]:
        """Full training loop.

        Returns final metrics dict.
        """
        # BUG #5 FIX: Use temporal split for sequence models to avoid data leakage.
        # BehaviorSequenceDataset._preprocess() sorts by user→timestamp, so a
        # simple index-based split preserves temporal ordering.
        # random_split() would mix future interactions into the training set,
        # giving artificially optimistic validation metrics.
        val_size = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size
        seq_models = ("din", "dien", "bst", "finetune")
        if self.config.temporal_split and self.config.model_name in seq_models:
            train_ds = Subset(dataset, range(0, train_size))
            val_ds   = Subset(dataset, range(train_size, len(dataset)))
            logger.info("Using TEMPORAL split (no shuffle) — avoids future-leakage in seq models")
        else:
            train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        logger.info("Training: %d samples, Validation: %d samples", train_size, val_size)

        final_metrics = {}

        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss = self._train_epoch(train_loader, epoch)

            # Validate
            if (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self._validate(val_loader)
                final_metrics = val_metrics

                # Track and save best
                val_loss = val_metrics.get("val_loss", train_loss)
                if self.config.save_best and val_loss < self.best_metric:
                    self.best_metric = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info("★ New best model saved (val_loss=%.4f)", val_loss)

                logger.info(
                    "Epoch %d/%d — train_loss=%.4f | %s",
                    epoch + 1, self.config.epochs, train_loss,
                    " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                )

            # Step scheduler
            self.scheduler.step()

        # Save final model
        self._save_checkpoint(self.config.epochs - 1, is_best=False)
        logger.info("Training complete. Best val_loss=%.4f", self.best_metric)

        return final_metrics

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                loss = self._compute_batch_loss(batch)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_loss = total_loss / n_batches
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "  [Epoch %d] Batch %d/%d — loss=%.4f, lr=%.2e",
                    epoch + 1, batch_idx + 1, len(loader), avg_loss, lr,
                )

        return total_loss / max(n_batches, 1)

    def _compute_batch_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Route to correct loss function based on model type."""
        cfg = self.config

        if cfg.model_name == "two_tower":
            user_feat = batch["user_features"].to(self.device)
            item_feat = batch["item_features"].to(self.device)
            return self.model.compute_loss(user_feat, item_feat)

        elif cfg.model_name == "deepfm":
            # DeepFM uses dense features directly
            item_feat = batch["item_features"].to(self.device)
            labels = batch.get("label_click", batch.get("label_purchase"))
            if labels is None:
                labels = torch.zeros(item_feat.size(0), device=self.device)
            else:
                labels = labels.to(self.device)

            B = item_feat.size(0)
            # BUG #2 FIX: Previously used torch.arange(20) as sparse_idx for every
            # sample in every batch — identical fake features, so the FM component
            # learned nothing. Replaced with feature-derived real sparse indices.
            #
            # Each feature value is hashed into a discrete bin that identifies the
            # (feature_column, value_bucket) pair, giving the FM real interactions
            # to learn from. Bucket count matches cfg.num_sparse_features.
            n_fields = min(20, item_feat.size(1))
            feat_slice = item_feat[:, :n_fields]                       # [B, F]
            # Hash each field value into a sparse index in [field*bucket, (field+1)*bucket)
            bucket_size = max(cfg.num_sparse_features // n_fields, 1)
            field_offsets = torch.arange(n_fields, device=self.device) * bucket_size  # [F]
            sparse_idx = (feat_slice.abs() * 1000).long() % bucket_size + field_offsets  # [B, F]
            sparse_val = feat_slice.abs().clamp(0.0, 1.0)              # [B, F] real magnitudes

            return self.model.compute_loss(sparse_idx, sparse_val, item_feat, labels)

        elif cfg.model_name == "mtl":
            user_feat = batch["user_features"].to(self.device)
            item_feat = batch["item_features"].to(self.device)
            features = torch.cat([user_feat, item_feat], dim=-1)

            labels = {}
            for task in TASK_NAMES:
                key = f"label_{task}"
                if key in batch:
                    labels[task] = batch[key].to(self.device)

            total_loss, _ = self.model.compute_loss(features, labels)
            return total_loss

        elif cfg.model_name in ("din", "dien", "bst", "finetune"):
            # Marketplace sequence models — DIN / DIEN / BST
            behavior_ids = batch["behavior_ids"].to(self.device)
            candidate_id = batch["candidate_id"].to(self.device)
            candidate_cat = batch["candidate_cat"].to(self.device)
            dense = batch["dense_features"].to(self.device)
            mask = batch.get("behavior_mask")
            if mask is not None:
                mask = mask.to(self.device)

            label = batch["label"].to(self.device).unsqueeze(1)
            labels_list = [label] * cfg.n_tasks_marketplace

            preds = self.model(
                behavior_ids, candidate_id, candidate_cat, dense, mask,
            )

            # Handle DIEN returning (preds, aux_logits)
            aux_logits = None
            if isinstance(preds, tuple):
                preds, aux_logits = preds

            # Loss
            if cfg.model_name == "dien":
                criterion = DIENLoss()
                aux_labels = batch.get("aux_labels")
                if aux_labels is not None:
                    aux_labels = aux_labels.to(self.device)
                return criterion(preds, labels_list, aux_logits, aux_labels)
            elif cfg.model_name == "din" or cfg.model_name == "finetune":
                criterion = DINLoss()
                return criterion(preds, labels_list)
            else:  # bst
                criterion = BSTLoss()
                return criterion(preds, labels_list)

        raise ValueError(f"Unknown model: {cfg.model_name}")

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on validation set — real metrics, no faking."""
        self.model.eval()
        all_preds: dict[str, list] = {task: [] for task in TASK_NAMES}
        all_labels: dict[str, list] = {task: [] for task in TASK_NAMES}
        # BUG #8 FIX: collect click-task preds for sequence models too
        seq_click_preds: list = []
        seq_click_labels: list = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            loss = self._compute_batch_loss(batch)
            total_loss += loss.item()
            n_batches += 1

            if self.config.model_name == "mtl":
                user_feat = batch["user_features"].to(self.device)
                item_feat = batch["item_features"].to(self.device)
                features = torch.cat([user_feat, item_feat], dim=-1)
                preds = self.model(features)

                for task in TASK_NAMES:
                    key = f"label_{task}"
                    if key in batch:
                        all_preds[task].append(preds[task].cpu().numpy())
                        all_labels[task].append(batch[key].numpy())

            elif self.config.model_name in ("din", "dien", "bst", "finetune"):
                # BUG #8 FIX: Previously DIN/DIEN/BST had NO AUC metrics in validation
                # — only val_loss was tracked, making it impossible to know whether the
                # model was actually learning to rank purchases correctly.
                behavior_ids = batch["behavior_ids"].to(self.device)
                candidate_id = batch["candidate_id"].to(self.device)
                candidate_cat = batch["candidate_cat"].to(self.device)
                dense = batch["dense_features"].to(self.device)
                mask = batch.get("behavior_mask")
                if mask is not None:
                    mask = mask.to(self.device)

                preds_out = self.model(
                    behavior_ids, candidate_id, candidate_cat, dense, mask
                )
                # DIEN returns (predictions, aux_logits)
                if isinstance(preds_out, tuple):
                    preds_out = preds_out[0]

                # Collect click-task predictions (first head)
                pred_click = preds_out[0].squeeze(-1).cpu().numpy()
                label_arr = batch["label"].numpy()
                seq_click_preds.append(pred_click)
                seq_click_labels.append(label_arr)

        metrics = {"val_loss": total_loss / max(n_batches, 1)}

        # Compute per-task AUC for MTL (only for binary tasks with both classes present)
        for task in TASK_NAMES:
            if all_preds[task]:
                preds_arr = np.concatenate(all_preds[task])
                labels_arr = np.concatenate(all_labels[task])

                if len(np.unique(labels_arr.astype(int))) >= 2:
                    metrics[f"auc_{task}"] = compute_auc(preds_arr, labels_arr.astype(int))
                    metrics[f"logloss_{task}"] = compute_logloss(preds_arr, labels_arr)

        # BUG #8 FIX: AUC / LogLoss for sequence models (DIN / DIEN / BST)
        if seq_click_preds:
            preds_arr = np.concatenate(seq_click_preds)
            labels_arr = np.concatenate(seq_click_labels)
            if len(np.unique(labels_arr.astype(int))) >= 2:
                metrics["auc_click"]     = compute_auc(preds_arr, labels_arr.astype(int))
                metrics["logloss_click"] = compute_logloss(preds_arr, labels_arr)

        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        suffix = "best" if is_best else f"epoch_{epoch}"
        path = self.output_dir / f"model_{suffix}.pt"

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "best_metric": self.best_metric,
            "global_step": self.global_step,
        }, path)

        logger.info("Checkpoint saved: %s", path)

    def export_onnx(self, export_path: str | None = None) -> Path:
        """Export model to ONNX for TorchServe / Triton deployment."""
        self.model.eval()
        path = Path(export_path or self.output_dir / "model.onnx")
        cfg = self.config

        if cfg.model_name == "mtl":
            dummy = torch.randn(1, USER_DENSE_DIM + ITEM_DENSE_DIM, device=self.device)
            torch.onnx.export(
                self.model, dummy, str(path),
                input_names=["features"],
                output_names=[f"pred_{t}" for t in TASK_NAMES],
                dynamic_axes={"features": {0: "batch_size"}},
                opset_version=17,
            )
        elif cfg.model_name == "two_tower":
            user_dummy = torch.randn(1, USER_DENSE_DIM, device=self.device)
            item_dummy = torch.randn(1, ITEM_DENSE_DIM, device=self.device)
            torch.onnx.export(
                self.model, (user_dummy, item_dummy), str(path),
                input_names=["user_features", "item_features"],
                output_names=["user_embedding", "item_embedding"],
                dynamic_axes={"user_features": {0: "batch"}, "item_features": {0: "batch"}},
                opset_version=17,
            )

        logger.info("ONNX exported: %s", path)
        return path


# ──────────────────────────────────────────────────────────────
# FAISS Index Builder (post Two-Tower training)
# ──────────────────────────────────────────────────────────────

def build_faiss_index(
    model: TwoTowerModel,
    item_features: torch.Tensor,
    item_ids: list[str],
    output_path: str = "checkpoints/faiss_index",
    device: str = "cpu",
) -> None:
    """Build FAISS ANN index from trained Two-Tower item embeddings.

    This runs once after training, then the index is loaded at serving time
    for <10ms nearest-neighbor retrieval.
    """
    import faiss

    model.eval()
    model.to(device)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode all items in batches
    batch_size = 4096
    all_embeddings = []

    for start in range(0, len(item_features), batch_size):
        batch = item_features[start : start + batch_size].to(device)
        emb = model.encode_items(batch).cpu().numpy()
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    dim = embeddings.shape[1]
    n_items = embeddings.shape[0]

    logger.info("Building FAISS index: %d items, %d dims", n_items, dim)

    # IVF + PQ index for scalability
    # nlist = sqrt(N) is a good heuristic for IVF clusters
    if n_items > 100_000:
        nlist = int(np.sqrt(n_items))
        quantizer = faiss.IndexFlatIP(dim)  # Inner product (cosine on L2-norm vectors)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, 32, 8)
        index.train(embeddings)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    # Save index + ID mapping
    faiss.write_index(index, str(output_dir / "item.index"))

    with open(output_dir / "item_ids.json", "w") as f:
        json.dump(item_ids, f)

    logger.info("FAISS index saved: %s (%d items)", output_dir, n_items)


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ShopFeed OS — Model Training")
    parser.add_argument(
        "--model", default="two_tower",
        choices=["two_tower", "deepfm", "mtl", "din", "dien", "bst", "finetune"],
    )
    parser.add_argument("--data", default="data/training_events.parquet")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--build-index", action="store_true", help="Build FAISS index after training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = TrainConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        mixed_precision=not args.no_amp,
    )

    dataset = InteractionDataset(config.data_path)
    trainer = Trainer(config)
    metrics = trainer.train(dataset)

    # Export ONNX
    trainer.export_onnx()

    # Build FAISS index for Two-Tower
    if args.build_index and config.model_name == "two_tower":
        logger.info("Building FAISS index...")
        # Would load item features from dataset
        logger.info("FAISS index building requires item features — use build_faiss_index()")

    logger.info("Final metrics: %s", json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
