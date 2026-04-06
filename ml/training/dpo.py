"""DPO — Direct Preference Optimization for ShopFeed ranking alignment.
====================================================================
Section 41-bis: Preference alignment post-training.

Why DPO over PPO/RLHF (verified 2026):
  - DPO eliminates the separate reward model and RL training loop.
  - Optimises the policy directly from preference pairs (chosen vs rejected).
  - More computationally stable and efficient than PPO (no reward hacking risk).
  - Compatible with PEFT/LoRA (already used in ml/training/finetune.py).
  - Reference: Rafailov et al. 2023, philschmid.de 2025, together.ai 2025.

Usage in ShopFeed:
  - After standard contrastive training of the Two-Tower / DIN / BST models,
    run DPO to align rankings toward preferred user outcomes (purchases > clicks).
  - Preference pairs: (user, chosen_item, rejected_item) triples constructed
    from interaction logs (purchased > viewed but not purchased > skipped).
  - The reference model is the pre-trained checkpoint (frozen during DPO).

Best-practice hyperparameters (verified 2025-2026):
  - beta: 0.1–0.3 (KL-divergence constraint; lower = more aggressive)
  - Learning rate: 1e-5 to 5e-5 (lower than contrastive training)
  - Use LoRA/PEFT to minimise memory overhead
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# DPO Config
# ──────────────────────────────────────────────────────────────

@dataclass
class DPOConfig:
    """Configuration for DPO alignment training.

    Hyperparameters verified against philschmid.de, together.ai, HuggingFace
    TRL best-practice guides (2025-2026).
    """
    # Model
    base_checkpoint: str = "checkpoints/two_tower/model_best.pt"
    output_dir: str = "checkpoints/two_tower_dpo"
    device: str = "auto"

    # DPO core
    # beta: KL-divergence constraint weight.
    # Range 0.05–0.5; 0.1 = standard starting point (together.ai 2025).
    beta: float = 0.1

    # Data
    preference_data_path: str = "data/preference_pairs.parquet"
    val_split: float = 0.1

    # Training
    epochs: int = 5              # DPO converges faster than contrastive training
    batch_size: int = 256        # Smaller batch: preference pairs are sparser
    lr: float = 2e-5             # Lower LR than standard training
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mixed_precision: bool = True

    # LoRA PEFT (reuse existing PEFT setup from finetune.py)
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16

    # Logging
    log_interval: int = 50
    save_best: bool = True


# ──────────────────────────────────────────────────────────────
# Preference Pair Dataset
# ──────────────────────────────────────────────────────────────

class PreferencePairDataset(Dataset):
    """Dataset of (user, chosen_item, rejected_item) preference triples.

    Preference pairs are constructed from interaction logs:
      - CHOSEN:   item the user purchased, added to cart, or saved
      - REJECTED: item the user viewed but skipped, or marked "not interested"

    The quality of preference pairs is critical for DPO (verified 2026):
    noisy or contradictory pairs degrade alignment significantly.

    Expected columns in the parquet:
        user_features:    float vector [user_dim]
        chosen_features:  float vector [item_dim]
        rejected_features: float vector [item_dim]
        preference_strength: float (0.0–1.0, used for sample weighting)
    """

    def __init__(self, data_path: str, max_samples: int | None = None):
        self.samples: list[dict[str, Any]] = []
        self._load(data_path, max_samples)

    def _load(self, data_path: str, max_samples: int | None) -> None:
        path = Path(data_path)
        if not path.exists():
            logger.warning(
                "Preference data not found at %s. "
                "Use build_preference_pairs() to generate it from interaction logs.",
                path,
            )
            return

        try:
            import pandas as pd
            df = pd.read_parquet(path)
            if max_samples:
                df = df.head(max_samples)
            self.samples = df.to_dict("records")
            logger.info("Loaded %d preference pairs from %s", len(self.samples), path)
        except ImportError:
            logger.error("pandas required for preference data loading")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "user_features":      torch.tensor(sample["user_features"],      dtype=torch.float32),
            "chosen_features":    torch.tensor(sample["chosen_features"],    dtype=torch.float32),
            "rejected_features":  torch.tensor(sample["rejected_features"],  dtype=torch.float32),
            "preference_strength": torch.tensor(
                float(sample.get("preference_strength", 1.0)), dtype=torch.float32
            ),
        }


def build_preference_pairs(
    interaction_logs: list[dict[str, Any]],
    action_weights: dict[str, float] | None = None,
    min_strength: float = 0.3,
) -> list[dict[str, Any]]:
    """Build preference pairs from raw interaction logs.

    Rules (aligned with action weights from monolith/streaming_trainer.py):
      - CHOSEN  = actions with weight > 4 (add_to_cart, save, purchase...)
      - REJECTED = actions with weight < 0 (skip, not_interested)
      - Pairs are filtered to have preference_strength > min_strength

    Args:
        interaction_logs: List of event dicts with user_id, item_id, action, features.
        action_weights:   Action weight dict (defaults to Monolith ACTION_WEIGHTS).
        min_strength:     Minimum preference strength to include a pair.

    Returns:
        List of preference pair dicts ready for PreferencePairDataset.
    """
    if action_weights is None:
        action_weights = {
            "buy_now": 12.0, "purchase": 10.0, "add_to_cart": 8.0,
            "save": 6.0, "share": 5.0, "question": 4.5,
            "review": 4.0, "visit_shop": 3.0, "follow": 2.5,
            "comment": 2.0, "like": 1.0,
            "skip": -4.0, "not_interested": -8.0,
        }

    # Group events by user
    user_events: dict[str, list] = {}
    for event in interaction_logs:
        uid = str(event.get("user_id", ""))
        if uid:
            user_events.setdefault(uid, []).append(event)

    pairs: list[dict[str, Any]] = []
    for uid, events in user_events.items():
        positive = [e for e in events if action_weights.get(e.get("action", ""), 0) > 4.0]
        negative = [e for e in events if action_weights.get(e.get("action", ""), 0) < 0.0]

        for pos_e in positive:
            for neg_e in negative:
                pos_weight = action_weights.get(pos_e.get("action", ""), 1.0)
                neg_weight = abs(action_weights.get(neg_e.get("action", ""), 1.0))
                strength = min(1.0, (pos_weight + neg_weight) / 20.0)

                if strength < min_strength:
                    continue

                user_feat = pos_e.get("user_features") or neg_e.get("user_features")
                chosen_feat = pos_e.get("item_features")
                rejected_feat = neg_e.get("item_features")

                if user_feat and chosen_feat and rejected_feat:
                    pairs.append({
                        "user_features":       user_feat,
                        "chosen_features":     chosen_feat,
                        "rejected_features":   rejected_feat,
                        "preference_strength": strength,
                    })

    logger.info("Built %d preference pairs from %d users", len(pairs), len(user_events))
    return pairs


# ──────────────────────────────────────────────────────────────
# DPO Loss
# ──────────────────────────────────────────────────────────────

class DPOLoss(nn.Module):
    """Direct Preference Optimization loss for embedding-based ranking.

    Adapted from the original DPO paper (Rafailov et al. 2023) for
    embedding-based recommendation models (not autoregressive LLMs).

    In the recsys context:
      - "log probability" = inner-product score between user and item embeddings
      - The policy model (being trained) generates updated embeddings
      - The reference model (frozen) generates baseline embeddings

    Loss formula:
        L_DPO = -E[ log σ( β * (r_chosen - r_rejected) ) ]
    where:
        r_x = log π_θ(x) - log π_ref(x)
             = score_policy(x) - score_reference(x)

    Args:
        beta: KL-divergence constraint (0.1 = standard, higher = more conservative)
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        chosen_policy_scores:   torch.Tensor,   # [B] — policy score for chosen items
        rejected_policy_scores: torch.Tensor,   # [B] — policy score for rejected items
        chosen_ref_scores:      torch.Tensor,   # [B] — reference score for chosen items
        rejected_ref_scores:    torch.Tensor,   # [B] — reference score for rejected items
        preference_weights:     torch.Tensor | None = None,  # [B] — optional sample weights
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss.

        Returns:
            loss: scalar DPO loss
            metrics: dict with reward_margin, accuracy for monitoring
        """
        # Implicit rewards: r_x = beta * (log π_θ(x) - log π_ref(x))
        # For embedding scores: use score directly as log-prob proxy
        chosen_rewards   = self.beta * (chosen_policy_scores   - chosen_ref_scores)
        rejected_rewards = self.beta * (rejected_policy_scores - rejected_ref_scores)

        # DPO binary cross-entropy
        reward_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(reward_diff)

        # Optional preference strength weighting
        if preference_weights is not None:
            loss = loss * preference_weights

        loss = loss.mean()

        # Monitoring metrics
        with torch.no_grad():
            accuracy  = (reward_diff > 0).float().mean().item()
            margin    = reward_diff.mean().item()

        metrics = {
            "dpo_loss": loss.item(),
            "reward_margin": margin,
            "dpo_accuracy": accuracy,  # % of pairs correctly ranked (target: > 0.7)
        }
        return loss, metrics


# ──────────────────────────────────────────────────────────────
# DPO Trainer
# ──────────────────────────────────────────────────────────────

class DPOTrainer:
    """Trains a Two-Tower model with Direct Preference Optimization.

    Workflow:
        1. Load pre-trained checkpoint as both policy and reference model
        2. Freeze the reference model (never updated)
        3. Apply LoRA to policy model for memory-efficient fine-tuning
        4. Train on preference pairs using DPO loss
        5. Save aligned model

    Usage:
        config = DPOConfig(base_checkpoint="checkpoints/two_tower/model_best.pt")
        trainer = DPOTrainer(config)
        trainer.train(preference_dataset)
    """

    def __init__(self, config: DPOConfig):
        self.config = config
        self.device = self._resolve_device()
        self.dpo_loss = DPOLoss(beta=config.beta)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        logger.info(
            "DPOTrainer ready — beta=%.3f, lr=%.2e, device=%s",
            config.beta, config.lr, self.device,
        )

    def _resolve_device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _load_models(self, user_input_dim: int, item_input_dim: int):
        """Load policy (trainable) and reference (frozen) models."""
        from ml.models.two_tower import TwoTowerModel

        # Policy model (will be fine-tuned)
        self.policy_model = TwoTowerModel(user_input_dim, item_input_dim).to(self.device)

        # Reference model (frozen — always the pre-trained baseline)
        self.ref_model = TwoTowerModel(user_input_dim, item_input_dim).to(self.device)

        ckpt_path = Path(self.config.base_checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model_state = state.get("model_state_dict", state)
            self.policy_model.load_state_dict(model_state)
            self.ref_model.load_state_dict(model_state)
            logger.info("Loaded checkpoint: %s", ckpt_path)
        else:
            logger.warning("No checkpoint found — starting from random init")

        # Freeze reference model entirely (DPO requirement)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Apply LoRA to policy model if configured
        if self.config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                lora_cfg = LoraConfig(
                    r=self.config.lora_rank,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=["proj", "gate"],   # Target projection + gate layers
                    lora_dropout=0.05,
                    bias="none",
                )
                self.policy_model = get_peft_model(self.policy_model, lora_cfg)
                self.policy_model.print_trainable_parameters()
                logger.info("LoRA applied to policy model")
            except ImportError:
                logger.warning("peft not installed — training full model without LoRA")

    def _compute_embedding_scores(
        self,
        model: nn.Module,
        user_feat: torch.Tensor,
        chosen_feat: torch.Tensor,
        rejected_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute inner-product scores as log-prob proxies."""
        user_emb, chosen_emb = model(user_feat, chosen_feat)
        _, rejected_emb       = model(user_feat, rejected_feat)

        chosen_score   = (user_emb * chosen_emb).sum(dim=-1)    # [B]
        rejected_score = (user_emb * rejected_emb).sum(dim=-1)  # [B]
        return chosen_score, rejected_score

    def train(
        self,
        dataset: PreferencePairDataset,
        user_input_dim: int = 764,
        item_input_dim: int = 1348,
    ) -> dict[str, float]:
        """Full DPO training loop.

        Returns final metrics dict.
        """
        self._load_models(user_input_dim, item_input_dim)

        # Split dataset
        val_size   = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size,
            shuffle=True, pin_memory=self.device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size * 2,
            shuffle=False, pin_memory=self.device.type == "cuda",
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.policy_model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(
            "cuda", enabled=self.config.mixed_precision and self.device.type == "cuda"
        )

        best_val_loss  = float("inf")
        final_metrics: dict[str, float] = {}

        for epoch in range(self.config.epochs):
            self.policy_model.train()
            epoch_loss   = 0.0
            epoch_acc    = 0.0
            epoch_margin = 0.0
            n_batches    = 0

            for batch_idx, batch in enumerate(train_loader):
                user_feat     = batch["user_features"].to(self.device)
                chosen_feat   = batch["chosen_features"].to(self.device)
                rejected_feat = batch["rejected_features"].to(self.device)
                pref_weights  = batch["preference_strength"].to(self.device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=self.device.type, enabled=self.config.mixed_precision):
                    # Policy model scores
                    pol_chosen, pol_rejected = self._compute_embedding_scores(
                        self.policy_model, user_feat, chosen_feat, rejected_feat
                    )
                    # Reference model scores (no grad)
                    with torch.no_grad():
                        ref_chosen, ref_rejected = self._compute_embedding_scores(
                            self.ref_model, user_feat, chosen_feat, rejected_feat
                        )

                    loss, metrics = self.dpo_loss(
                        pol_chosen, pol_rejected,
                        ref_chosen, ref_rejected,
                        pref_weights,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.config.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss   += metrics["dpo_loss"]
                epoch_acc    += metrics["dpo_accuracy"]
                epoch_margin += metrics["reward_margin"]
                n_batches    += 1

                if (batch_idx + 1) % self.config.log_interval == 0:
                    logger.info(
                        "[Epoch %d] Batch %d/%d — loss=%.4f, acc=%.2f%%, margin=%.4f",
                        epoch + 1, batch_idx + 1, len(train_loader),
                        epoch_loss / n_batches,
                        100.0 * epoch_acc / n_batches,
                        epoch_margin / n_batches,
                    )

            # Validation
            val_metrics = self._validate(val_loader)
            val_loss    = val_metrics["dpo_loss"]

            logger.info(
                "Epoch %d/%d — train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.2f%% | val_margin=%.4f",
                epoch + 1, self.config.epochs,
                epoch_loss / max(n_batches, 1),
                val_loss,
                100.0 * val_metrics.get("dpo_accuracy", 0),
                val_metrics.get("reward_margin", 0),
            )

            if self.config.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save(epoch, "best")
                logger.info("★ Best DPO model saved (val_loss=%.4f)", val_loss)

            final_metrics = val_metrics

        self._save(self.config.epochs - 1, "final")
        logger.info("DPO training complete. Best val_loss=%.4f", best_val_loss)
        return final_metrics

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.policy_model.eval()
        total = {"dpo_loss": 0.0, "dpo_accuracy": 0.0, "reward_margin": 0.0}
        n = 0

        for batch in loader:
            user_feat     = batch["user_features"].to(self.device)
            chosen_feat   = batch["chosen_features"].to(self.device)
            rejected_feat = batch["rejected_features"].to(self.device)
            pref_weights  = batch["preference_strength"].to(self.device)

            pol_chosen, pol_rejected = self._compute_embedding_scores(
                self.policy_model, user_feat, chosen_feat, rejected_feat
            )
            ref_chosen, ref_rejected = self._compute_embedding_scores(
                self.ref_model, user_feat, chosen_feat, rejected_feat
            )
            _, metrics = self.dpo_loss(
                pol_chosen, pol_rejected, ref_chosen, ref_rejected, pref_weights
            )
            for k, v in metrics.items():
                total[k] += v
            n += 1

        self.policy_model.train()
        return {k: v / max(n, 1) for k, v in total.items()}

    def _save(self, epoch: int, suffix: str) -> None:
        path = self.output_dir / f"dpo_model_{suffix}.pt"
        model_to_save = (
            self.policy_model.merge_and_unload()
            if hasattr(self.policy_model, "merge_and_unload")
            else self.policy_model
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "config": self.config.__dict__,
            "beta": self.config.beta,
        }, path)
        logger.info("DPO checkpoint saved: %s", path)
