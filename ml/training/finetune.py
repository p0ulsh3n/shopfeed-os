"""
LoRA Fine-Tuning Pipeline (Section 41)
========================================
Implements the Pre-Training → Fine-Tuning Bridge from the blueprint.

Architecture of frozen vs trainable layers:
  🔒 FROZEN: CLIP Visual Encoder (ViT-B-16, 86M params)
  🔒 FROZEN: Text Encoder (sentence-transformers multilingual)
  🔒 FROZEN: DIN/DIEN lower layers (behavioral pattern extraction)
  ✏️ TRAINABLE: Vendor embeddings (learned from scratch)
  ✏️ TRAINABLE: Upper DIN layers via LoRA (<1% params)
  ✏️ TRAINABLE: Category embeddings (new categories like African Food)
  🔄 REPLACED: Output classification head (adapted to our tasks)

Paper refs: LoRA (Hu et al. 2021), PEFT (HuggingFace)
Blueprint: Section 41 — FLIP (Netflix 2024), Transfer Learning, LoRA PEFT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─── Fine-Tuning Configuration ─────────────────────────────────────

@dataclass
class FineTuneConfig:
    """Configuration for LoRA fine-tuning pipeline."""

    # Model selection
    base_model: str = "din"  # din, dien, bst
    checkpoint_path: str | None = None  # Pre-trained checkpoint to load

    # LoRA hyperparameters (Section 41)
    lora_rank: int = 8          # Low-rank decomposition dimension
    lora_alpha: float = 16.0    # Scaling factor (alpha/rank = scaling)
    lora_dropout: float = 0.05  # Dropout on LoRA layers
    target_modules: list[str] = field(
        default_factory=lambda: ["mlp.0", "mlp.3", "mlp.6"]  # Upper MLP layers
    )

    # Frozen layers (Section 41 — what we keep from pre-training)
    freeze_embeddings: bool = False  # Item/category embeddings: fine-tune
    freeze_attention: bool = True    # Lower attention layers: freeze
    freeze_visual: bool = True       # CLIP visual encoder: always freeze
    freeze_text: bool = True         # Text encoder: always freeze

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 5
    batch_size: int = 512
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Data
    train_data_path: str = "data/finetune/train.parquet"
    val_data_path: str = "data/finetune/val.parquet"
    output_dir: str = "checkpoints/finetuned"

    # Vendor embeddings (Section 41 — always trained from scratch)
    n_vendors: int = 10000
    vendor_embed_dim: int = 64


# ─── LoRA Layer ─────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """LoRA adapter for nn.Linear layers.

    Decomposes the weight update as: ΔW = B × A
    where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, r << min(d_in, d_out)

    Only A and B are trainable (<1% of original params).
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.scaling = alpha / rank

        d_in = original_linear.in_features
        d_out = original_linear.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # LoRA decomposition matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen forward
        result = self.original(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling

    @property
    def trainable_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.original.parameters()) + self.trainable_params


# ─── Apply LoRA to Model ───────────────────────────────────────────

def apply_lora(
    model: nn.Module,
    config: FineTuneConfig,
) -> nn.Module:
    """Apply LoRA adapters to specified layers of a model.

    1. Freeze all parameters
    2. Add LoRA to target modules
    3. Unfreeze vendor embeddings (always trainable)
    4. Replace output head (always re-initialized)

    Returns the modified model (in-place).
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    total_lora_params = 0

    # Step 2: Apply LoRA to target modules
    for name, module in model.named_modules():
        if any(target in name for target in config.target_modules):
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(
                    module,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout,
                )
                # Replace the module in its parent
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model if not parent_name else dict(model.named_modules())[parent_name]
                setattr(parent, attr_name, lora_layer)
                total_lora_params += lora_layer.trainable_params
                logger.info("Applied LoRA to %s (rank=%d, params=%d)", name, config.lora_rank, lora_layer.trainable_params)

    # Step 3: Unfreeze vendor embeddings (Section 41 — always trained from scratch)
    for name, param in model.named_parameters():
        if "vendor" in name.lower():
            param.requires_grad = True
            logger.info("Unfroze vendor parameter: %s", name)

    # Step 4: Unfreeze item/category embeddings if configured
    if not config.freeze_embeddings:
        for name, param in model.named_parameters():
            if "embedding" in name.lower() and "vendor" not in name.lower():
                param.requires_grad = True
                logger.info("Unfroze embedding: %s", name)

    # Step 5: Unfreeze and re-initialize task heads (always replaced)
    for name, module in model.named_modules():
        if "task_heads" in name or "head" in name:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                for param in module.parameters():
                    param.requires_grad = True
                logger.info("Re-initialized and unfroze task head: %s", name)

    # Report
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0
    logger.info(
        "LoRA applied: %d trainable / %d total params (%.2f%%). "
        "LoRA adapter params: %d",
        trainable, total, pct, total_lora_params,
    )

    return model


# ─── Fine-Tuning Trainer ───────────────────────────────────────────

class FineTuneTrainer:
    """Orchestrates the LoRA fine-tuning process.

    Phases (Section 41):
      Phase 1: Load pre-trained model (from batch training or public datasets)
      Phase 2: Apply LoRA adapters (freeze base, add low-rank updates)
      Phase 3: Fine-tune on proprietary data (<1% params trainable)
      Phase 4: Export fine-tuned model for serving
    """

    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_base_model(self) -> nn.Module:
        """Load the pre-trained base model."""
        from .din import DINModel
        from .dien import DIENModel
        from .bst import BSTModel

        model_map = {
            "din": lambda: DINModel(n_items=1_000_000, n_categories=500),
            "dien": lambda: DIENModel(n_items=1_000_000, n_categories=500),
            "bst": lambda: BSTModel(n_items=1_000_000, n_categories=500),
        }

        if self.config.base_model not in model_map:
            raise ValueError(f"Unknown model: {self.config.base_model}")

        model = model_map[self.config.base_model]()

        if self.config.checkpoint_path:
            state = torch.load(self.config.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            logger.info("Loaded checkpoint: %s", self.config.checkpoint_path)

        return model

    def prepare_model(self) -> nn.Module:
        """Phase 1+2: Load and apply LoRA."""
        model = self.load_base_model()
        model = apply_lora(model, self.config)
        return model.to(self.device)

    def train(self, model: nn.Module, train_loader: Any, val_loader: Any) -> nn.Module:
        """Phase 3: Fine-tune with LoRA."""
        from .din import DINLoss

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs,
        )

        criterion = DINLoss()
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")

        best_val_loss = float("inf")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
                    preds = model(
                        behavior_ids=batch["behavior_ids"],
                        candidate_id=batch["candidate_id"],
                        candidate_cat=batch["candidate_cat"],
                        dense_features=batch["dense_features"],
                        behavior_mask=batch.get("behavior_mask"),
                    )
                    # Handle DIEN returning (preds, aux_logits)
                    if isinstance(preds, tuple):
                        preds = preds[0]

                    labels = [batch["label"].unsqueeze(1)] * len(preds)
                    loss = criterion(preds, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("Epoch %d/%d — Loss: %.4f — LR: %.2e",
                        epoch + 1, self.config.epochs, avg_loss, scheduler.get_last_lr()[0])

            # Save best
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                self._save_lora_weights(model, output_dir / "best_lora.pt")

        return model

    def _save_lora_weights(self, model: nn.Module, path: Path) -> None:
        """Save only the LoRA adapter weights (tiny file)."""
        lora_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_state[name] = param.data.cpu()

        torch.save(lora_state, path)
        size_mb = path.stat().st_size / 1024 / 1024
        logger.info("Saved LoRA weights: %s (%.1f MB)", path, size_mb)

    def export_merged(self, model: nn.Module, path: Path) -> None:
        """Phase 4: Merge LoRA weights into base model and export for serving."""
        # Merge LoRA weights into original Linear layers
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                # Merge: W_merged = W_original + B @ A * scaling
                with torch.no_grad():
                    delta = module.lora_B @ module.lora_A * module.scaling
                    module.original.weight.add_(delta)

        # Save full merged model
        torch.save(model.state_dict(), path)
        logger.info("Exported merged model: %s", path)
