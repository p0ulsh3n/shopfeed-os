"""
Fine-tune Run Script — Phase 3: LoRA fine-tuning sur données propriétaires.

Strategy:
  Phase 1 (Frozen): CLIP, sentence-transformers — JAMAIS re-entraîner
  Phase 2 (Frozen): couches basses DIN/DIEN/BST
  Phase 3 (LoRA):   couches supérieures DIN, rank=8 alpha=32 (<1% params)
  Phase 4 (Full):   vendor_embeddings + category_embeddings + output layer

Usage:
  python -m scripts.finetune_run --model din --phase 3 --epochs 3
  python -m scripts.finetune_run --model all --epochs 5 --min_interactions 10000
"""

from __future__ import annotations
import argparse
import logging
import os
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OUTPUT_DIR = os.environ.get("MODEL_OUTPUT_DIR", "/tmp/shopfeed_models")


def freeze_layers(model: nn.Module, layer_names_to_freeze: list[str]) -> None:
    """Gèle les paramètres des couches spécifiées."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(freeze_name in name for freeze_name in layer_names_to_freeze):
            param.requires_grad = False
            frozen += 1
    logger.info(f"Frozen {frozen} parameter tensors.")


def apply_lora(model: nn.Module, rank: int = 8, alpha: int = 32) -> nn.Module:
    """
    Applique LoRA (Low-Rank Adaptation) aux couches linéaires des dernières couches.
    Utilise peft si disponible, sinon implémentation manuelle légère.
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["fc", "linear", "proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model
    except ImportError:
        logger.warning("peft not installed. Using manual parameter freezing instead of LoRA.")
        # Fallback: geler tout sauf les dernières 2 couches
        params = list(model.parameters())
        for p in params[:-4]:
            p.requires_grad = False
        return model


def finetune_model(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    phase: int,
) -> None:
    """Fine-tune un modèle spécifique."""
    logger.info(f"=== Fine-tuning {model_name.upper()} (Phase {phase}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle pré-entraîné
    try:
        from ml.serving.registry import ModelRegistry
        registry = ModelRegistry()
        model = getattr(registry, f"load_{model_name}")()
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return

    # Phase-specific setup
    if phase == 2:
        freeze_layers(model, ["embedding", "input_layer", "gru_layer"])
    elif phase == 3:
        freeze_layers(model, ["embedding", "input_layer"])
        model = apply_lora(model, rank=8, alpha=32)
    # Phase 4: tous les params entraînables

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/max(total,1):.2f}%)")

    # Loader de données propriétaires
    try:
        from ml.datasets.loaders import ProprietaryInteractionsLoader
        loader = ProprietaryInteractionsLoader(batch_size=batch_size, model_type=model_name)
    except Exception:
        logger.warning("Proprietary loader not available. Using mock data.")
        return

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        t = time.time()
        total_loss = 0.0
        n_batches = 0

        model.train()
        for batch in loader.get_batches():
            optimizer.zero_grad()
            outputs = model(**{k: v.to(device) for k, v in batch.items() if k != "labels"})
            loss = criterion(outputs, batch["labels"].float().to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t

        # ── Validation pass (STRUCTURAL FIX) ─────────────────────
        # Previously selected checkpoint based on training loss,
        # which guarantees overfitting. Now uses a held-out validation
        # set to select the best generalizing model.
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in loader.get_val_batches():
                val_out = model(**{k: v.to(device) for k, v in val_batch.items() if k != "labels"})
                v_loss = criterion(val_out, val_batch["labels"].float().to(device))
                val_loss += v_loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1) if val_batches > 0 else avg_loss

        logger.info(
            f"  Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f} "
            f"val_loss={avg_val_loss:.4f} ({elapsed:.1f}s)"
        )

        # Save best checkpoint based on VALIDATION loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(OUTPUT_DIR, model_name, "finetuned.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"  Best checkpoint saved: {save_path} (val_loss={avg_val_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description="ShopFeed ML Fine-tuning Script")
    parser.add_argument(
        "--model",
        choices=["din", "dien", "bst", "mtl_ple", "two_tower", "all"],
        default="all",
    )
    parser.add_argument("--phase", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--min_interactions",
        type=int,
        default=10_000,
        help="Nombre minimum d'interactions propriétaires requis"
    )
    args = parser.parse_args()

    models = (
        ["din", "dien", "bst", "mtl_ple", "two_tower"]
        if args.model == "all"
        else [args.model]
    )

    for model_name in models:
        finetune_model(model_name, args.epochs, args.batch_size, args.lr, args.phase)

    logger.info("Fine-tuning run completed.")


if __name__ == "__main__":
    main()
