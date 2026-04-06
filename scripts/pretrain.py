"""
Pre-training Script — Phase 1: entraînement initial sur datasets publics.

Pipeline:
  Phase 1a: Two-Tower sur Alibaba UserBehavior
  Phase 1b: CLIP fashion sur DeepFashion2 + iMaterialist
  Phase 1c: DIN/DIEN/BST sur Alibaba UserBehavior (séquences)

Usage:
  python -m scripts.pretrain --phase 1a --epochs 5 --batch_size 1024
  python -m scripts.pretrain --phase all --output_dir s3://shopfeed-ml-models/
"""

from __future__ import annotations
import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def pretrain_two_tower(epochs: int, batch_size: int, output_dir: str) -> None:
    """Phase 1a: Two-Tower sur Alibaba UserBehavior."""
    logger.info("=== Phase 1a: Two-Tower Pre-training ===")
    from ml.training.two_tower import TwoTowerModel
    from ml.datasets.loaders import AlibabaUserBehaviorLoader

    model = TwoTowerModel(user_input_dim=764, item_input_dim=1348)
    loader = AlibabaUserBehaviorLoader(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        t = time.time()
        total_loss = 0.0
        n_batches = 0

        for batch in loader.get_batches():
            user_feats = batch["user_features"].to(device)
            item_feats = batch["item_features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            user_emb, item_emb = model(user_feats, item_feats)
            # In-batch negatives loss
            scores = torch.matmul(user_emb, item_emb.T)
            loss = nn.CrossEntropyLoss()(
                scores,
                torch.arange(scores.size(0), device=device)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t
        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} ({elapsed:.1f}s)")

    # Sauvegarde
    save_path = os.path.join(output_dir, "two_tower", "pretrained.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Two-Tower pre-trained model saved to {save_path}")


def pretrain_behavior_models(epochs: int, batch_size: int, output_dir: str) -> None:
    """Phase 1c: DIN/DIEN/BST sur séquences Alibaba UserBehavior."""
    logger.info("=== Phase 1c: DIN/DIEN/BST Pre-training ===")
    from ml.training.din import DINModel
    from ml.training.dien import DIENModel
    from ml.training.bst import BSTModel
    from ml.datasets.loaders import AlibabaSequenceLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = AlibabaSequenceLoader(batch_size=batch_size)

    for ModelClass, name in [(DINModel, "din"), (DIENModel, "dien"), (BSTModel, "bst")]:
        logger.info(f"Pre-training {name.upper()}...")
        model = ModelClass(n_items=1_000_000, n_categories=500).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in loader.get_batches():
                optimizer.zero_grad()
                outputs = model(**{k: v.to(device) for k, v in batch.items() if k != "labels"})
                loss = criterion(outputs, batch["labels"].float().to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            logger.info(f"  {name} Epoch {epoch+1}/{epochs}: loss={total_loss/max(n_batches,1):.4f}")

        save_path = os.path.join(output_dir, name, "pretrained.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info(f"  {name} saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="ShopFeed ML Pre-training Script")
    parser.add_argument(
        "--phase",
        choices=["1a", "1c", "all"],
        default="all",
        help="Phase de pre-training: 1a=Two-Tower, 1c=DIN/DIEN/BST, all=tout"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("MODEL_OUTPUT_DIR", "/tmp/shopfeed_models"),
        help="Répertoire de sortie des modèles (local ou s3://)"
    )
    args = parser.parse_args()

    logger.info(f"Starting pre-training: phase={args.phase}, epochs={args.epochs}, batch={args.batch_size}")

    if args.phase in ("1a", "all"):
        pretrain_two_tower(args.epochs, args.batch_size, args.output_dir)

    if args.phase in ("1c", "all"):
        pretrain_behavior_models(args.epochs, args.batch_size, args.output_dir)

    logger.info("Pre-training completed.")


if __name__ == "__main__":
    main()
