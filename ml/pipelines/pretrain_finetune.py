"""
ShopFeed — Pre-Production Fine-Tuning Pipeline
===============================================
Script unique qui orchestre TOUT le fine-tuning sur ton catalogue produits
avant de partir en production. Lance-le une seule fois avec tes données.

Ce que ce script fait dans l'ordre :
  ÉTAPE 1 — LoRA adapters visuels  (marqo-ecommerce-L par catégorie)
  ÉTAPE 2 — Projection Head        (1024d → 512d, apprise sur tes produits)
  ÉTAPE 3 — Fine-tuning DIN/BST    (behavioral models sur tes interactions)
  ÉTAPE 4 — DPO alignment          (si tu as des données de préférences)
  ÉTAPE 5 — FAISS re-indexation    (tous les produits avec les nouveaux embeddings)
  ÉTAPE 6 — Validation finale      (AUC, recall@10, latence)

Usage :
    python -m ml.pipelines.pretrain_finetune \\
        --products-data data/products.parquet \\
        --interactions-data data/interactions.parquet \\
        --categories fashion electronics food \\
        --skip-steps dpo          # si pas encore de données préférences

Format de products.parquet :
    image_path  (str)  : chemin vers l'image produit ou URL
    title       (str)  : titre du produit
    category    (str)  : catégorie (fashion, electronics, food, auto...)
    product_id  (str)  : identifiant unique

Format de interactions.parquet :
    user_id     (str)
    product_id  (str)
    action      (str)  : click, add_to_cart, purchase, save, skip
    timestamp   (int)  : unix timestamp
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _banner(step: int, title: str, total: int = 6) -> None:
    logger.info("")
    logger.info("━" * 60)
    logger.info("  ÉTAPE %d/%d — %s", step, total, title)
    logger.info("━" * 60)


def _load_products(path: str) -> "pd.DataFrame":
    import pandas as pd
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    elif p.suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"Format non supporté: {p.suffix}. Utiliser .parquet, .csv ou .json")


def _load_interactions(path: str) -> "pd.DataFrame":
    import pandas as pd
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


# ──────────────────────────────────────────────────────────────
# ÉTAPE 1 — LoRA Adapters Visuels
# ──────────────────────────────────────────────────────────────

def step_visual_lora(
    products_df,
    categories: list[str],
    epochs: int = 10,
    batch_size: int = 32,
    min_products_per_cat: int = 100,
) -> dict[str, Path]:
    """Entraîne un adapter LoRA visuel pour chaque catégorie.

    Nécessite au minimum min_products_per_cat images par catégorie.
    Les catégories avec moins d'images → modèle de base utilisé (pas d'adapter).
    """
    _banner(1, "LoRA ADAPTERS VISUELS (marqo-ecommerce-L par catégorie)")

    from ml.feature_store.multi_domain_encoder import CategoryAdapterTrainer

    results = {}
    for cat in categories:
        cat_df = products_df[products_df["category"].str.lower() == cat.lower()]
        n = len(cat_df)

        if n < min_products_per_cat:
            logger.warning(
                "  ⚠ Catégorie '%s' : seulement %d produits (min=%d) "
                "→ adapter ignoré, modèle de base utilisé",
                cat, n, min_products_per_cat,
            )
            continue

        logger.info("  → Entraînement adapter '%s' sur %d produits...", cat, n)
        t0 = time.time()

        trainer = CategoryAdapterTrainer(category=cat)
        adapter_path = trainer.train(
            image_paths=cat_df["image_path"].tolist(),
            titles=cat_df["title"].tolist(),
            epochs=epochs,
            batch_size=batch_size,
        )

        elapsed = time.time() - t0
        logger.info("  ✓ Adapter '%s' sauvegardé en %.1fs : %s", cat, elapsed, adapter_path)
        results[cat] = adapter_path

    logger.info("  Adapters entraînés : %d/%d catégories", len(results), len(categories))
    return results


# ──────────────────────────────────────────────────────────────
# ÉTAPE 2 — Projection Head (1024d → 512d)
# ──────────────────────────────────────────────────────────────

def step_projection_head(
    products_df,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
) -> Path:
    """Entraîne la couche de projection 1024d → 512d sur tes produits.

    Apprentissage contrastif : (image, titre) = paire positive.
    Indispensable pour que les embeddings 512d soient bien calibrés
    sur tes données et pas juste une troncature arbitraire de 1024d.
    """
    _banner(2, "PROJECTION HEAD (1024d → 512d apprise sur tes produits)")

    from torch.utils.data import DataLoader, Dataset
    from ml.feature_store.multi_domain_encoder import (
        EcommerceEncoder,
        ProjectionHead,
        UNIFIED_DIM,
        ADAPTER_DIR,
    )
    from ml.feature_store.encoders import get_text_encoder
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Device: %s", device)

    # Charger l'encodeur de base (sans adapter — projection générale)
    enc = EcommerceEncoder.get_instance()
    base = enc._ensure_base_model()
    if base is None:
        logger.error("  ✗ Modèle de base indisponible — étape sautée")
        return None

    text_enc = get_text_encoder()
    if text_enc is None:
        logger.error("  ✗ Text encoder indisponible — étape sautée")
        return None

    # Dataset
    class PairDataset(Dataset):
        def __init__(self, df, preprocess):
            self.df = df.reset_index(drop=True)
            self.preprocess = preprocess
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            try:
                from PIL import Image as PILImage
                img = PILImage.open(row["image_path"]).convert("RGB")
                return self.preprocess(img), row["title"]
            except Exception:
                # Image invalide → zéros
                return torch.zeros(3, 224, 224), row.get("title", "")

    logger.info("  %d produits dans le dataset de projection", len(products_df))
    dataset = PairDataset(products_df, base.preprocess)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Projection + optimizer
    projection = ProjectionHead(input_dim=base.output_dim, output_dim=UNIFIED_DIM).to(device)
    optimizer  = torch.optim.AdamW(projection.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    base.model.eval()
    projection.train()

    best_loss = float("inf")
    ckpt_path = ADAPTER_DIR / "projection.pt"
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches  = 0

        for imgs, titles in loader:
            # Image embeddings
            with torch.no_grad():
                img_embs = base.model.encode_image(imgs.to(device))
                img_embs = F.normalize(img_embs.float(), dim=-1)

            # Projeter
            proj_embs = projection(img_embs)          # [B, 512]

            # Text embeddings
            with torch.no_grad():
                txt_embs = torch.tensor(
                    text_enc.encode(list(titles), show_progress_bar=False),
                    device=device, dtype=torch.float32,
                )
                # Adapter dimension texte → 512 si nécessaire
                if txt_embs.shape[-1] != UNIFIED_DIM:
                    txt_embs = F.adaptive_avg_pool1d(
                        txt_embs.unsqueeze(1), UNIFIED_DIM
                    ).squeeze(1)
                txt_embs = F.normalize(txt_embs, dim=-1)

            # InfoNCE contrastif (image ↔ titre)
            temperature = 0.07
            logits = torch.matmul(proj_embs, txt_embs.T) / temperature
            labels = torch.arange(len(imgs), device=device)
            loss   = (
                F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)
            ) / 2.0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        logger.info("  Epoch %d/%d — loss=%.4f", epoch + 1, epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(projection.state_dict(), ckpt_path)
            logger.info("    ★ Meilleure projection sauvegardée (loss=%.4f)", best_loss)

    # Injecter la projection dans le singleton EcommerceEncoder
    enc._projection = projection.eval()
    logger.info("  ✓ Projection head entraînée et chargée dans EcommerceEncoder")
    return ckpt_path


# ──────────────────────────────────────────────────────────────
# ÉTAPE 3 — Fine-tuning DIN/BST sur interactions
# ──────────────────────────────────────────────────────────────

def step_finetune_behavioral(
    interactions_df,
    model_names: list[str] = ("din", "bst"),
    checkpoint_dir: str = "checkpoints",
) -> None:
    """Fine-tune DIN et/ou BST sur les interactions réelles ShopFeed."""
    _banner(3, "FINE-TUNING COMPORTEMENTAL (DIN / BST sur tes interactions)")

    from ml.training.finetune import FineTuneConfig, FineTuneTrainer
    from ml.training.train import create_data_loaders

    for model_name in model_names:
        ckpt_candidates = sorted(Path(checkpoint_dir).glob(f"{model_name}_*.pt"))
        ckpt_path = str(ckpt_candidates[-1]) if ckpt_candidates else None

        if ckpt_path:
            logger.info("  → Fine-tune %s depuis checkpoint %s", model_name, ckpt_path)
        else:
            logger.warning(
                "  ⚠ Aucun checkpoint pré-entraîné trouvé pour %s. "
                "Lancer d'abord: python -m ml.training.train --model %s",
                model_name, model_name,
            )
            continue

        config = FineTuneConfig(
            base_model=model_name,
            checkpoint_path=ckpt_path,
            epochs=5,
            batch_size=512,
            learning_rate=1e-4,
            output_dir=f"checkpoints/finetuned_{model_name}",
        )

        try:
            train_loader, val_loader = create_data_loaders(interactions_df, config.batch_size)
            trainer = FineTuneTrainer(config)
            model   = trainer.prepare_model()
            trainer.train(model, train_loader, val_loader)
            logger.info("  ✓ %s fine-tuné → %s/best_lora.pt", model_name, config.output_dir)
        except Exception as e:
            logger.error("  ✗ Fine-tuning %s échoué: %s", model_name, e)


# ──────────────────────────────────────────────────────────────
# ÉTAPE 4 — DPO Alignment (optionnel)
# ──────────────────────────────────────────────────────────────

def step_dpo_alignment(interactions_df, epochs: int = 3) -> None:
    """Aligne le modèle sur les préférences réelles (achats > clics).

    Nécessite au minimum 1000 paires (produit préféré, produit rejeté).
    Skip automatiquement si pas assez de données.
    """
    _banner(4, "DPO ALIGNMENT (achats >> clics)")

    # Construire des paires préférences depuis les interactions
    purchases = interactions_df[interactions_df["action"] == "purchase"]
    skips     = interactions_df[interactions_df["action"] == "skip"]

    if len(purchases) < 500 or len(skips) < 500:
        logger.warning(
            "  ⚠ Pas assez de données pour DPO (purchases=%d, skips=%d, min=500). "
            "Étape sautée — relancer après accumulation d'interactions réelles.",
            len(purchases), len(skips),
        )
        return

    logger.info("  %d achats + %d skips → construction des paires...", len(purchases), len(skips))

    try:
        from ml.training.dpo import DPOConfig, DPOTrainer, PreferencePairDataset

        config  = DPOConfig(epochs=epochs, beta=0.1, learning_rate=1e-5)
        dataset = PreferencePairDataset.from_interactions(interactions_df)
        trainer = DPOTrainer(config=config)
        trainer.train(dataset)
        logger.info("  ✓ DPO alignment terminé")
    except Exception as e:
        logger.error("  ✗ DPO échoué: %s", e)


# ──────────────────────────────────────────────────────────────
# ÉTAPE 5 — FAISS Re-indexation
# ──────────────────────────────────────────────────────────────

def step_faiss_reindex(
    products_df,
    faiss_index_path: str = "checkpoints/faiss/product_embeddings.index",
    batch_size: int = 256,
) -> None:
    """Re-encode tous les produits avec les nouveaux embeddings et recrée l'index FAISS."""
    _banner(5, "FAISS RE-INDEXATION (tous les produits avec les nouveaux embeddings)")

    from ml.feature_store.encoders import encode_product_batch

    n = len(products_df)
    logger.info("  %d produits à ré-indexer...", n)

    image_paths = products_df["image_path"].tolist()
    categories  = products_df["category"].tolist()
    product_ids = products_df["product_id"].tolist()

    t0 = time.time()
    embeddings = encode_product_batch(image_paths, categories, batch_size=batch_size)
    elapsed    = time.time() - t0
    logger.info(
        "  %d embeddings calculés en %.1fs (%.0f produits/s)",
        n, elapsed, n / max(elapsed, 1),
    )

    # Sauvegarder embeddings + IDs
    try:
        import faiss
        import numpy as np

        emb_np = embeddings.numpy()
        dim    = emb_np.shape[1]

        # Index HNSW (recommandé prod 2026 : rappel >95% à <2ms)
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 200
        index.add(emb_np)

        out_path = Path(faiss_index_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(out_path))

        # Sauvegarder la correspondance idx → product_id
        ids_path = out_path.with_suffix(".ids.txt")
        ids_path.write_text("\n".join(product_ids))

        logger.info(
            "  ✓ FAISS index sauvegardé: %s (%d vecteurs, dim=%d)",
            out_path, len(product_ids), dim,
        )
    except ImportError:
        logger.warning("  ⚠ faiss-cpu non installé — index non sauvegardé. pip install faiss-cpu")
    except Exception as e:
        logger.error("  ✗ FAISS indexation échouée: %s", e)


# ──────────────────────────────────────────────────────────────
# ÉTAPE 6 — Validation finale
# ──────────────────────────────────────────────────────────────

def step_validate(faiss_index_path: str, products_df) -> dict:
    """Vérifie la qualité des embeddings avec des métriques offline."""
    _banner(6, "VALIDATION FINALE")

    metrics = {}

    # Test recall@10 sur un sous-ensemble
    try:
        import faiss
        import numpy as np
        from ml.feature_store.encoders import encode_query_image

        index    = faiss.read_index(faiss_index_path)
        n_test   = min(100, len(products_df))
        test_df  = products_df.sample(n=n_test, random_state=42)

        hits10 = 0
        for _, row in test_df.iterrows():
            query_emb = encode_query_image(row["image_path"]).numpy()
            _, I = index.search(query_emb.reshape(1, -1), k=10)
            # Le produit lui-même doit être dans le top-10
            if row.name in I[0]:
                hits10 += 1

        recall10 = hits10 / n_test
        metrics["recall@10"] = recall10
        logger.info(
            "  recall@10 (self-retrieval) : %.3f %s",
            recall10,
            "✓" if recall10 > 0.85 else "⚠ (attendu >0.85)",
        )
    except Exception as e:
        logger.warning("  ⚠ Validation recall@10 ignorée: %s", e)

    # Test latence encode_query
    try:
        sample_img = products_df.iloc[0]["image_path"]
        from ml.feature_store.encoders import encode_query_image
        # Warmup
        encode_query_image(sample_img)
        # Mesure
        t0 = time.time()
        for _ in range(10):
            encode_query_image(sample_img)
        latency_ms = (time.time() - t0) / 10 * 1000
        metrics["latency_ms_p50"] = latency_ms
        logger.info(
            "  Latence encode_query (p50) : %.1fms %s",
            latency_ms,
            "✓" if latency_ms < 15 else "⚠ (attendu <15ms GPU)",
        )
    except Exception as e:
        logger.warning("  ⚠ Test latence ignoré: %s", e)

    logger.info("  Métriques finales : %s", metrics)
    return metrics


# ──────────────────────────────────────────────────────────────
# ORCHESTRATEUR PRINCIPAL
# ──────────────────────────────────────────────────────────────

def run_pretrain_finetune(
    products_data: str,
    interactions_data: Optional[str] = None,
    categories: list[str] | None = None,
    skip_steps: list[str] | None = None,
    visual_epochs: int = 10,
    projection_epochs: int = 5,
    behavioral_epochs: int = 5,
    dpo_epochs: int = 3,
    faiss_index: str = "checkpoints/faiss/product_embeddings.index",
    min_products_per_cat: int = 100,
) -> None:
    """Pipeline complet de pre-production fine-tuning.

    Args:
        products_data:        Chemin vers le parquet produits
        interactions_data:    Chemin vers le parquet interactions (optionnel)
        categories:           Catégories à traiter (défaut: toutes dans les données)
        skip_steps:           Étapes à sauter (ex: ["dpo", "faiss"])
        visual_epochs:        Epochs pour les adapters LoRA visuels
        projection_epochs:    Epochs pour la projection head
        behavioral_epochs:    Epochs pour DIN/BST
        dpo_epochs:           Epochs pour DPO
        faiss_index:          Chemin de sortie de l'index FAISS
        min_products_per_cat: Minimum de produits pour créer un adapter
    """
    skip_steps = [s.lower() for s in (skip_steps or [])]
    t_total = time.time()

    logger.info("=" * 60)
    logger.info("  SHOPFEED — PRE-PRODUCTION FINE-TUNING PIPELINE")
    logger.info("=" * 60)
    logger.info("  Products data    : %s", products_data)
    logger.info("  Interactions data: %s", interactions_data or "non fourni")
    logger.info("  Skip steps       : %s", skip_steps or "aucune")

    # Charger les données
    logger.info("\n  Chargement des données...")
    products_df = _load_products(products_data)
    logger.info("  %d produits chargés", len(products_df))

    if categories is None:
        categories = products_df["category"].str.lower().unique().tolist()
        logger.info("  Catégories détectées : %s", categories)

    interactions_df = None
    if interactions_data:
        interactions_df = _load_interactions(interactions_data)
        logger.info("  %d interactions chargées", len(interactions_df))

    # ─── ÉTAPE 1
    if "visual" not in skip_steps and "lora" not in skip_steps:
        step_visual_lora(products_df, categories, epochs=visual_epochs, min_products_per_cat=min_products_per_cat)
    else:
        logger.info("  [SKIP] Étape 1 — LoRA adapters visuels")

    # ─── ÉTAPE 2
    if "projection" not in skip_steps:
        step_projection_head(products_df, epochs=projection_epochs)
    else:
        logger.info("  [SKIP] Étape 2 — Projection head")

    # ─── ÉTAPE 3
    if interactions_df is not None and "behavioral" not in skip_steps and "din" not in skip_steps:
        step_finetune_behavioral(interactions_df)
    else:
        logger.info("  [SKIP] Étape 3 — Fine-tuning comportemental (pas d'interactions ou skip)")

    # ─── ÉTAPE 4
    if interactions_df is not None and "dpo" not in skip_steps:
        step_dpo_alignment(interactions_df, epochs=dpo_epochs)
    else:
        logger.info("  [SKIP] Étape 4 — DPO alignment")

    # ─── ÉTAPE 5
    if "faiss" not in skip_steps:
        step_faiss_reindex(products_df, faiss_index)
    else:
        logger.info("  [SKIP] Étape 5 — FAISS re-indexation")

    # ─── ÉTAPE 6
    if "validate" not in skip_steps and not ("faiss" in skip_steps):
        step_validate(faiss_index, products_df)

    total_min = (time.time() - t_total) / 60
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ✅ FINE-TUNING PRÉ-PRODUCTION TERMINÉ en %.1f minutes", total_min)
    logger.info("  Prêt pour la production 🚀")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="ShopFeed — Pipeline de fine-tuning pré-production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  # Fine-tuning complet (avec interactions réelles)
  python -m ml.pipelines.pretrain_finetune \\
      --products-data data/products.parquet \\
      --interactions-data data/interactions.parquet

  # Juste les encodeurs visuels (sans comportemental)
  python -m ml.pipelines.pretrain_finetune \\
      --products-data data/products.parquet \\
      --skip-steps behavioral dpo

  # Seulement 3 catégories, sauter DPO et FAISS
  python -m ml.pipelines.pretrain_finetune \\
      --products-data data/products.parquet \\
      --categories fashion electronics food \\
      --skip-steps dpo faiss
        """,
    )

    parser.add_argument("--products-data",      required=True,  help="Chemin vers products.parquet")
    parser.add_argument("--interactions-data",  default=None,   help="Chemin vers interactions.parquet (optionnel)")
    parser.add_argument("--categories",         nargs="+",      help="Catégories à traiter (défaut: toutes)")
    parser.add_argument("--skip-steps",         nargs="+",      default=[], help="Étapes à sauter: visual lora projection behavioral dpo faiss validate")
    parser.add_argument("--visual-epochs",      type=int, default=10)
    parser.add_argument("--projection-epochs",  type=int, default=5)
    parser.add_argument("--behavioral-epochs",  type=int, default=5)
    parser.add_argument("--dpo-epochs",         type=int, default=3)
    parser.add_argument("--min-products",       type=int, default=100, help="Minimum produits pour créer un adapter LoRA")
    parser.add_argument("--faiss-index",        default="checkpoints/faiss/product_embeddings.index")

    args = parser.parse_args()

    run_pretrain_finetune(
        products_data=args.products_data,
        interactions_data=args.interactions_data,
        categories=args.categories,
        skip_steps=args.skip_steps,
        visual_epochs=args.visual_epochs,
        projection_epochs=args.projection_epochs,
        behavioral_epochs=args.behavioral_epochs,
        dpo_epochs=args.dpo_epochs,
        faiss_index=args.faiss_index,
        min_products_per_cat=args.min_products,
    )
