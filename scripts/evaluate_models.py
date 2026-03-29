"""
Evaluate Models — Évaluation offline des modèles ML.
Métriques: AUC-ROC, Recall@10, NDCG@10, Precision@5.

Usage:
  python -m scripts.evaluate_models --model two_tower --split test
  python -m scripts.evaluate_models --model all --output_csv eval_results.csv
"""

from __future__ import annotations
import argparse
import csv
import logging
import os
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AUC-ROC via sklearn."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_scores))
    except Exception as e:
        logger.error(f"AUC-ROC failed: {e}")
        return 0.0


def compute_recall_at_k(ranked_lists: list[list[str]], relevant_items: list[set], k: int = 10) -> float:
    """Recall@K moyen sur toutes les requêtes."""
    recalls = []
    for ranked, relevant in zip(ranked_lists, relevant_items):
        if not relevant:
            continue
        hits = sum(1 for item in ranked[:k] if item in relevant)
        recalls.append(hits / min(len(relevant), k))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_ndcg_at_k(ranked_lists: list[list[str]], relevant_items: list[set], k: int = 10) -> float:
    """NDCG@K moyen."""
    def dcg(items, relevant, k):
        return sum(
            (1.0 / np.log2(i + 2)) if items[i] in relevant else 0.0
            for i in range(min(k, len(items)))
        )

    def idcg(relevant, k):
        return sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))

    ndcgs = []
    for ranked, relevant in zip(ranked_lists, relevant_items):
        if not relevant:
            continue
        dcg_val = dcg(ranked, relevant, k)
        idcg_val = idcg(relevant, k)
        ndcgs.append(dcg_val / idcg_val if idcg_val > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def compute_precision_at_k(ranked_lists: list[list[str]], relevant_items: list[set], k: int = 5) -> float:
    """Precision@K moyen."""
    precisions = []
    for ranked, relevant in zip(ranked_lists, relevant_items):
        hits = sum(1 for item in ranked[:k] if item in relevant)
        precisions.append(hits / k)
    return float(np.mean(precisions)) if precisions else 0.0


def evaluate_model(
    model_name: str,
    split: str = "test",
) -> dict[str, float]:
    """Évalue un modèle sur le split de test et retourne les métriques."""
    logger.info(f"Evaluating {model_name} on {split} split...")
    t = time.time()

    metrics = {
        "model": model_name,
        "split": split,
        "auc_roc": 0.0,
        "recall_at_10": 0.0,
        "ndcg_at_10": 0.0,
        "precision_at_5": 0.0,
        "latency_ms": 0.0,
    }

    try:
        from ml.serving.registry import ModelRegistry
        registry = ModelRegistry()

        # Charger données de test depuis ml_training_samples
        # Pour l'instant, on crée un mock qui représente le flow complet
        logger.info(f"  Loading {split} samples for {model_name}...")

        # Mock évaluation — à remplacer par vraie logique DB
        y_true = np.random.randint(0, 2, size=1000)
        y_scores = np.random.random(size=1000)

        metrics["auc_roc"] = compute_auc_roc(y_true, y_scores)

        # Retrieval metrics (Two-Tower ou ranking models)
        n_queries = 100
        ranked_lists = [
            [str(i) for i in np.random.choice(1000, 20, replace=False)]
            for _ in range(n_queries)
        ]
        relevant_items = [
            {str(i) for i in np.random.choice(1000, 5, replace=False)}
            for _ in range(n_queries)
        ]

        metrics["recall_at_10"] = compute_recall_at_k(ranked_lists, relevant_items, k=10)
        metrics["ndcg_at_10"] = compute_ndcg_at_k(ranked_lists, relevant_items, k=10)
        metrics["precision_at_5"] = compute_precision_at_k(ranked_lists, relevant_items, k=5)
        metrics["latency_ms"] = (time.time() - t) * 1000

    except Exception as e:
        logger.error(f"Evaluation failed for {model_name}: {e}")

    logger.info(
        f"  {model_name}: AUC={metrics['auc_roc']:.4f} "
        f"R@10={metrics['recall_at_10']:.4f} "
        f"NDCG@10={metrics['ndcg_at_10']:.4f} "
        f"P@5={metrics['precision_at_5']:.4f} "
        f"({metrics['latency_ms']:.1f}ms)"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Offline model evaluation")
    parser.add_argument("--model", default="all",
                        choices=["two_tower", "mtl_ple", "din", "dien", "bst", "deepfm", "all"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output_csv", default="eval_results.csv")
    args = parser.parse_args()

    models = (
        ["two_tower", "mtl_ple", "din", "dien", "bst", "deepfm"]
        if args.model == "all"
        else [args.model]
    )

    all_metrics = [evaluate_model(m, args.split) for m in models]

    # Export CSV
    if all_metrics:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        logger.info(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
