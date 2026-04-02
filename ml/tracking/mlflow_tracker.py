"""
MLflow Integration — Experiment Tracking & Model Registry (archi-2026 §9.6)
=============================================================================
Production-grade tracking for all training runs. Every training invocation
logs metrics, hyperparameters, and model artifacts to MLflow.

Integration points:
    - ml/training/train.py → log_training_run()
    - ml/training/finetune.py → log_finetune_run()
    - Model promotion → register_model() + promote_model()

Requires:
    pip install mlflow>=2.12
    MLFLOW_TRACKING_URI env var (default: local ./mlruns)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# MLflow is an optional dependency — graceful fallback
try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning(
        "mlflow not installed — training metrics will only be logged to stdout. "
        "Install with: pip install mlflow>=2.12"
    )


# ── Configuration ──────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///mlruns")
MLFLOW_EXPERIMENT_PREFIX = "shopfeed"


def _ensure_mlflow() -> bool:
    """Initialize MLflow connection. Returns True if available."""
    if not HAS_MLFLOW:
        return False
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        return True
    except Exception as e:
        logger.warning("MLflow init failed: %s — falling back to stdout logging", e)
        return False


# ── Training Run Tracking ──────────────────────────────────────

def log_training_run(
    model_name: str,
    config: dict[str, Any],
    metrics: dict[str, float],
    model_path: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Log a complete training run to MLflow.

    Args:
        model_name: e.g. "two_tower", "mtl", "din", "dien", "bst"
        config: hyperparameters dict (lr, batch_size, epochs, etc.)
        metrics: final metrics dict (val_loss, auc_click, ndcg, etc.)
        model_path: path to saved model checkpoint (.pt)
        tags: optional tags (git_hash, dataset_version, etc.)

    Returns:
        MLflow run_id if logged, None otherwise.
    """
    if not _ensure_mlflow():
        logger.info("Training metrics (stdout): model=%s metrics=%s", model_name, metrics)
        return None

    experiment_name = f"{MLFLOW_EXPERIMENT_PREFIX}/{model_name}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log hyperparameters
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log tags
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("timestamp", str(int(time.time())))

        # Log model artifact
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(model_path)

        run_id = run.info.run_id
        logger.info(
            "MLflow run logged: experiment=%s run_id=%s metrics=%s",
            experiment_name, run_id, metrics,
        )
        return run_id


def log_finetune_run(
    base_model: str,
    lora_rank: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    model_path: Optional[str] = None,
) -> Optional[str]:
    """Log a LoRA fine-tuning run to MLflow.

    Separate from log_training_run to capture LoRA-specific params.
    """
    config_augmented = {
        **config,
        "base_model": base_model,
        "lora_rank": lora_rank,
        "finetune": True,
    }
    return log_training_run(
        model_name=f"{base_model}_lora_r{lora_rank}",
        config=config_augmented,
        metrics=metrics,
        model_path=model_path,
        tags={"training_type": "lora_finetune"},
    )


# ── Model Registry ─────────────────────────────────────────────

def register_model(
    model_name: str,
    run_id: str,
    artifact_path: str = "model",
) -> Optional[str]:
    """Register a trained model in MLflow Model Registry.

    Returns model version string if registered.
    """
    if not _ensure_mlflow():
        return None

    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        logger.info(
            "Model registered: name=%s version=%s",
            model_name, result.version,
        )
        return result.version
    except Exception as e:
        logger.error("Model registration failed: %s", e)
        return None


def promote_model(
    model_name: str,
    version: str,
    stage: str = "Production",
) -> bool:
    """Promote a model version to a stage (Staging/Production/Archived).

    In production, this is called after A/B test validation confirms
    the new model outperforms the current one.
    """
    if not _ensure_mlflow():
        return False

    try:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        logger.info(
            "Model promoted: %s version=%s → %s", model_name, version, stage,
        )
        return True
    except Exception as e:
        logger.error("Model promotion failed: %s", e)
        return False


def get_production_model_uri(model_name: str) -> Optional[str]:
    """Get the URI of the current production model for loading."""
    if not _ensure_mlflow():
        return None
    try:
        return f"models:/{model_name}/Production"
    except Exception:
        return None
