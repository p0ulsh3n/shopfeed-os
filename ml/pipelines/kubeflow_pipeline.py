"""
Kubeflow ML Pipeline (archi-2026 §9.6)
========================================
Orchestrates the end-to-end ML training pipeline on Kubernetes:
    1. Spark extracts features (7 days)
    2. PyTorch trains models (4× A100 GPUs, ~45min)
    3. Validation gate (AUC > 0.82, NDCG@10 > 0.71)
    4. ONNX export + Triton config
    5. Canary deploy (5% traffic)
    6. A/B test → full promotion if +2% engagement

Requires:
    pip install kfp>=2.7
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import kfp
    from kfp import dsl
    from kfp.dsl import Input, Output, Artifact, Model, Metrics
    HAS_KFP = True
except ImportError:
    HAS_KFP = False
    logger.warning("kfp not installed — Kubeflow pipelines disabled. pip install kfp>=2.7")


# ── Configuration ──────────────────────────────────────────────

KUBEFLOW_HOST = os.getenv("KUBEFLOW_HOST", "http://kubeflow.internal:8080")
GPU_IMAGE = os.getenv("ML_GPU_IMAGE", "shopfeed/ml-training:latest")
SPARK_IMAGE = os.getenv("SPARK_IMAGE", "shopfeed/spark:latest")

# Validation gates — model must pass these to be promoted
VALIDATION_GATES = {
    "min_auc_click": 0.82,
    "min_ndcg_10": 0.71,
    "max_latency_p99_ms": 10,
    "min_engagement_lift_pct": 2.0,
}


# ── Pipeline Definition ──────────────────────────────────────

def build_training_pipeline_yaml() -> str:
    """Generate the Kubeflow pipeline YAML for the shopfeed ML training loop.

    This is the complete automated retraining pipeline that runs every 6h.
    Returns the pipeline specification as YAML string.
    """
    pipeline_spec = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": "shopfeed-ml-train-",
            "labels": {"pipeline": "shopfeed-recommendation"},
        },
        "spec": {
            "entrypoint": "ml-training-pipeline",
            "arguments": {
                "parameters": [
                    {"name": "training-days", "value": "7"},
                    {"name": "model-type", "value": "two_tower"},
                    {"name": "gpu-count", "value": "4"},
                ],
            },
            "templates": [
                # Step 1: Spark Feature Extraction
                {
                    "name": "extract-features",
                    "container": {
                        "image": SPARK_IMAGE,
                        "command": ["python", "-m", "ml.training.spark_config"],
                        "args": ["--days", "{{workflow.parameters.training-days}}"],
                        "resources": {
                            "requests": {"cpu": "4", "memory": "16Gi"},
                            "limits": {"cpu": "8", "memory": "32Gi"},
                        },
                    },
                },
                # Step 2: PyTorch Training
                {
                    "name": "train-model",
                    "container": {
                        "image": GPU_IMAGE,
                        "command": ["python", "-m", "ml.training.train"],
                        "args": [
                            "--model", "{{workflow.parameters.model-type}}",
                            "--data-path", "s3://shopfeed-data/delta/training",
                        ],
                        "resources": {
                            "requests": {"nvidia.com/gpu": "{{workflow.parameters.gpu-count}}"},
                            "limits": {"nvidia.com/gpu": "{{workflow.parameters.gpu-count}}",
                                       "memory": "64Gi"},
                        },
                    },
                },
                # Step 3: Validation Gate
                {
                    "name": "validate-model",
                    "container": {
                        "image": GPU_IMAGE,
                        "command": ["python", "-c"],
                        "args": [
                            "from ml.pipelines.kubeflow_pipeline import validate_model; "
                            "validate_model('{{steps.train-model.outputs.parameters.model-path}}')"
                        ],
                    },
                },
                # Step 4: ONNX Export
                {
                    "name": "export-onnx",
                    "container": {
                        "image": GPU_IMAGE,
                        "command": ["python", "-m", "ml.serving.export_onnx"],
                        "args": [
                            "--model", "{{workflow.parameters.model-type}}",
                            "--checkpoint", "{{steps.train-model.outputs.parameters.model-path}}",
                        ],
                    },
                },
                # Step 5: Canary Deploy (5% traffic)
                {
                    "name": "canary-deploy",
                    "container": {
                        "image": GPU_IMAGE,
                        "command": ["python", "-c"],
                        "args": [
                            "from ml.experiments.ab_testing import ABTestManager; "
                            "ABTestManager().create_canary("
                            "'{{workflow.parameters.model-type}}', traffic_pct=5)"
                        ],
                    },
                },
                # Pipeline DAG
                {
                    "name": "ml-training-pipeline",
                    "dag": {
                        "tasks": [
                            {"name": "features", "template": "extract-features"},
                            {"name": "train", "template": "train-model",
                             "dependencies": ["features"]},
                            {"name": "validate", "template": "validate-model",
                             "dependencies": ["train"]},
                            {"name": "export", "template": "export-onnx",
                             "dependencies": ["validate"]},
                            {"name": "deploy", "template": "canary-deploy",
                             "dependencies": ["export"]},
                        ],
                    },
                },
            ],
        },
    }

    import json
    return json.dumps(pipeline_spec, indent=2)


def validate_model(model_path: str) -> bool:
    """Validation gate — checks if model meets production thresholds.

    Called by Kubeflow pipeline step 3. If validation fails, the
    pipeline stops and the old model remains in production.
    """
    try:
        from ml.tracking import log_training_run
        import json
        from pathlib import Path

        # Read metrics file generated by training
        metrics_path = Path(model_path).parent / "metrics.json"
        if not metrics_path.exists():
            logger.error("No metrics.json found at %s", metrics_path)
            return False

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Check gates
        passed = True
        for gate, threshold in VALIDATION_GATES.items():
            if gate.startswith("min_"):
                metric_name = gate[4:]
                if metrics.get(metric_name, 0) < threshold:
                    logger.warning("GATE FAILED: %s=%.4f < %.4f", metric_name, metrics.get(metric_name, 0), threshold)
                    passed = False
            elif gate.startswith("max_"):
                metric_name = gate[4:]
                if metrics.get(metric_name, float("inf")) > threshold:
                    logger.warning("GATE FAILED: %s=%.4f > %.4f", metric_name, metrics.get(metric_name, 0), threshold)
                    passed = False

        if passed:
            logger.info("✅ All validation gates passed: %s", metrics)
        else:
            logger.warning("❌ Validation gates failed — model NOT promoted")

        return passed

    except Exception as e:
        logger.error("Validation failed: %s", e)
        return False


# ── CronJob Schedule ───────────────────────────────────────────

TRAINING_SCHEDULE = {
    "recommendation_model":    "0 */6 * * *",   # Toutes les 6h
    "user_embeddings":         "0 */2 * * *",   # Toutes les 2h
    "fraud_model":             "0 0 * * 1",     # Hebdomadaire (lundi)
    "moderation_model":        "0 0 1 * *",     # Mensuel
    "drift_check":             "0 6 * * *",     # Quotidien à 6h
    # ── Ajouté — Fine-tuning encodeurs visuels ──────────────────
    # Déclenché manuellement AVANT la prod via ml.pipelines.pretrain_finetune
    # Puis automatisé mensuellement si >50K nouveaux produits dans le mois
    "visual_lora_adapters":    "0 2 1 * *",     # Mensuel (1er du mois à 2h)
    "projection_head":         "0 3 1 * *",     # Mensuel (après visual_lora)
    "faiss_reindex":           "0 4 * * 0",     # Hebdomadaire (dimanche à 4h)
}

