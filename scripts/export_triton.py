"""
Export Triton — Export des modèles PyTorch en TorchScript + config Triton Inference Server.

Usage:
  python -m scripts.export_triton --model two_tower --version 1
  python -m scripts.export_triton --model all --output_dir s3://shopfeed-ml-models/triton/
"""

from __future__ import annotations
import argparse
import logging
import os
import json
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRITON_MODEL_DIR = os.environ.get("TRITON_MODEL_REPO", "/opt/triton/models")


def export_torchscript(
    model: torch.nn.Module,
    model_name: str,
    version: int,
    example_inputs: tuple,
    output_dir: str,
) -> str:
    """Trace + exporte un modèle en TorchScript."""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)

    # Structure répertoire Triton: models/{name}/{version}/model.pt
    version_dir = Path(output_dir) / model_name / str(version)
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.pt"
    traced.save(str(model_path))
    logger.info(f"TorchScript exported: {model_path}")
    return str(model_path)


def create_triton_config(
    model_name: str,
    input_shapes: list[dict],
    output_shapes: list[dict],
    output_dir: str,
    max_batch_size: int = 64,
    instance_count: int = 2,
    use_gpu: bool = True,
) -> str:
    """Génère le config.pbtxt pour Triton."""
    platform = "pytorch_libtorch"
    instance_kind = "KIND_GPU" if use_gpu else "KIND_CPU"

    inputs_str = "\n".join([
        f"""input {{
  name: "{inp["name"]}"
  data_type: {inp.get("dtype", "TYPE_FP32")}
  dims: [{", ".join(str(d) for d in inp["dims"])}]
}}""" for inp in input_shapes
    ])

    outputs_str = "\n".join([
        f"""output {{
  name: "{out["name"]}"
  data_type: {out.get("dtype", "TYPE_FP32")}
  dims: [{", ".join(str(d) for d in out["dims"])}]
}}""" for out in output_shapes
    ])

    config = f"""name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}
{inputs_str}
{outputs_str}
instance_group [
  {{
    count: {instance_count}
    kind: {instance_kind}
  }}
]
dynamic_batching {{
  preferred_batch_size: [32, 64]
  max_queue_delay_microseconds: 5000
}}
"""
    config_path = Path(output_dir) / model_name / "config.pbtxt"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config)
    logger.info(f"Triton config written: {config_path}")
    return str(config_path)


# ── Configs par modèle ────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "two_tower": {
        "inputs": [
            {"name": "user_features", "dims": [-1, 410]},
            {"name": "item_features", "dims": [-1, 780]},
        ],
        "outputs": [
            {"name": "user_embedding", "dims": [-1, 256]},
            {"name": "item_embedding", "dims": [-1, 256]},
        ],
        "max_batch_size": 128,
    },
    "mtl_ple": {
        "inputs": [
            {"name": "user_features", "dims": [-1, 410]},
            {"name": "item_features", "dims": [-1, 780]},
            {"name": "context_features", "dims": [-1, 64]},
        ],
        "outputs": [
            {"name": "p_buy_now", "dims": [-1, 1]},
            {"name": "p_purchase", "dims": [-1, 1]},
            {"name": "p_add_to_cart", "dims": [-1, 1]},
            {"name": "p_save", "dims": [-1, 1]},
            {"name": "p_share", "dims": [-1, 1]},
            {"name": "e_watch_time", "dims": [-1, 1]},
            {"name": "p_negative", "dims": [-1, 1]},
        ],
        "max_batch_size": 64,
    },
    "din": {
        "inputs": [
            {"name": "target_item", "dims": [-1, 64]},
            {"name": "user_history", "dims": [-1, 50, 64]},
            {"name": "user_features", "dims": [-1, 128]},
        ],
        "outputs": [{"name": "ctr_score", "dims": [-1, 1]}],
        "max_batch_size": 64,
    },
}


def export_model(model_name: str, version: int, output_dir: str) -> None:
    """Exporte un modèle en TorchScript + config Triton."""
    from ml.serving.registry import ModelRegistry
    registry = ModelRegistry()

    try:
        model = getattr(registry, f"load_{model_name}")()
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return

    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        logger.warning(f"No Triton config defined for {model_name}. Skipping.")
        return

    # Créer les inputs example pour le tracing
    example_inputs = tuple(
        torch.zeros(1, *inp["dims"][1:], dtype=torch.float32)
        for inp in cfg["inputs"]
    )

    export_torchscript(model, model_name, version, example_inputs, output_dir)
    create_triton_config(
        model_name,
        cfg["inputs"],
        cfg["outputs"],
        output_dir,
        max_batch_size=cfg.get("max_batch_size", 64),
    )


def main():
    parser = argparse.ArgumentParser(description="Export models to TorchScript + Triton config")
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--output_dir", default=TRITON_MODEL_DIR)
    args = parser.parse_args()

    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    for m in models:
        logger.info(f"Exporting {m} v{args.version}...")
        export_model(m, args.version, args.output_dir)

    logger.info(f"Export complete. Models at: {args.output_dir}")


if __name__ == "__main__":
    main()
