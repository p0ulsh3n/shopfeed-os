"""
Triton Model Publisher — pushes weights for serving (Section 14)
================================================================
Publishes updated model weights for Triton Inference Server.
Triton polls a model repository directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from .delta_model import DeltaModel

logger = logging.getLogger(__name__)


class TritonModelPublisher:
    """Publishes updated model weights for Triton Inference Server.

    Triton polls a model repository directory. We write updated
    ONNX/TorchScript models there every sync interval.

    Production setup:
        /models/
            delta_model/
                config.pbtxt      (Triton config)
                1/model.onnx      (version 1)
                2/model.onnx      (version 2 — after sync)
    """

    def __init__(
        self,
        model_repo: str = "/models/delta_model",
        export_format: str = "torchscript",  # torchscript | onnx
    ):
        self.model_repo = Path(model_repo)
        self.export_format = export_format
        self._version = 0

    def publish(self, delta_model: DeltaModel, embed_dim: int = 64) -> Path:
        """Export and publish a new model version for Triton."""
        self._version += 1
        version_dir = self.model_repo / str(self._version)
        version_dir.mkdir(parents=True, exist_ok=True)

        delta_model.eval()
        dummy_item = torch.randn(1, embed_dim)
        dummy_user = torch.randn(1, embed_dim)

        if self.export_format == "torchscript":
            path = version_dir / "model.pt"
            traced = torch.jit.trace(delta_model, (dummy_item, dummy_user))
            traced.save(str(path))
        else:  # onnx
            path = version_dir / "model.onnx"
            torch.onnx.export(
                delta_model, (dummy_item, dummy_user), str(path),
                input_names=["item_embedding", "user_embedding"],
                output_names=["delta_score"],
                dynamic_axes={
                    "item_embedding": {0: "batch"},
                    "user_embedding": {0: "batch"},
                },
                opset_version=17,
            )

        logger.info("Triton model published: version=%d, path=%s", self._version, path)
        return path
