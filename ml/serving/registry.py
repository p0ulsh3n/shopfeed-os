"""Model Serving — loads trained models and serves inference.

This module runs on the API servers (not GPU). It loads:
    - FAISS index for Two-Tower retrieval
    - DeepFM for pre-ranking
    - MTL/PLE for final ranking
    - DIN / DIEN / BST for marketplace ranking
    - Monolith delta model for real-time corrections

In production (2026 stack):
    - Models are served via Triton Inference Server
    - FAISS index is loaded directly for <10ms retrieval
    - This registry acts as a client to Triton, or as a
      standalone fallback for smaller deployments
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


# Models we support — matches train.py choices
SUPPORTED_MODELS = (
    "two_tower",   # Retrieval (ANN)
    "deepfm",      # Pre-ranking
    "mtl",         # Final ranking (PLE multi-task)
    "din",         # Marketplace ranking (attention-based)
    "dien",        # Sequential ranking (interest evolution)
    "bst",         # Transformer ranking (complex patterns)
    "delta",       # Monolith online delta model
)


class ModelRegistry:
    """Singleton model registry — loads all models once at API startup.

    All models are loaded into CPU/GPU memory and shared across all
    request handlers. Thread-safe for inference (no gradient computation).

    Model priority for ranking (Section 04):
        1. Two-Tower → ANN retrieval (2,000 candidates)
        2. DeepFM → pre-ranking (→ 400 candidates)
        3. DIN/DIEN/BST → marketplace scoring (→ 80)
        4. MTL/PLE → final ranking with multi-task heads
        5. Delta (Monolith) → real-time corrections
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._models: dict[str, Any] = {}
        self._faiss_index = None
        self._item_ids: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_all(self, checkpoint_dir: str = "checkpoints") -> None:
        """Load all trained model checkpoints."""
        base = Path(checkpoint_dir)

        # Core pipeline models
        for model_name in SUPPORTED_MODELS:
            model_dir = base / model_name
            best_path = model_dir / "model_best.pt"
            latest_path = model_dir / "model_latest.pt"

            if best_path.exists():
                self._load_model(model_name, best_path)
            elif latest_path.exists():
                self._load_model(model_name, latest_path)

        # Monolith delta model (separate path)
        monolith_path = base / "monolith" / "delta_model_latest.pt"
        if monolith_path.exists():
            self._load_model("delta", monolith_path)

        # FAISS index
        self._load_faiss(base / "faiss_index")

        logger.info(
            "ModelRegistry: %d models loaded: %s",
            len(self._models), list(self._models.keys()),
        )

    def _load_model(self, name: str, path: Path) -> None:
        if not path.exists():
            logger.warning("Model checkpoint not found: %s", path)
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self._models[name] = checkpoint

        n_params = len(checkpoint.get("model_state_dict", {}))
        logger.info("Loaded %s from %s (%d state entries)", name, path.name, n_params)

    def _load_faiss(self, index_dir: Path) -> None:
        index_path = index_dir / "item.index"
        ids_path = index_dir / "item_ids.json"

        if not index_path.exists():
            logger.warning("FAISS index not found: %s", index_path)
            return

        try:
            import faiss
            self._faiss_index = faiss.read_index(str(index_path))
            with open(ids_path) as f:
                self._item_ids = json.load(f)
            logger.info("FAISS index loaded: %d items", self._faiss_index.ntotal)
        except Exception as exc:
            logger.warning("FAISS load failed: %s", exc)

    # ── Retrieval ────────────────────────────────────────────

    def retrieve_candidates(
        self, user_embedding: np.ndarray, top_k: int = 2000
    ) -> list[tuple[str, float]]:
        """Two-Tower ANN retrieval — <10ms for 10M+ items.

        Args:
            user_embedding: (256,) L2-normalized user vector

        Returns:
            List of (item_id, similarity_score) tuples
        """
        if self._faiss_index is None:
            return []

        query = user_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss_index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self._item_ids):
                results.append((self._item_ids[idx], float(score)))

        return results

    # ── Model access ─────────────────────────────────────────

    def get_model_state(self, name: str) -> dict | None:
        return self._models.get(name)

    def has_model(self, name: str) -> bool:
        return name in self._models

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0 or self._faiss_index is not None


# ═══════════════════════════════════════════════════════════════
# Triton Client (for production deployments)
# ═══════════════════════════════════════════════════════════════

class TritonInferenceClient:
    """Client for Triton Inference Server (production serving).

    In production, models are deployed to Triton and this client
    sends inference requests via gRPC. Falls back to local
    ModelRegistry for development.

    Usage:
        client = TritonInferenceClient("triton-server:8001")
        scores = await client.predict_din(user_features, item_features)
    """

    def __init__(self, triton_url: str = "localhost:8001"):
        self._url = triton_url
        self._client = None

    async def connect(self) -> bool:
        try:
            import tritonclient.grpc.aio as grpcclient
            self._client = grpcclient.InferenceServerClient(url=self._url)
            is_ready = await self._client.is_server_ready()
            logger.info("Triton connected: %s (ready=%s)", self._url, is_ready)
            return is_ready
        except ImportError:
            logger.warning("tritonclient not installed — using local ModelRegistry")
            return False
        except Exception as e:
            logger.warning("Triton connection failed: %s", e)
            return False

    async def predict(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        output_names: list[str] | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Send inference request to Triton.

        Args:
            model_name: e.g. "din_ranking", "bst_attention"
            inputs: {name: numpy_array} for each input tensor
            output_names: which outputs to request

        Returns:
            {output_name: numpy_array} or None if unavailable
        """
        if self._client is None:
            return None

        try:
            import tritonclient.grpc.aio as grpcclient

            triton_inputs = []
            for name, data in inputs.items():
                inp = grpcclient.InferInput(name, data.shape, "FP32")
                inp.set_data_from_numpy(data)
                triton_inputs.append(inp)

            triton_outputs = None
            if output_names:
                triton_outputs = [
                    grpcclient.InferRequestedOutput(n) for n in output_names
                ]

            result = await self._client.infer(
                model_name=model_name,
                inputs=triton_inputs,
                outputs=triton_outputs,
            )

            outputs = {}
            for name in (output_names or []):
                outputs[name] = result.as_numpy(name)
            return outputs

        except Exception as e:
            logger.warning("Triton inference failed (%s): %s", model_name, e)
            return None
