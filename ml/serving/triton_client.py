"""
Triton Inference Server client — remplace triton_publisher.py du monolith.
Client HTTP pour envoyer des delta weights au Triton Inference Server.
Gère aussi le chargement et le reload de modèles en production.
"""

from __future__ import annotations
import logging
import json
from typing import Any, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

import os as _os

def _get_triton_config() -> dict:
    try:
        from ml.config_loader import get_infrastructure_config
        return get_infrastructure_config().get("triton", {})
    except Exception:
        return {}

_TRITON_CFG = _get_triton_config()

# URLs Triton — priorité : env var > infrastructure.yaml > fallback Docker hostname
TRITON_URL         = _os.environ.get("TRITON_URL",         _TRITON_CFG.get("http_url",     "http://triton:8000"))
TRITON_GRPC_URL    = _os.environ.get("TRITON_GRPC_URL",    _TRITON_CFG.get("grpc_url",     "triton:8001"))
TRITON_METRICS_URL = _os.environ.get("TRITON_METRICS_URL", _TRITON_CFG.get("metrics_url",  "http://triton:8002"))


class TritonClient:
    """
    Client HTTP pour Triton Inference Server.
    Supporte:
      - Inférence HTTP (v2 protocol)
      - Chargement / reload de modèles
      - Health check
      - Push delta weights depuis le monolith streaming trainer
    """

    def __init__(self, base_url: str = TRITON_URL, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def health(self) -> bool:
        """Retourne True si Triton est disponible."""
        try:
            r = await self._client.get("/v2/health/ready")
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"Triton health check failed: {e}")
            return False

    async def list_models(self) -> list[dict]:
        """Retourne la liste des modèles chargés sur Triton."""
        try:
            r = await self._client.get("/v2/repository/index")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"Triton list models failed: {e}")
            return []

    async def load_model(self, model_name: str) -> bool:
        """Charge un modèle depuis le model repository."""
        try:
            r = await self._client.post(f"/v2/repository/models/{model_name}/load")
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Triton load model {model_name} failed: {e}")
            return False

    async def unload_model(self, model_name: str) -> bool:
        """Décharge un modèle de la mémoire GPU."""
        try:
            r = await self._client.post(f"/v2/repository/models/{model_name}/unload")
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Triton unload model {model_name} failed: {e}")
            return False

    async def infer(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
        model_version: str = "1",
    ) -> dict[str, np.ndarray]:
        """
        Inférence HTTP v2 protocol.

        Args:
            model_name:    nom du modèle Triton
            inputs:        {input_name: numpy_array}
            model_version: version du modèle (default "1")

        Returns:
            {output_name: numpy_array}
        """
        # Construire le payload v2 HTTP
        triton_inputs = []
        for name, arr in inputs.items():
            triton_inputs.append({
                "name": name,
                "shape": list(arr.shape),
                "datatype": _numpy_dtype_to_triton(arr.dtype),
                "data": arr.flatten().tolist(),
            })

        payload = {
            "inputs": triton_inputs,
            "outputs": [],
        }

        try:
            r = await self._client.post(
                f"/v2/models/{model_name}/versions/{model_version}/infer",
                json=payload,
            )
            r.raise_for_status()
            response = r.json()

            outputs = {}
            for out in response.get("outputs", []):
                dtype = _triton_dtype_to_numpy(out["datatype"])
                arr = np.array(out["data"], dtype=dtype).reshape(out["shape"])
                outputs[out["name"]] = arr

            return outputs
        except Exception as e:
            logger.error(f"Triton inference failed for {model_name}: {e}")
            raise

    async def push_delta_weights(
        self,
        model_name: str,
        delta_weights: dict[str, np.ndarray],
        step: int,
    ) -> bool:
        """
        Envoie des delta weights au Triton pour mise à jour online.
        Utilisé par ml/monolith/triton_publisher.py toutes les 5-15min.

        Protocol custom: POST /v2/models/{name}/delta
        """
        try:
            serializable = {
                "step": step,
                "model": model_name,
                "deltas": {
                    k: v.flatten().tolist()
                    for k, v in delta_weights.items()
                },
                "shapes": {
                    k: list(v.shape)
                    for k, v in delta_weights.items()
                },
            }
            r = await self._client.post(
                f"/v2/models/{model_name}/delta",
                json=serializable,
            )
            if r.status_code == 200:
                logger.info(f"Delta weights pushed to Triton for {model_name} (step={step})")
                return True
            logger.warning(f"Triton delta push returned {r.status_code}")
            return False
        except Exception as e:
            logger.error(f"Triton delta push failed: {e}")
            return False

    async def get_model_stats(self, model_name: str) -> dict:
        """Retourne les stats d'inférence d'un modèle."""
        try:
            r = await self._client.get(f"/v2/models/{model_name}/stats")
            r.raise_for_status()
            return r.json()
        except Exception:
            return {}

    async def close(self) -> None:
        await self._client.aclose()


# ── Helpers de conversion dtype ──────────────────────────────────────────────

def _numpy_dtype_to_triton(dtype: np.dtype) -> str:
    mapping = {
        np.float32: "FP32",
        np.float16: "FP16",
        np.float64: "FP64",
        np.int32: "INT32",
        np.int64: "INT64",
        np.uint8: "UINT8",
        np.bool_: "BOOL",
    }
    for np_type, triton_type in mapping.items():
        if np.issubdtype(dtype, np_type):
            return triton_type
    return "FP32"


def _triton_dtype_to_numpy(triton_dtype: str) -> np.dtype:
    mapping = {
        "FP32": np.float32,
        "FP16": np.float16,
        "FP64": np.float64,
        "INT32": np.int32,
        "INT64": np.int64,
        "UINT8": np.uint8,
        "BOOL": np.bool_,
    }
    return mapping.get(triton_dtype, np.float32)


# ── Singleton global ─────────────────────────────────────────────────────────

_triton_client: Optional[TritonClient] = None


def get_triton_client() -> TritonClient:
    global _triton_client
    if _triton_client is None:
        import os
        url = os.environ.get("TRITON_URL", TRITON_URL)
        _triton_client = TritonClient(base_url=url)
    return _triton_client
