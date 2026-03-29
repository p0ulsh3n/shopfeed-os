"""
VGGish Encoder — Audio embedding 128d via le modèle VGGish de Google.

Utilisé pour:
  - feed_content.audio_embedding (128d) — stocké en DB
  - Content Embedding Continuity: similarité audio entre contenus consécutifs
  - Diversity injection dans le feed (éviter audios trop similaires)

Modèle: torchvggish ou tensorflow/hub VGGish
"""

from __future__ import annotations
import asyncio
import logging
import os
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_vggish_model = None


def _load_vggish():
    global _vggish_model
    if _vggish_model is not None:
        return _vggish_model
    try:
        import torchvggish
        _vggish_model = torchvggish.vggish()
        _vggish_model.eval()
        logger.info("VGGish model loaded.")
    except ImportError:
        logger.warning("torchvggish not installed. VGGish encoding unavailable.")
    except Exception as e:
        logger.error(f"Failed to load VGGish: {e}")
    return _vggish_model


async def encode_audio(audio_url: str) -> np.ndarray:
    """
    Encode un audio en vecteur 128d via VGGish.

    Args:
        audio_url: URL de la vidéo ou audio (S3, CDN)

    Returns:
        np.ndarray[128] — audio embedding L2-normalisé
        Stocké dans feed_content.audio_embedding.
    """
    model = _load_vggish()
    if model is None:
        logger.warning("VGGish not loaded. Returning zero embedding.")
        return np.zeros(128, dtype=np.float32)

    loop = asyncio.get_event_loop()

    # Télécharger l'audio
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(resp.content)

        embedding = await loop.run_in_executor(
            None, lambda: _run_vggish(model, tmp_path)
        )
        return embedding

    except Exception as e:
        logger.error(f"VGGish encode failed for {audio_url}: {e}")
        return np.zeros(128, dtype=np.float32)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _run_vggish(model, audio_path: str) -> np.ndarray:
    """Exécuté dans un thread executor."""
    try:
        import torch
        import soundfile as sf
        import resampy

        # Charger et resampler à 16kHz (format VGGish)
        data, sr = sf.read(audio_path)
        if sr != 16000:
            data = resampy.resample(data, sr, 16000)

        # Convertir en mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Convertir en tensor
        audio_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            embedding = model(audio_tensor)

        # Moyenne sur le temps si plusieurs frames
        if embedding.ndim > 1:
            embedding = embedding.mean(dim=0)

        emb_np = embedding.cpu().numpy().flatten()[:128]

        # Padding si < 128
        if len(emb_np) < 128:
            emb_np = np.pad(emb_np, (0, 128 - len(emb_np)))

        # L2 normalize
        norm = np.linalg.norm(emb_np)
        return emb_np / max(norm, 1e-8)

    except Exception as e:
        logger.error(f"VGGish processing failed: {e}")
        return np.zeros(128, dtype=np.float32)


def audio_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity entre deux audio embeddings (L2-normalisés)."""
    return float(np.dot(emb1, emb2))
