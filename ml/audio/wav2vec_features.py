"""
Wav2Vec2 Features — Extraction de features vocales analytiques.

Modèle: facebook/wav2vec2-base
Features extraites:
  - energy_ratio: ratio énergie vocale vs silence
  - speech_rate_wpm: mots par minute
  - silence_ratio: % silence dans l'audio
  - pitch_variance: variance du pitch (monotone vs dynamique)
  - engagement_score: score composite prédictif engagement

Utilisé pour les recommandations vendeur dans seller_hub/ai_recommendations.
"""

from __future__ import annotations
import asyncio
import logging
import os
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_wav2vec_model = None
_wav2vec_processor = None

WAV2VEC_MODEL = os.environ.get("WAV2VEC_MODEL", "facebook/wav2vec2-base")


def _load_wav2vec():
    global _wav2vec_model, _wav2vec_processor
    if _wav2vec_model is not None:
        return _wav2vec_processor, _wav2vec_model
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
        _wav2vec_model = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
        _wav2vec_model.eval()
        logger.info(f"Wav2Vec2 loaded: {WAV2VEC_MODEL}")
    except ImportError:
        logger.warning("transformers not installed. Wav2Vec2 unavailable.")
    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2: {e}")
    return _wav2vec_processor, _wav2vec_model


async def extract_speech_features(audio_url: str) -> dict:
    """
    Extrait des features vocales analytiques depuis un audio.

    Args:
        audio_url: URL de la vidéo ou audio

    Returns:
        {
            energy_ratio: float,        # ratio énergie vocale vs silence
            speech_rate_wpm: float,     # mots par minute
            silence_ratio: float,       # % silence dans l'audio [0,1]
            pitch_variance: float,      # variance du pitch
            engagement_score: float,    # score composite [0,1]
            recommendations: [str],     # recommandations actionnables
        }
    """
    processor, model = _load_wav2vec()
    if model is None:
        return _default_features()

    loop = asyncio.get_event_loop()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                f.write(resp.content)

        features = await loop.run_in_executor(
            None, lambda: _compute_features(processor, model, tmp_path)
        )
        return features

    except Exception as e:
        logger.error(f"Speech feature extraction failed for {audio_url}: {e}")
        return _default_features()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _compute_features(processor, model, audio_path: str) -> dict:
    """Exécuté dans un thread."""
    try:
        import torch
        import soundfile as sf
        import resampy

        data, sr = sf.read(audio_path)
        if sr != 16000:
            data = resampy.resample(data, sr, 16000)
        if data.ndim > 1:
            data = data.mean(axis=1)

        audio_float = data.astype(np.float32)

        # Energie RMS par frame (40ms)
        frame_size = int(16000 * 0.04)
        frames = [audio_float[i:i+frame_size] for i in range(0, len(audio_float), frame_size)]
        energies = [np.sqrt(np.mean(f**2)) for f in frames if len(f) == frame_size]
        energies = np.array(energies)

        # Seuil de silence (< 10% de l'énergie max)
        silence_threshold = energies.max() * 0.10 if energies.max() > 0 else 0.01
        silence_ratio = float(np.mean(energies < silence_threshold))
        energy_ratio = 1.0 - silence_ratio

        # Wav2Vec2 features
        inputs = processor(audio_float, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # [T, 768]

        # Pitch variance proxy (via variance des features hautes fréquences)
        pitch_variance = float(np.var(hidden[:, -128:].mean(axis=1)))

        # Speech rate proxy (via activations non-silence)
        active_frames = int(len(energies) * energy_ratio)
        duration_s = len(audio_float) / 16000
        # Approximation: ~3 syllabes/sec → ~150 WPM en français
        speech_rate_wpm = (active_frames / max(duration_s, 1)) * 0.5 * 60

        recommendations = []
        if silence_ratio > 0.20:
            recommendations.append(
                f"Ratio de silence élevé ({silence_ratio:.0%}) — cible <15%"
            )
        if speech_rate_wpm < 80:
            recommendations.append("Débit trop lent — parle à ~120-150 mots/min pour garder l'attention")
        if speech_rate_wpm > 200:
            recommendations.append("Débit trop élevé — ralentis pour une meilleure compréhension")
        if pitch_variance < 0.1:
            recommendations.append("Voix monotone détectée — varie le ton pour plus d'engagement")

        return {
            "energy_ratio": round(energy_ratio, 4),
            "speech_rate_wpm": round(speech_rate_wpm, 1),
            "silence_ratio": round(silence_ratio, 4),
            "pitch_variance": round(pitch_variance, 6),
            "engagement_score": 0.0,  # calculé dans engagement_scorer
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Wav2Vec2 feature extraction failed: {e}")
        return _default_features()


def _default_features() -> dict:
    return {
        "energy_ratio": 0.75,
        "speech_rate_wpm": 120.0,
        "silence_ratio": 0.25,
        "pitch_variance": 0.3,
        "engagement_score": 0.5,
        "recommendations": [],
    }
