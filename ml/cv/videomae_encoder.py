"""
VideoMAE Video Encoder (archi-2026 §9.3)
==========================================
Temporal video embeddings using VideoMAE — captures motion patterns
that frame-by-frame CLIP cannot see.

Use cases:
    - Product demonstrations (rotation, unboxing)
    - Live action classification (dance, cooking, sports)
    - Video similarity beyond visual thumbnails

Architecture:
    VideoMAE extracts spatio-temporal features from 16 uniformly-sampled
    frames, producing a single 768D embedding that encodes both appearance
    AND motion.

CLIP vs VideoMAE:
    - CLIP: frame-by-frame, no temporal understanding
    - VideoMAE: understands how frames relate over time (movement, transitions)

Requires:
    pip install transformers>=4.40 decord>=0.6
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import VideoMAEModel, VideoMAEImageProcessor
    HAS_VIDEOMAE = True
except ImportError:
    HAS_VIDEOMAE = False
    logger.warning("transformers not installed — VideoMAE disabled. pip install transformers>=4.40")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class VideoMAEEncoder:
    """Production video encoder using VideoMAE.

    VideoMAE (Masked Autoencoder for Video) learns spatio-temporal
    representations by masking 90% of video patches and predicting
    the missing content.

    In production:
        1. Video uploaded → decode 16 frames (uniform sampling)
        2. VideoMAE encodes → 768D embedding
        3. Embedding stored in Milvus for similarity search
        4. Also used as input features for the recommendation pipeline
    """

    # Model configuration
    MODEL_NAME = "MCG-NJU/videomae-base"
    EMBED_DIM = 768
    N_FRAMES = 16

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.model_name = model_name or self.MODEL_NAME
        self.device = device
        self._model = None
        self._processor = None

    @property
    def processor(self) -> Optional[Any]:
        """Lazy-load VideoMAE processor."""
        if self._processor is None and HAS_VIDEOMAE:
            try:
                self._processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
            except Exception as e:
                logger.error("VideoMAE processor load failed: %s", e)
        return self._processor

    @property
    def model(self) -> Optional[Any]:
        """Lazy-load VideoMAE model (frozen, inference only)."""
        if self._model is None and HAS_VIDEOMAE and HAS_TORCH:
            try:
                self._model = VideoMAEModel.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                for param in self._model.parameters():
                    param.requires_grad = False
                logger.info("VideoMAE loaded: %s on %s", self.model_name, self.device)
            except Exception as e:
                logger.error("VideoMAE model load failed: %s", e)
        return self._model

    def sample_frames(self, video_path: str, n_frames: int = 16) -> Optional[list]:
        """Sample n_frames uniformly from a video file.

        Uses decord for fast GPU-accelerated video decoding.
        Falls back to OpenCV if decord is unavailable.
        """
        if not HAS_NUMPY:
            return None

        try:
            import decord
            decord.bridge.set_bridge("numpy")
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)

            if total_frames < n_frames:
                indices = list(range(total_frames))
                # Pad with last frame if video is too short
                indices += [total_frames - 1] * (n_frames - total_frames)
            else:
                indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

            frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)
            return list(frames)

        except ImportError:
            logger.info("decord not available, trying cv2")
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if total_frames <= 0:
                    return None

                indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    elif frames:
                        frames.append(frames[-1])  # Repeat last frame
                cap.release()
                return frames if len(frames) == n_frames else None

            except ImportError:
                logger.error("Neither decord nor cv2 available for video decoding")
                return None

    def encode_frames(self, frames: list) -> Optional[Any]:
        """Encode a list of frames into a single video embedding.

        Args:
            frames: list of N numpy arrays (H, W, C) in RGB

        Returns:
            Tensor of shape (768,) — the video embedding
        """
        if not HAS_TORCH or not self.model or not self.processor:
            return None

        try:
            # VideoMAE expects (batch, n_frames, C, H, W)
            inputs = self.processor(frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pool the sequence embeddings → single vector
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # (768,)

            return embedding.cpu()

        except Exception as e:
            logger.error("VideoMAE encoding failed: %s", e)
            return None

    def encode_video(self, video_path: str) -> Optional[Any]:
        """Full pipeline: video file → 768D embedding.

        This is the main entry point for production use.
        """
        frames = self.sample_frames(video_path, self.N_FRAMES)
        if frames is None:
            return None
        return self.encode_frames(frames)

    def encode_batch(self, video_paths: list[str]) -> list[Optional[Any]]:
        """Encode multiple videos. Non-failing — returns None for errors."""
        results = []
        for path in video_paths:
            try:
                emb = self.encode_video(path)
                results.append(emb)
            except Exception as e:
                logger.error("Batch encode failed for %s: %s", path, e)
                results.append(None)
        return results
