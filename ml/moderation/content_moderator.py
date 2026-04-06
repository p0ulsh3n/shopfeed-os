"""
Content Moderation Engine (archi-2026 §9.4)
=============================================
Multi-level content moderation combining fast perceptual hashing
with deep ViT classification and Llama Scout explainability.

Architecture:
    Level 1 — Perceptual hash (pHash) instant blocking (<1ms)
    Level 2 — ViT frame-by-frame classification (<30s after upload)
    Level 3 — Llama Scout explanation (WHY was it flagged, transparent)
    Level 4 — Human review for ambiguous cases (score 0.5-0.85)

Violation categories:
    violence, nudity, self_harm, hate_symbols,
    dangerous_activity, spam, copyright, misinformation,
    counterfeit, regulated_product, price_manipulation

Requires:
    pip install transformers>=4.40 Pillow>=10
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from transformers import ViTModel, ViTImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not installed — content moderation disabled. pip install transformers>=4.40")


# ── Violation Categories (expanded for marketplace 2026) ──────

VIOLATION_CATEGORIES = [
    "violence",
    "nudity",
    "self_harm",
    "hate_symbols",
    "dangerous_activity",
    "spam",
    "copyright",
    "misinformation",
    "counterfeit",           # Fake branded products
    "regulated_product",     # Weapons, drugs, tobacco, etc.
    "price_manipulation",    # Fake discounts, inflated compare_at_price
]

# Thresholds de modération (configurable via configs/infrastructure.yaml)
def _get_moderation_config() -> dict:
    try:
        from ml.config_loader import get_infrastructure_config
        return get_infrastructure_config().get("moderation", {})
    except Exception:
        return {}

_MOD_CFG = _get_moderation_config()

AUTO_REMOVE_THRESHOLD  = float(_MOD_CFG.get("auto_remove_threshold",  0.85))
HUMAN_REVIEW_THRESHOLD = float(_MOD_CFG.get("human_review_threshold", 0.50))
ALLOW_THRESHOLD        = float(_MOD_CFG.get("human_review_threshold", 0.50))

# URL du LLM pour les explications Scout — lu depuis infrastructure.yaml
# Override via env var MODERATION_LLM_URL ou SCOUT_URL
import os as _os
LLAMA_VLLM_BASE = (
    _os.environ.get("MODERATION_LLM_URL")
    or _MOD_CFG.get("llm_base_url")
    or _os.environ.get("SCOUT_URL", "http://localhost:8200") + "/v1"
)


class ContentModerator(nn.Module if HAS_TORCH else object):
    """Multi-level content moderation engine.

    Architecture:
        - Backbone: ViT-Base-Patch16-224 (pre-trained, fine-tuned on moderation data)
        - Temporal pooling: Multi-head attention over frame embeddings
        - Classifier: Linear head → per-category violation scores
        - Explainer: Llama Scout provides human-readable explanation

    In production:
        1. Video uploaded → extract 16 frames uniformly
        2. Each frame → ViT backbone → 768D embedding
        3. Temporal attention pools 16 embeddings → 1 video representation
        4. Classifier outputs score [0, 1] per violation category
        5. If flagged → Llama Scout explains WHY (transparent to vendor)
    """

    N_FRAMES = 16
    EMBED_DIM = 768  # ViT-Base output dim

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        if HAS_TORCH:
            super().__init__()

        self.model_name = model_name
        self._processor = None
        self._backbone = None

        if HAS_TORCH:
            # Temporal pooling: aggregate frame embeddings
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=self.EMBED_DIM,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
            self.temporal_norm = nn.LayerNorm(self.EMBED_DIM)

            # Classification head (expanded for new categories)
            self.classifier = nn.Sequential(
                nn.Linear(self.EMBED_DIM, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, len(VIOLATION_CATEGORIES)),
                nn.Sigmoid(),
            )

    @property
    def processor(self):
        """Lazy-load ViT image processor."""
        if self._processor is None and HAS_TRANSFORMERS:
            self._processor = ViTImageProcessor.from_pretrained(self.model_name)
        return self._processor

    @property
    def backbone(self):
        """Lazy-load ViT backbone (frozen)."""
        if self._backbone is None and HAS_TRANSFORMERS and HAS_TORCH:
            self._backbone = ViTModel.from_pretrained(self.model_name)
            for param in self._backbone.parameters():
                param.requires_grad = False
            self._backbone.eval()
        return self._backbone

    def extract_frame_embeddings(self, frames: list) -> Optional[Any]:
        """Extract ViT embeddings from video frames.

        Args:
            frames: list of PIL.Image or numpy arrays (RGB)

        Returns:
            Tensor of shape (1, N_frames, 768)
        """
        if not HAS_TORCH or not HAS_TRANSFORMERS or not self.processor or not self.backbone:
            return None

        inputs = self.processor(images=frames, return_tensors="pt")
        with torch.no_grad():
            outputs = self.backbone(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # (N_frames, 768)

        return embeddings.unsqueeze(0)  # (1, N_frames, 768)

    def forward(self, frame_embeddings: Any) -> Any:
        """Forward pass: frame embeddings → violation scores.

        Args:
            frame_embeddings: (batch, N_frames, 768)

        Returns:
            (batch, n_categories) — violation probability per category
        """
        if not HAS_TORCH:
            return None

        # Temporal attention pooling
        attn_out, _ = self.temporal_attn(
            frame_embeddings, frame_embeddings, frame_embeddings,
        )
        pooled = self.temporal_norm(attn_out.mean(dim=1))  # (batch, 768)

        # Classification
        scores = self.classifier(pooled)  # (batch, n_categories)
        return scores

    def moderate_video(self, frames: list) -> dict[str, Any]:
        """Full moderation pipeline: frames → scores → action → explanation.

        Args:
            frames: list of PIL.Image (16 uniformly sampled frames)

        Returns:
            {
                "scores": {"violence": 0.12, "nudity": 0.03, ...},
                "max_score": 0.12,
                "max_category": "violence",
                "action": "allow" | "review" | "remove",
            }
        """
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logger.warning("Content moderation unavailable — sending to human review (fail-safe)")
            return {
                "scores": {cat: 0.0 for cat in VIOLATION_CATEGORIES},
                "max_score": 0.0,
                "max_category": "none",
                "action": "review",
                "error": "dependencies not installed — queued for human review",
            }

        try:
            embeddings = self.extract_frame_embeddings(frames)
            if embeddings is None:
                return {"action": "review", "error": "embedding failed — queued for human review"}

            self.eval()
            with torch.no_grad():
                scores_tensor = self.forward(embeddings)

            scores = {
                cat: round(float(scores_tensor[0, i]), 4)
                for i, cat in enumerate(VIOLATION_CATEGORIES)
            }

            max_score = max(scores.values())
            max_category = max(scores, key=scores.get)

            if max_score >= AUTO_REMOVE_THRESHOLD:
                action = "remove"
            elif max_score >= HUMAN_REVIEW_THRESHOLD:
                action = "review"
            else:
                action = "allow"

            return {
                "scores": scores,
                "max_score": round(max_score, 4),
                "max_category": max_category,
                "action": action,
            }

        except Exception as e:
            logger.error("Moderation failed: %s — sending to human review", e)
            return {"action": "review", "error": str(e)}

    async def explain_violation(
        self,
        product_title: str,
        image_url: str,
        scores: dict[str, float],
        action: str,
    ) -> dict[str, Any]:
        """Level 3: Llama Scout explains WHY content was flagged.

        Called after ViT scoring when action is 'review' or 'remove'.
        Provides vendor-friendly, transparent explanations.

        Returns:
            {
                "explanation": "This image was flagged because...",
                "severity": "low" | "medium" | "high" | "critical",
                "suggested_fixes": ["Crop the image to...", "Replace background..."],
                "false_positive_likelihood": 0.0-1.0,
                "appeal_recommendation": "approve" | "needs_review" | "reject"
            }
        """
        try:
            from ml.llm.llm_enrichment import explain_moderation
            result = await explain_moderation(
                product_title=product_title,
                image_url=image_url,
                moderation_scores=scores,
                moderation_action=action,
                flagged_reasons=[
                    cat for cat, score in scores.items()
                    if score >= HUMAN_REVIEW_THRESHOLD
                ],
            )
            return result
        except Exception as e:
            logger.error("Llama explanation failed: %s", e)
            return {
                "explanation": "Automated review flagged potential policy violation.",
                "severity": "medium",
                "suggested_fixes": ["Please review your content against our policies."],
                "false_positive_likelihood": 0.5,
                "appeal_recommendation": "needs_review",
            }


# ── Perceptual Hash — Level 1 Instant Block ────────────────────

class PerceptualHashChecker:
    """Level 1: instant blocking via perceptual hash matching.

    Compares the video's thumbnail/frames against a database of
    known harmful content hashes (CSAM, terrorism, etc.).

    This runs BEFORE the ViT classifier (<1ms).
    """

    def __init__(self):
        # In production: loaded from a secure, encrypted Redis set
        self._blocked_hashes: set[str] = set()

    def add_blocked_hash(self, phash: str) -> None:
        """Add a content hash to the block list."""
        self._blocked_hashes.add(phash)

    def remove_blocked_hash(self, phash: str) -> None:
        """Remove a content hash from the block list."""
        self._blocked_hashes.discard(phash)

    def check_frame(self, frame_hash: str) -> bool:
        """Returns True if the frame matches a blocked hash."""
        return frame_hash in self._blocked_hashes

    def check_hamming_distance(self, hash1: str, hash2: str, threshold: int = 10) -> bool:
        """Check if two hashes are similar within Hamming distance threshold.

        Perceptual hashes that differ by < threshold bits are considered
        the same image (robust to compression, resizing, minor edits).
        """
        if len(hash1) != len(hash2):
            return False
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            distance = bin(val1 ^ val2).count("1")
            return distance <= threshold
        except ValueError:
            return False

    def check_frame_fuzzy(self, frame_hash: str, threshold: int = 10) -> bool:
        """Check if frame matches ANY blocked hash within Hamming distance."""
        for blocked in self._blocked_hashes:
            if self.check_hamming_distance(frame_hash, blocked, threshold):
                return True
        return False

    @staticmethod
    def compute_phash(image: Any, hash_size: int = 16) -> str:
        """Compute perceptual hash of an image.

        Perceptual hashes are robust to resizing, compression, and
        minor edits — unlike cryptographic hashes.
        """
        if not HAS_PIL:
            return ""

        try:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # Resize to hash_size × hash_size
            img = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)

            # Binary hash: 1 if pixel > average, 0 otherwise
            bits = "".join("1" if p > avg else "0" for p in pixels)
            # Convert to hex
            return hex(int(bits, 2))[2:]
        except Exception as e:
            logger.error("pHash computation failed: %s", e)
            return ""
