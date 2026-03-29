"""
Vendor Embedding Table (Section 41 — ✏️ TRAINABLE)
====================================================
Vendor embeddings learned from interactions.
These are the ONLY embeddings trained from scratch (not pre-trained).

BUG #6 FIX: Previously used a plain Python dict to store tensors — those
embeddings were completely invisible to PyTorch's state_dict() mechanism
and lost on every process restart.

Fixed by implementing VendorEmbeddingTable as an nn.Module backed by
nn.Embedding, so embeddings are properly serialized in checkpoints.
A string→int registry is maintained alongside for str vendor IDs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VendorEmbeddingTable(nn.Module):
    """Vendor embeddings learned from interactions.

    Section 41: "100% appris depuis tes données. Initialisé aléatoirement.
    Capture le style et la réputation de chaque vendeur."

    These are the ONLY embeddings trained from scratch (not pre-trained).

    BUG #6 FIX: Backed by nn.Embedding so embeddings are included in
    model.state_dict() and survive process restarts via checkpointing.
    The string→int registry is serialized separately via save_registry()
    / load_registry().
    """

    def __init__(self, max_vendors: int = 100_000, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_vendors = max_vendors

        # nn.Embedding: weights are part of state_dict() → saved in checkpoints
        # Index 0 is reserved as the padding / unknown-vendor vector.
        self.embedding = nn.Embedding(max_vendors, embed_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=0.01)
        self.embedding.weight.data[0].zero_()  # padding idx = zero vector

        # String vendor_id → integer index mapping (serialized separately)
        self._id_to_idx: dict[str, int] = {}
        self._next_idx: int = 1  # 0 = padding/unknown

    # ── Index Management ─────────────────────────────────────────────

    def get_or_create_idx(self, vendor_id: str) -> int:
        """Return the integer index for a vendor_id, creating one if needed."""
        if vendor_id not in self._id_to_idx:
            if self._next_idx >= self.max_vendors:
                logger.warning(
                    "VendorEmbeddingTable capacity reached (%d). "
                    "Increase max_vendors or clean up stale entries.",
                    self.max_vendors,
                )
                return 0  # Return padding/unknown index
            self._id_to_idx[vendor_id] = self._next_idx
            self._next_idx += 1
        return self._id_to_idx[vendor_id]

    # ── Public API ───────────────────────────────────────────────────

    def get(self, vendor_id: str) -> torch.Tensor:
        """Get the embedding vector for a vendor (creates it if new)."""
        idx = self.get_or_create_idx(vendor_id)
        return self.embedding(torch.tensor(idx, dtype=torch.long))

    def get_batch(self, vendor_ids: list[str]) -> torch.Tensor:
        """Get embeddings for a list of vendor IDs — [N, embed_dim]."""
        indices = torch.tensor(
            [self.get_or_create_idx(vid) for vid in vendor_ids],
            dtype=torch.long,
        )
        return self.embedding(indices)

    def update(self, vendor_id: str, embedding: torch.Tensor) -> None:
        """Manually set an embedding (e.g., from external pre-training).

        For gradient-based updates, use the standard PyTorch optimizer
        path — do NOT call this during training.
        """
        idx = self.get_or_create_idx(vendor_id)
        with torch.no_grad():
            self.embedding.weight[idx] = embedding.detach()

    # ── Serialization ────────────────────────────────────────────────

    def save_registry(self, path: str | Path) -> None:
        """Save the string→int registry to a JSON file.

        Call this alongside torch.save() of the model state_dict() to
        ensure full checkpoint fidelity (embedding weights + ID mapping).
        """
        registry = {
            "_id_to_idx": self._id_to_idx,
            "_next_idx": self._next_idx,
        }
        Path(path).write_text(json.dumps(registry, indent=2))
        logger.info("Vendor registry saved: %s (%d vendors)", path, len(self._id_to_idx))

    def load_registry(self, path: str | Path) -> None:
        """Load the string→int registry from a JSON file.

        Call this after loading the model state_dict() to restore full
        vendor ID→embedding mapping.
        """
        data = json.loads(Path(path).read_text())
        self._id_to_idx = data["_id_to_idx"]
        self._next_idx = data["_next_idx"]
        logger.info("Vendor registry loaded: %s (%d vendors)", path, len(self._id_to_idx))

    @property
    def n_vendors(self) -> int:
        """Number of known vendors (excluding padding slot 0)."""
        return len(self._id_to_idx)

    def __repr__(self) -> str:
        return (
            f"VendorEmbeddingTable("
            f"n_vendors={self.n_vendors}, "
            f"max_vendors={self.max_vendors}, "
            f"embed_dim={self.embed_dim})"
        )
