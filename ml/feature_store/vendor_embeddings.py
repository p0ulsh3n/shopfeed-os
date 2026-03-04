"""
Vendor Embedding Table (Section 41 — ✏️ TRAINABLE)
====================================================
Vendor embeddings learned from interactions.
These are the ONLY embeddings trained from scratch (not pre-trained).
"""

from __future__ import annotations

import torch


class VendorEmbeddingTable:
    """Vendor embeddings learned from interactions.

    Section 41: "100% appris depuis tes données. Initialisé aléatoirement.
    Capture le style et la réputation de chaque vendeur."

    These are the ONLY embeddings trained from scratch (not pre-trained).
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self._table: dict[str, torch.Tensor] = {}

    def get(self, vendor_id: str) -> torch.Tensor:
        if vendor_id not in self._table:
            # Initialize randomly — will be learned during fine-tuning
            self._table[vendor_id] = torch.randn(self.embed_dim) * 0.01
        return self._table[vendor_id]

    def update(self, vendor_id: str, embedding: torch.Tensor) -> None:
        self._table[vendor_id] = embedding.detach().clone()
