"""
Cuckoo Embedding Table (concept from Monolith paper)
=====================================================
Collision-free embedding table using Cuckoo Hashing.
LRU eviction of products inactive >30 days.
Embeddings of new products initialized via content features (CLIP).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class _CuckooEntry:
    item_id: str
    created_at: float = field(default_factory=time.time)


class CuckooEmbeddingTable:
    """Collision-free embedding table using Cuckoo Hashing.

    Paper concept: "Table sans collision. Eviction LRU des produits inactifs >30j.
    Embeddings nouveaux produits initialisés via content features (CLIP)."

    Cuckoo hashing: two hash functions, O(1) guaranteed lookup.
    On collision, existing entry kicked to alternate slot.
    LRU eviction keeps table bounded.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        capacity: int = 2_000_000,
        max_eviction_age_days: int = 30,
        max_kicks: int = 500,
    ):
        self.embed_dim = embed_dim
        self.capacity = capacity
        self.max_eviction_age_s = max_eviction_age_days * 86400
        self.max_kicks = max_kicks

        # Dual hash tables for cuckoo hashing
        self._table_a: dict[int, _CuckooEntry] = {}
        self._table_b: dict[int, _CuckooEntry] = {}

        # LRU tracking: item_id → last_access_timestamp
        self._access_order: OrderedDict[str, float] = OrderedDict()

        # Embedding storage: item_id → tensor
        self._embeddings: dict[str, torch.Tensor] = {}

        logger.info(
            "CuckooEmbeddingTable: dim=%d, capacity=%d, eviction=%dd",
            embed_dim, capacity, max_eviction_age_days,
        )

    def _hash_a(self, item_id: str) -> int:
        return int(hashlib.md5(item_id.encode()).hexdigest()[:8], 16) % self.capacity

    def _hash_b(self, item_id: str) -> int:
        return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % self.capacity

    def get(self, item_id: str) -> torch.Tensor | None:
        """O(1) lookup. Returns None if not found."""
        for table, h in [(self._table_a, self._hash_a(item_id)),
                         (self._table_b, self._hash_b(item_id))]:
            entry = table.get(h)
            if entry is not None and entry.item_id == item_id:
                self._touch(item_id)
                return self._embeddings.get(item_id)
        return None

    def put(
        self,
        item_id: str,
        embedding: torch.Tensor,
        initial_clip: torch.Tensor | None = None,
    ) -> bool:
        """Insert or update. New items initialized from CLIP content features."""
        # Update existing
        if self.get(item_id) is not None:
            self._embeddings[item_id] = embedding.detach().clone()
            self._touch(item_id)
            return True

        self._maybe_evict()

        # Initialize from CLIP if available (Section 14: content-based cold start)
        if initial_clip is not None:
            if initial_clip.shape[0] > self.embed_dim:
                init_emb = initial_clip[:self.embed_dim].clone()
            else:
                init_emb = F.pad(initial_clip, (0, self.embed_dim - initial_clip.shape[0]))
        else:
            init_emb = embedding.detach().clone()

        self._embeddings[item_id] = init_emb
        return self._cuckoo_insert(_CuckooEntry(item_id=item_id))

    def _cuckoo_insert(self, entry: _CuckooEntry) -> bool:
        current = entry
        for _ in range(self.max_kicks):
            # Try table A
            ha = self._hash_a(current.item_id)
            existing = self._table_a.get(ha)
            if existing is None:
                self._table_a[ha] = current
                self._touch(current.item_id)
                return True
            self._table_a[ha] = current
            current = existing

            # Try table B
            hb = self._hash_b(current.item_id)
            existing = self._table_b.get(hb)
            if existing is None:
                self._table_b[hb] = current
                self._touch(current.item_id)
                return True
            self._table_b[hb] = current
            current = existing

        self._evict_lru()
        return self._cuckoo_insert(entry)

    def _touch(self, item_id: str) -> None:
        self._access_order[item_id] = time.time()
        self._access_order.move_to_end(item_id)

    def _maybe_evict(self) -> None:
        now = time.time()
        to_remove = []
        for item_id, last_access in self._access_order.items():
            if now - last_access > self.max_eviction_age_s:
                to_remove.append(item_id)
            else:
                break
        for item_id in to_remove:
            self._remove(item_id)
        if to_remove:
            logger.info("Evicted %d inactive items (>%dd)", len(to_remove), self.max_eviction_age_s // 86400)

    def _evict_lru(self) -> None:
        if self._access_order:
            self._remove(next(iter(self._access_order)))

    def _remove(self, item_id: str) -> None:
        ha, hb = self._hash_a(item_id), self._hash_b(item_id)
        if ha in self._table_a and self._table_a[ha].item_id == item_id:
            del self._table_a[ha]
        if hb in self._table_b and self._table_b[hb].item_id == item_id:
            del self._table_b[hb]
        self._embeddings.pop(item_id, None)
        self._access_order.pop(item_id, None)

    def mark_inactive(self, item_id: str) -> None:
        """Out-of-stock product → immediate eviction."""
        self._remove(item_id)
        logger.debug("Item %s marked inactive, removed from Cuckoo table", item_id)

    def update_embedding(self, item_id: str, gradient: torch.Tensor, lr: float = 0.01) -> None:
        """SGD step on a single item embedding (streaming update)."""
        emb = self.get(item_id)
        if emb is not None:
            with torch.no_grad():
                emb -= lr * gradient
            self._embeddings[item_id] = emb

    @property
    def size(self) -> int:
        return len(self._embeddings)

    def export_all(self) -> dict[str, list[float]]:
        """Export all embeddings for Redis sync."""
        return {k: v.tolist() for k, v in self._embeddings.items()}
