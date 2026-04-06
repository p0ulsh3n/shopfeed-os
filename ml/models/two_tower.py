"""Two-Tower Retrieval Model — Section 04 (Étape 1 · Retrieval · ~10ms).

Architecture: Two independent encoder towers that learn to map users and items
into a shared embedding space. At inference, user embedding is computed once,
then FAISS ANN finds the nearest item embeddings in <10ms.

This is the FIRST stage of the recommendation pipeline. It reduces
10M+ catalog items down to ~2,000 candidates for the ranking stage.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes user features → dense embedding.

    Input features (concatenated):
        - category_prefs:        N_CATEGORIES floats
        - price_sensitivity:     1 float (normalized)
        - purchase_frequency:    1 float (log-normalized)
        - persona_embedding:     4 floats (one-hot or learned)
        - geo_cluster:           1 float (normalized)
        - active_categories:     N_CATEGORIES binary mask
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized user embedding."""
        return F.normalize(self.encoder(x), p=2, dim=-1)


class ItemTower(nn.Module):
    """Encodes item features → dense embedding.

    Input features (concatenated):
        - clip_embedding:        512 floats (frozen CLIP visual)
        - text_embedding:        768 floats (frozen sentence-transformer)
        - price_normalized:      1 float (log-norm relative to category)
        - category_onehot:       N_CATEGORIES
        - cv_score:              1 float
        - freshness_score:       1 float (exp decay)
        - vendor_embedding:      64 floats (learned)
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        hidden_dims: tuple[int, ...] = (1024, 512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized item embedding."""
        return F.normalize(self.encoder(x), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Two-Tower retrieval model with in-batch negative sampling.

    Training: Contrastive loss with in-batch negatives (sampled softmax).
    Inference: Pre-compute all item embeddings → build FAISS index.
               At query time, encode user → ANN search in <10ms.

    Reference: YouTube DNN (Covington 2016), adapted for commerce.
    """

    def __init__(
        self,
        user_input_dim: int,
        item_input_dim: int,
        embedding_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(user_input_dim, embedding_dim)
        self.item_tower = ItemTower(item_input_dim, embedding_dim)
        self.temperature = temperature
        self.embedding_dim = embedding_dim

    def forward(
        self,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute user and item embeddings.

        Args:
            user_features: (batch_size, user_input_dim)
            item_features: (batch_size, item_input_dim)

        Returns:
            user_emb: (batch_size, embedding_dim) — L2-normalized
            item_emb: (batch_size, embedding_dim) — L2-normalized
        """
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        return user_emb, item_emb

    def compute_loss(
        self,
        user_features: torch.Tensor,
        pos_item_features: torch.Tensor,
        neg_item_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sampled softmax contrastive loss with in-batch negatives.

        If neg_item_features is None, uses in-batch negatives (every other
        item in the batch serves as a negative for each user). This is
        efficient and well-proven at scale (YouTube, Pinterest, Alibaba).

        Args:
            user_features:      (B, user_dim)
            pos_item_features:  (B, item_dim) — positive (interacted) items
            neg_item_features:  (B, K, item_dim) — optional explicit negatives

        Returns:
            Scalar loss
        """
        user_emb = self.user_tower(user_features)       # (B, D)
        pos_emb = self.item_tower(pos_item_features)     # (B, D)

        if neg_item_features is not None:
            # Explicit negatives: (B, K, item_dim) → (B, K, D)
            B, K, _ = neg_item_features.shape
            neg_flat = neg_item_features.reshape(B * K, -1)
            neg_emb = self.item_tower(neg_flat).reshape(B, K, -1)

            # Positive scores: (B, 1)
            pos_scores = (user_emb * pos_emb).sum(dim=-1, keepdim=True) / self.temperature

            # Negative scores: (B, K)
            neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1) / self.temperature

            # Log-softmax over [pos, neg1, ..., negK]
            logits = torch.cat([pos_scores, neg_scores], dim=-1)  # (B, 1+K)
            labels = torch.zeros(B, dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, labels)

        else:
            # In-batch negatives: all items in batch serve as negatives
            # Similarity matrix: (B, B)
            logits = torch.mm(user_emb, pos_emb.t()) / self.temperature
            labels = torch.arange(logits.size(0), device=logits.device)
            return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_users(self, user_features: torch.Tensor) -> torch.Tensor:
        """Batch encode users for offline computation."""
        self.eval()
        return self.user_tower(user_features)

    @torch.no_grad()
    def encode_items(self, item_features: torch.Tensor) -> torch.Tensor:
        """Batch encode items for FAISS index building."""
        self.eval()
        return self.item_tower(item_features)
