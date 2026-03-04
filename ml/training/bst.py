"""
BST — Behavior Sequence Transformer (Alibaba, 2019)
=====================================================
Replaces the GRU in DIEN with a Transformer encoder for user behavior
modeling. Multi-head self-attention captures long-range dependencies in
behavioral sequences better than recurrent models.

Paper: Chen et al., "Behavior Sequence Transformer for E-commerce Recommendation"
       ACM DLP Workshop 2019, deployed on Taobao

Key insight:
  A user who browsed: sneakers → socks → shorts → T-shirts
  reveals a *complete sportswear shopping intent* that BST captures
  via full self-attention, whereas GRU struggles with long-range deps.

Architecture:
  1. Behavior item embeddings + positional encoding
  2. Transformer encoder (multi-head self-attention × L layers)
  3. Target-aware attention (like DIN) on transformer outputs
  4. Concatenate with candidate features → MLP → P(click)

Usage in ShopFeed OS:
  - Marketplace ranking for complex browsing patterns
  - Especially effective for premeditated purchase sessions
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Learnable Positional Encoding ──────────────────────────────────
class LearnablePositionalEncoding(nn.Module):
    """Learnable position embeddings for behavior sequences."""

    def __init__(self, max_len: int, embed_dim: int):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → x + pos_emb"""
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        return x + self.position_embedding(positions)


# ─── Transformer Block ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """Standard Transformer encoder block: MultiHead Attention + FFN."""

    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        # Convert padding mask: True=valid → create key_padding_mask where True=ignore
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


# ─── Target Attention ───────────────────────────────────────────────
class TargetAttention(nn.Module):
    """Target-aware attention (DIN-style) on Transformer outputs.

    Computes attention between each transformer output and the candidate
    item to produce a single user interest vector.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        transformer_out: torch.Tensor,  # [B, T, D]
        candidate: torch.Tensor,        # [B, D]
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns [B, D] user interest vector."""
        B, T, D = transformer_out.shape
        cand_exp = candidate.unsqueeze(1).expand_as(transformer_out)

        att_input = torch.cat([
            transformer_out,
            cand_exp,
            transformer_out * cand_exp,
        ], dim=-1)  # [B, T, 3D]

        weights = self.attn(att_input).squeeze(-1)  # [B, T]
        if mask is not None:
            weights = weights.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weights, dim=-1)

        return torch.bmm(weights.unsqueeze(1), transformer_out).squeeze(1)


# ─── BST Model ──────────────────────────────────────────────────────
class BSTModel(nn.Module):
    """Behavior Sequence Transformer for marketplace ranking.

    Args:
        n_items:        Total items in catalog
        n_categories:   Number of categories
        embed_dim:      Embedding dimension
        n_heads:        Number of attention heads
        n_layers:       Number of Transformer layers
        max_seq_len:    Maximum behavior sequence length
        ff_dim:         Feed-forward hidden dimension
        mlp_dims:       Final prediction MLP dimensions
        n_tasks:        Number of prediction tasks
        dropout:        Dropout rate
    """

    def __init__(
        self,
        n_items: int,
        n_categories: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 200,
        ff_dim: int = 256,
        mlp_dims: tuple[int, ...] = (256, 128, 64),
        n_tasks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Embeddings
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(n_categories, embed_dim // 2, padding_idx=0)
        self.positional_encoding = LearnablePositionalEncoding(max_seq_len, embed_dim)

        # Transformer encoder stack
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        # Target attention (DIN-style on transformer outputs)
        self.target_attention = TargetAttention(embed_dim)

        # Dense feature projection
        self.dense_proj = nn.Linear(5, 32)

        # Final MLP
        mlp_input = embed_dim + embed_dim + (embed_dim // 2) + 32
        layers: list[nn.Module] = []
        prev = mlp_input
        for h in mlp_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.mlp = nn.Sequential(*layers)

        # Multi-task heads
        self.task_heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_tasks)])
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(
        self,
        behavior_ids: torch.Tensor,    # [B, T]
        candidate_id: torch.Tensor,    # [B]
        candidate_cat: torch.Tensor,   # [B]
        dense_features: torch.Tensor,  # [B, 5]
        behavior_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Forward pass. Returns list of [B, 1] predictions."""
        # Embeddings
        behavior_emb = self.item_embedding(behavior_ids)   # [B, T, D]
        candidate_emb = self.item_embedding(candidate_id)  # [B, D]
        category_emb = self.category_embedding(candidate_cat)

        # Auto mask from padding
        if behavior_mask is None:
            behavior_mask = behavior_ids != 0  # [B, T]

        # Add positional encoding
        behavior_emb = self.positional_encoding(behavior_emb)

        # Transformer encoder
        x = behavior_emb
        for block in self.transformer_blocks:
            x = block(x, behavior_mask)  # [B, T, D]

        # Target-aware attention → user interest
        user_interest = self.target_attention(x, candidate_emb, behavior_mask)  # [B, D]

        # Dense features
        dense_out = self.dense_proj(dense_features)

        # Predict
        combined = torch.cat([user_interest, candidate_emb, category_emb, dense_out], dim=-1)
        hidden = self.mlp(combined)
        outputs = [torch.sigmoid(head(hidden)) for head in self.task_heads]
        return outputs


# ─── Loss ───────────────────────────────────────────────────────────
class BSTLoss(nn.Module):
    """Multi-task BCE loss for BST. Same interface as DIN/DIEN."""

    def __init__(self, task_weights: tuple[float, ...] = (1.0, 3.0, 5.0)):
        super().__init__()
        self.task_weights = task_weights
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        predictions: list[torch.Tensor],
        labels: list[torch.Tensor],
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=predictions[0].device)
        for pred, label, w in zip(predictions, labels, self.task_weights):
            total_loss = total_loss + w * self.bce(pred, label).mean()
        return total_loss
