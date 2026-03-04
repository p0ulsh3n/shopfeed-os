"""
DIN — Deep Interest Network (Alibaba, KDD 2018)
=================================================
Marketplace ranking model. Unlike fixed user representations, DIN computes
an *attention-weighted* user interest vector that changes for every candidate
item. This captures "local activation" — only the relevant past behaviors
are activated for each candidate.

Paper: Zhou et al., "Deep Interest Network for Click-Through Rate Prediction"
       KDD 2018, deployed on Taobao main traffic.

Architecture:
  1. Behavior sequence of item embeddings  [B, T, D]
  2. Candidate item embedding               [B, D]
  3. Local Activation Unit computes attention weights per behavior
  4. Weighted-sum → user interest vector     [B, D]
  5. Concatenate with other features → MLP → P(click)

Usage in ShopFeed OS:
  - Marketplace CTR prediction (static product pages)
  - Input: user behavior sequence + candidate product features
  - Output: P(click), P(add_to_cart), P(purchase)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Local Activation Unit ──────────────────────────────────────────
class LocalActivationUnit(nn.Module):
    """Attention mechanism from DIN paper.

    For each (behavior_i, candidate) pair, compute an attention weight
    using the element-wise difference and product as interaction features.
    """

    def __init__(self, embed_dim: int, hidden_units: tuple[int, ...] = (64, 32)):
        super().__init__()
        # Input: [behavior, candidate, behavior-candidate, behavior*candidate]
        input_dim = embed_dim * 4
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_units:
            layers.extend([nn.Linear(prev, h), nn.PReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))  # scalar attention weight
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        behavior: torch.Tensor,   # [B, T, D]
        candidate: torch.Tensor,  # [B, D]
        mask: torch.Tensor | None = None,  # [B, T] bool, True=valid
    ) -> torch.Tensor:
        """Returns attention-weighted user interest vector [B, D]."""
        B, T, D = behavior.shape
        # Expand candidate to match behavior sequence length
        cand_exp = candidate.unsqueeze(1).expand_as(behavior)  # [B, T, D]

        # Interaction features (DIN paper Section 3.2)
        diff = behavior - cand_exp
        prod = behavior * cand_exp
        att_input = torch.cat([behavior, cand_exp, diff, prod], dim=-1)  # [B, T, 4D]

        # Attention weights
        weights = self.mlp(att_input).squeeze(-1)  # [B, T]

        # Mask padding positions
        if mask is not None:
            weights = weights.masked_fill(~mask, float("-inf"))

        weights = F.softmax(weights, dim=-1)  # [B, T]

        # Weighted sum of behavior embeddings
        user_interest = torch.bmm(weights.unsqueeze(1), behavior).squeeze(1)  # [B, D]
        return user_interest


# ─── DIN Model ──────────────────────────────────────────────────────
class DINModel(nn.Module):
    """Deep Interest Network for marketplace CTR/CVR prediction.

    Args:
        n_items:       Total number of items (products + videos)
        n_categories:  Number of product categories
        embed_dim:     Embedding dimension for items
        mlp_dims:      Hidden layer sizes for the prediction MLP
        n_tasks:       Number of prediction tasks (default 3: click, cart, purchase)
        dropout:       Dropout rate
    """

    def __init__(
        self,
        n_items: int,
        n_categories: int,
        embed_dim: int = 64,
        mlp_dims: tuple[int, ...] = (256, 128, 64),
        n_tasks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_tasks = n_tasks

        # Embedding tables
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(n_categories, embed_dim // 2, padding_idx=0)

        # Local Activation Unit (core DIN innovation)
        self.attention = LocalActivationUnit(embed_dim)

        # Dense feature projection (price, freshness, cv_score, stock, seller_weight)
        self.dense_proj = nn.Linear(5, 32)

        # Final MLP: user_interest(D) + candidate(D) + category(D/2) + dense(32)
        mlp_input = embed_dim + embed_dim + (embed_dim // 2) + 32
        layers: list[nn.Module] = []
        prev = mlp_input
        for h in mlp_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.PReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.mlp = nn.Sequential(*layers)

        # Multi-task heads: P(click), P(add_to_cart), P(purchase)
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
        behavior_ids: torch.Tensor,     # [B, T]  past item IDs (padded with 0)
        candidate_id: torch.Tensor,     # [B]     candidate item ID
        candidate_cat: torch.Tensor,    # [B]     candidate category ID
        dense_features: torch.Tensor,   # [B, 5]  price_norm, freshness, cv_score, stock, seller_weight
        behavior_mask: torch.Tensor | None = None,  # [B, T] bool
    ) -> list[torch.Tensor]:
        """Forward pass. Returns list of [B, 1] predictions per task."""
        # Embed behavior sequence and candidate
        behavior_emb = self.item_embedding(behavior_ids)   # [B, T, D]
        candidate_emb = self.item_embedding(candidate_id)  # [B, D]
        category_emb = self.category_embedding(candidate_cat)  # [B, D/2]

        # Auto-generate mask from padding if not provided
        if behavior_mask is None:
            behavior_mask = behavior_ids != 0  # [B, T]

        # DIN attention: compute user interest relative to this candidate
        user_interest = self.attention(behavior_emb, candidate_emb, behavior_mask)  # [B, D]

        # Dense features
        dense_out = self.dense_proj(dense_features)  # [B, 32]

        # Concatenate all features
        combined = torch.cat([user_interest, candidate_emb, category_emb, dense_out], dim=-1)

        # Shared MLP
        hidden = self.mlp(combined)

        # Multi-task predictions
        outputs = [torch.sigmoid(head(hidden)) for head in self.task_heads]
        return outputs


# ─── Loss Function ──────────────────────────────────────────────────
class DINLoss(nn.Module):
    """Multi-task binary cross-entropy with task weights.

    Weights reflect business value:
      P(click)=1.0, P(add_to_cart)=3.0, P(purchase)=5.0
    """

    def __init__(self, task_weights: tuple[float, ...] = (1.0, 3.0, 5.0)):
        super().__init__()
        self.task_weights = task_weights
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        predictions: list[torch.Tensor],  # list of [B, 1]
        labels: list[torch.Tensor],       # list of [B, 1]
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=predictions[0].device)
        for pred, label, w in zip(predictions, labels, self.task_weights):
            task_loss = self.bce(pred, label).mean()
            total_loss = total_loss + w * task_loss
        return total_loss
