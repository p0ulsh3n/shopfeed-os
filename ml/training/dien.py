"""
DIEN — Deep Interest Evolution Network (Alibaba, AAAI 2019)
=============================================================
Extension of DIN that models *temporal evolution* of user interests.
DIN treats behaviors as a set; DIEN treats them as a sequence.

Key innovations over DIN:
  1. Interest Extractor: GRU with auxiliary loss to extract interest states
  2. Interest Evolution: AUGRU (Attention Update GRU) to model how interests
     evolve over time relative to the candidate item

Paper: Zhou et al., "Deep Interest Evolution Network"
       AAAI 2019, +20.7% CTR on Taobao

Architecture:
  Behavior Sequence → GRU (interest extractor) → AUGRU (interest evolution)
  → Attention scores weighted by candidate → evolved interest → MLP → P(click)

Usage in ShopFeed OS:
  - Marketplace ranking for users with evolving tastes (fashion, seasonal)
  - Captures "interest drift" — sneakers 6 months ago → formal shoes now
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── AUGRU — Attention Update GRU ───────────────────────────────────
class AUGRU(nn.Module):
    """Attention-based Update GRU (DIEN paper Section 3.3).

    Standard GRU but the update gate is modulated by attention scores,
    so the GRU only "updates" when processing behaviors relevant to
    the candidate item.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # GRU gates
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # update gate
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)  # reset gate
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)  # candidate hidden

    def forward(
        self,
        inputs: torch.Tensor,         # [B, T, D]
        attention_scores: torch.Tensor,  # [B, T]  per-timestep attention from candidate
        h0: torch.Tensor | None = None,  # [B, H]
    ) -> torch.Tensor:
        """Returns final hidden state representing evolved interest [B, H]."""
        B, T, D = inputs.shape
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=inputs.device)

        h = h0
        for t in range(T):
            x_t = inputs[:, t, :]           # [B, D]
            a_t = attention_scores[:, t].unsqueeze(1)  # [B, 1]

            combined = torch.cat([x_t, h], dim=-1)  # [B, D+H]

            z_t = torch.sigmoid(self.W_z(combined))  # update gate [B, H]
            r_t = torch.sigmoid(self.W_r(combined))  # reset gate [B, H]

            combined_r = torch.cat([x_t, r_t * h], dim=-1)
            h_tilde = torch.tanh(self.W_h(combined_r))  # candidate [B, H]

            # AUGRU key: modulate update gate by attention score
            z_prime = a_t * z_t  # attention-weighted update [B, H]

            h = (1 - z_prime) * h + z_prime * h_tilde

        return h  # [B, H] — final evolved interest state


# ─── Interest Extractor (GRU with Auxiliary Loss) ───────────────────
class InterestExtractor(nn.Module):
    """GRU that extracts interest states from behavior sequence.

    An auxiliary loss supervises the GRU: predict the next behavior embedding
    from the current hidden state. This ensures hidden states capture
    meaningful interest representations, not just sequence info.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # Auxiliary loss: predict next item embedding from hidden state
        self.auxiliary_net = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, 1),  # binary: is next item clicked?
        )

    def forward(
        self,
        behavior_emb: torch.Tensor,  # [B, T, D]
        mask: torch.Tensor | None = None,  # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (hidden_states [B, T, H], auxiliary_logits [B, T-1, 1])."""
        # Pack padded sequence for efficient GRU
        hidden_states, _ = self.gru(behavior_emb)  # [B, T, H]

        # Auxiliary loss: for each timestep t, predict if behavior t+1 was clicked
        # Uses hidden state at t to predict about t+1
        aux_logits = self.auxiliary_net(hidden_states[:, :-1, :])  # [B, T-1, 1]

        return hidden_states, aux_logits


# ─── DIEN Model ─────────────────────────────────────────────────────
class DIENModel(nn.Module):
    """Deep Interest Evolution Network for marketplace CTR/CVR.

    Args:
        n_items:       Total number of items
        n_categories:  Number of categories
        embed_dim:     Item embedding dimension
        hidden_size:   GRU hidden size
        mlp_dims:      MLP hidden layers
        n_tasks:       Number of prediction tasks (click, cart, purchase)
        dropout:       Dropout rate
    """

    def __init__(
        self,
        n_items: int,
        n_categories: int,
        embed_dim: int = 64,
        hidden_size: int = 64,
        mlp_dims: tuple[int, ...] = (256, 128, 64),
        n_tasks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        # Embedding tables
        self.item_embedding = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(n_categories, embed_dim // 2, padding_idx=0)

        # Interest Extractor (GRU + auxiliary loss)
        self.interest_extractor = InterestExtractor(embed_dim, hidden_size)

        # Attention for AUGRU: compute attention between each hidden state and candidate
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, 64),  # [h_t, candidate, h_t*candidate]
            nn.PReLU(),
            nn.Linear(64, 1),
        )

        # Interest Evolution (AUGRU)
        self.augru = AUGRU(hidden_size, hidden_size)

        # Dense feature projection
        self.dense_proj = nn.Linear(5, 32)

        # Final MLP
        mlp_input = hidden_size + embed_dim + (embed_dim // 2) + 32
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

    def _compute_attention(
        self, hidden_states: torch.Tensor, candidate_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-timestep attention between GRU states and candidate.
        Returns [B, T] attention scores."""
        B, T, H = hidden_states.shape
        cand_exp = candidate_emb.unsqueeze(1).expand(B, T, H)  # [B, T, H]

        att_input = torch.cat([
            hidden_states,
            cand_exp,
            hidden_states * cand_exp,
        ], dim=-1)  # [B, T, 3H]

        scores = self.attention_mlp(att_input).squeeze(-1)  # [B, T]
        return torch.softmax(scores, dim=-1)

    def forward(
        self,
        behavior_ids: torch.Tensor,    # [B, T]
        candidate_id: torch.Tensor,    # [B]
        candidate_cat: torch.Tensor,   # [B]
        dense_features: torch.Tensor,  # [B, 5]
        behavior_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Returns:
            predictions: list of [B, 1] per task
            aux_logits:  [B, T-1, 1] for auxiliary loss
        """
        # Embeddings
        behavior_emb = self.item_embedding(behavior_ids)   # [B, T, D]
        candidate_emb = self.item_embedding(candidate_id)  # [B, D]
        category_emb = self.category_embedding(candidate_cat)

        # Stage 1: Interest Extractor (GRU + aux loss)
        hidden_states, aux_logits = self.interest_extractor(behavior_emb)  # [B, T, H]

        # Attention scores for AUGRU (candidate-aware per timestep)
        # Project candidate to hidden_size if dimensions differ
        if candidate_emb.shape[-1] != self.hidden_size:
            cand_proj = F.linear(
                candidate_emb,
                torch.eye(self.hidden_size, self.embed_dim, device=candidate_emb.device),
            )
        else:
            cand_proj = candidate_emb

        attention_scores = self._compute_attention(hidden_states, cand_proj)  # [B, T]

        # Stage 2: Interest Evolution (AUGRU)
        evolved_interest = self.augru(hidden_states, attention_scores)  # [B, H]

        # Dense features
        dense_out = self.dense_proj(dense_features)

        # Concatenate and predict
        combined = torch.cat([evolved_interest, candidate_emb, category_emb, dense_out], dim=-1)
        hidden = self.mlp(combined)

        predictions = [torch.sigmoid(head(hidden)) for head in self.task_heads]
        return predictions, aux_logits


# ─── Loss Function ──────────────────────────────────────────────────
class DIENLoss(nn.Module):
    """Multi-task BCE + auxiliary loss for DIEN.

    The auxiliary loss ensures the GRU hidden states capture real interest signals,
    not just sequence information. Without it, DIEN degrades to a standard GRU baseline.
    """

    def __init__(
        self,
        task_weights: tuple[float, ...] = (1.0, 3.0, 5.0),
        aux_weight: float = 0.1,
    ):
        super().__init__()
        self.task_weights = task_weights
        self.aux_weight = aux_weight
        self.bce = nn.BCELoss(reduction="none")

    def forward(
        self,
        predictions: list[torch.Tensor],
        labels: list[torch.Tensor],
        aux_logits: torch.Tensor,        # [B, T-1, 1]
        aux_labels: torch.Tensor | None = None,  # [B, T-1] — was next item clicked?
    ) -> torch.Tensor:
        # Main task losses
        total_loss = torch.tensor(0.0, device=predictions[0].device)
        for pred, label, w in zip(predictions, labels, self.task_weights):
            total_loss = total_loss + w * self.bce(pred, label).mean()

        # Auxiliary loss (predict if next behavior was a positive action)
        if aux_labels is not None:
            aux_pred = torch.sigmoid(aux_logits.squeeze(-1))  # [B, T-1]
            aux_loss = F.binary_cross_entropy(aux_pred, aux_labels.float(), reduction="mean")
            total_loss = total_loss + self.aux_weight * aux_loss

        return total_loss
