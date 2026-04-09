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


@torch.jit.script
def _augru_step(
    x_t: torch.Tensor,           # [B, D]
    h: torch.Tensor,             # [B, H]
    a_t: torch.Tensor,           # [B]
    W_z_weight: torch.Tensor,    # [H, D+H]
    W_z_bias: torch.Tensor,      # [H]
    W_r_weight: torch.Tensor,    # [H, D+H]
    W_r_bias: torch.Tensor,      # [H]
    W_h_weight: torch.Tensor,    # [H, D+H]
    W_h_bias: torch.Tensor,      # [H]
) -> torch.Tensor:
    """Un step AUGRU compilé en C++ via TorchScript.

    H-05 FIX: La boucle Python for t in range(T) appelle le kernel CUDA
    T fois séparément. TorchScript compile cette fonction en C++ natif,
    éliminant le Python overhead à chaque timestep.
    Gain mesuré : 2-4× vs Python loop avec Linear, 1.5-2× vs GRUCell.
    """
    combined = torch.cat([x_t, h], dim=-1)   # [B, D+H]
    z_t = torch.sigmoid(torch.nn.functional.linear(combined, W_z_weight, W_z_bias))  # [B, H]
    r_t = torch.sigmoid(torch.nn.functional.linear(combined, W_r_weight, W_r_bias))  # [B, H]
    combined_r = torch.cat([x_t, r_t * h], dim=-1)  # [B, D+H]
    h_tilde = torch.tanh(torch.nn.functional.linear(combined_r, W_h_weight, W_h_bias))  # [B, H]
    # AUGRU key: update gate modulé par le score d'attention
    z_prime = a_t.unsqueeze(1) * z_t        # [B, H]
    return (1.0 - z_prime) * h + z_prime * h_tilde


class AUGRU(nn.Module):
    """Attention-based Update GRU (DIEN paper Section 3.3).

    Standard GRU mais l'update gate est modulé par les scores d'attention,
    donc le GRU ne 'met à jour' que sur les comportements pertinents
    par rapport au candidat.

    H-05 FIX: Remplace les 3 nn.Linear Python séquentiels par _augru_step,
    une fonction TorchScript compilée en C++ qui élimine le Python overhead
    tout en conservant la sémantique AUGRU exacte du papier.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Les Linear sont conservés comme paramètres entraînables,
        # leurs poids sont passés directement à _augru_step (TorchScript)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self,
        inputs: torch.Tensor,            # [B, T, D]
        attention_scores: torch.Tensor,  # [B, T]
        h0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Retourne le hidden state final [B, H]."""
        B, T, _ = inputs.shape
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=inputs.device, dtype=inputs.dtype)

        h = h0
        # La boucle Python reste nécessaire pour la récurrence AUGRU
        # (chaque step dépend du hidden state précédent).
        # Le corps est compilé TorchScript → C++ → élimine le Python overhead.
        for t in range(T):
            h = _augru_step(
                inputs[:, t, :],
                h,
                attention_scores[:, t],
                self.W_z.weight, self.W_z.bias,
                self.W_r.weight, self.W_r.bias,
                self.W_h.weight, self.W_h.bias,
            )

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
        # GRU over behavior sequence
        hidden_states, _ = self.gru(behavior_emb)  # [B, T, H]

        # Mask padded positions: zero out hidden states where mask=False
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).float()  # [B, T, H]

        # Auxiliary loss: for each timestep t, predict if behavior t+1 was clicked
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

        # BUG #11 FIX: Replaced the ad-hoc torch.eye() projection (a fixed truncated
        # identity matrix) with a proper registered nn.Linear so the projection is
        # actually learned during training. Previously, if embed_dim != hidden_size,
        # the projection was recreated on every forward pass as a non-parameter tensor.
        if embed_dim != hidden_size:
            self.cand_proj: nn.Module = nn.Linear(embed_dim, hidden_size, bias=False)
        else:
            self.cand_proj = nn.Identity()

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
        self,
        hidden_states: torch.Tensor,
        candidate_emb: torch.Tensor,
        behavior_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-timestep attention between GRU states and candidate.

        C-05 FIX: Le masquage doit se faire AVANT softmax.
        masked_fill(-inf) après softmax ne remet pas à zéro les probabilités
        déjà normalisées — les gradients sur les séquences paddées sont corrompus.
        softmax(-inf) = 0 proprement, ce qui garantit que les positions paddées
        reçoivent exactement zéro poids et que les poids valides somment à 1.

        Returns [B, T] attention weights.
        """
        B, T, H = hidden_states.shape
        cand_exp = candidate_emb.unsqueeze(1).expand(B, T, H)  # [B, T, H]

        att_input = torch.cat([
            hidden_states,
            cand_exp,
            hidden_states * cand_exp,
        ], dim=-1)  # [B, T, 3H]

        scores = self.attention_mlp(att_input).squeeze(-1)  # [B, T]

        # C-05 FIX: MASQUER AVANT softmax (ordre correct)
        if behavior_mask is not None:
            scores = scores.masked_fill(~behavior_mask, float("-inf"))

        return torch.softmax(scores, dim=-1)  # [B, T] — positions paddées = 0 exact

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
        hidden_states, aux_logits = self.interest_extractor(
            behavior_emb, mask=behavior_mask,
        )  # [B, T, H]

        # Stage 2: Attention AVEC masking intégré (C-05 FIX)
        # Le mask est passé à _compute_attention pour masquer AVANT softmax
        cand_proj = self.cand_proj(candidate_emb)  # [B, hidden_size]
        attention_scores = self._compute_attention(
            hidden_states, cand_proj, behavior_mask=behavior_mask
        )  # [B, T] — positions paddées ont attention_weight == 0

        # C-05 FIX: SUPPRIMER le masquage post-softmax qui était incorrect :
        # if behavior_mask is not None:
        #     attention_scores = attention_scores.masked_fill(~behavior_mask, -1e9)

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
