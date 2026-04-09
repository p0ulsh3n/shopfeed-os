"""DeepFM Ranking Model — Section 04 (Étape 3 · Ranking · ~40ms).

Architecture: DeepFM combines a factorization machine (FM) component
for explicit feature interactions with a deep neural network for
implicit high-order interactions. This dual approach captures both
memorization (FM) and generalization (DNN).

This is the SECOND stage: takes ~400 candidates from pre-ranking
and predicts fine-grained P(click), P(purchase), etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMLayer(nn.Module):
    """Factorization Machine layer — captures 2nd-order feature interactions.

    Standard FM computes:  sum_{i<j} <v_i, v_j> * x_i * x_j
    Optimized via:  0.5 * (sum(V*X)^2 - sum(V^2 * X^2))
    """

    def __init__(self, num_features: int, embedding_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, feature_indices: torch.Tensor, feature_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_indices: (B, F) — indices of active features
            feature_values:  (B, F) — values of those features

        Returns:
            (B, 1) — FM interaction output
        """
        # (B, F, E)
        embeddings = self.embedding(feature_indices)
        # Weight by feature values: (B, F, E) * (B, F, 1)
        weighted = embeddings * feature_values.unsqueeze(-1)

        # O(n) optimization for pairwise interactions
        sum_of_square = weighted.sum(dim=1).pow(2)       # (B, E)
        square_of_sum = weighted.pow(2).sum(dim=1)       # (B, E)

        interaction = 0.5 * (sum_of_square - square_of_sum).sum(dim=-1, keepdim=True)
        return interaction


class DNNLayer(nn.Module):
    """Deep Neural Network component — learns high-order feature interactions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
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
        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DeepFM(nn.Module):
    """DeepFM model combining FM + DNN for CTR prediction.

    Architecture:
        Input → [FM Component] → concat → Output (sigmoid)
              → [DNN Component]

    The dense features (CLIP embedding, text embedding, price, etc.)
    go through the DNN. The sparse feature indices (category, vendor_id, etc.)
    go through the FM for interaction modeling.
    """

    def __init__(
        self,
        num_sparse_features: int,
        dense_input_dim: int,
        fm_embedding_dim: int = 16,
        dnn_hidden_dims: tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
        # H-07 FIX: n_sparse_fields était hardcodé à 20 dans le DNN input dim.
        # Si on avait 18 ou 25 features sparse, le modèle acceptait quand même
        # et produisait des shapes incorrectes silencieusement.
        # On paramétrise et on valide dans forward().
        n_sparse_fields: int = 20,
    ):
        super().__init__()
        self.n_sparse_fields = n_sparse_fields
        # First-order linear weights for sparse features
        self.linear = nn.Embedding(num_sparse_features, 1)

        # FM component — second-order interactions
        self.fm = FMLayer(num_sparse_features, fm_embedding_dim)

        # Dense feature projection
        self.dense_proj = nn.Linear(dense_input_dim, 128)

        # DNN component — concatenated sparse embeddings + dense
        self.sparse_embedding = nn.Embedding(num_sparse_features, fm_embedding_dim)
        # H-07 FIX: dnn_input_dim dépend du n_sparse_fields paramétrisé, pas 20 hardcodé
        dnn_input_dim = 128 + fm_embedding_dim * n_sparse_fields
        self.dnn = DNNLayer(dnn_input_dim, dnn_hidden_dims, dropout)

        # Final prediction head
        self.head = nn.Linear(1 + self.dnn.output_dim + 1, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        sparse_indices: torch.Tensor,
        sparse_values: torch.Tensor,
        dense_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            sparse_indices: (B, F_sparse) — feature index IDs
            sparse_values:  (B, F_sparse) — feature values (1.0 for binary)
            dense_features: (B, D_dense)  — continuous features

        Returns:
            (B,) — click probability [0, 1]
        """
        # First-order: linear weights
        linear_out = (self.linear(sparse_indices).squeeze(-1) * sparse_values).sum(
            dim=-1, keepdim=True
        )  # (B, 1)

        # FM: second-order interactions
        fm_out = self.fm(sparse_indices, sparse_values)  # (B, 1)

        # DNN: high-order interactions
        sparse_emb = self.sparse_embedding(sparse_indices)  # (B, F, E)
        B = sparse_emb.size(0)
        n = self.n_sparse_fields  # H-07 FIX: utilise le paramètre, pas 20 hardcodé

        # H-07 FIX: Valider que le batch a assez de features sparse
        actual_fields = sparse_emb.size(1)
        if actual_fields < n:
            raise ValueError(
                f"DeepFM expected at least {n} sparse fields, got {actual_fields}. "
                f"Rebuild the model with n_sparse_fields={actual_fields} or "
                f"pad sparse_indices to {n} fields."
            )

        sparse_flat = sparse_emb[:, :n, :].reshape(B, -1)  # (B, n*E)
        dense_proj = self.dense_proj(dense_features)          # (B, 128)
        dnn_input = torch.cat([dense_proj, sparse_flat], dim=-1)
        dnn_out = self.dnn(dnn_input)  # (B, last_hidden)

        # Combine all three
        combined = torch.cat([linear_out, fm_out, dnn_out], dim=-1)
        logit = self.head(combined).squeeze(-1)  # (B,)

        return torch.sigmoid(logit)

    def compute_loss(
        self,
        sparse_indices: torch.Tensor,
        sparse_values: torch.Tensor,
        dense_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy loss for CTR prediction.

        Args:
            labels: (B,) — 0 or 1 (did user click?)
        """
        preds = self.forward(sparse_indices, sparse_values, dense_features)
        return F.binary_cross_entropy(preds, labels.float())
