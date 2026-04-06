"""
SIM — Search-based Interest Modeling (Alibaba 2020)
Pour les utilisateurs avec de longs historiques (1000+ interactions).
Hard Search + Soft SDIM module pour long-term interest modeling.

Architecture:
  Hard Search : item similarity → retrieve top-K items de l'historique long-terme
  Soft Search : learned SDIM module → soft attention sur l'historique
  Input: [target_item, long_term_sequence(30j), short_term(BST)]
  Utilisé pour: utilisateurs très actifs, marketplace long-term interests
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDIMModule(nn.Module):
    """
    Soft Deep Interest Matching (SDIM) — composant de recherche soft.
    Apprend un espace latent pour retriever les items pertinents de l'historique long.
    """

    def __init__(self, item_dim: int = 64, num_hash_buckets: int = 1024, hash_layers: int = 3):
        super().__init__()
        self.item_dim = item_dim
        self.num_hash_buckets = num_hash_buckets
        self.hash_layers = hash_layers

        # Hash embeddings pour soft search
        self.hash_embeddings = nn.ModuleList([
            nn.Embedding(num_hash_buckets, item_dim // 4)
            for _ in range(hash_layers)
        ])

        # Projection finale
        self.proj = nn.Linear(item_dim, item_dim)

    def forward(self, item_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        item_ids: [batch, seq_len] — IDs des items historique long-terme
        embeddings: [batch, seq_len, item_dim] — pre-computed item embeddings
        Returns: [batch, item_dim] — représentation compressée de l'historique
        """
        # Random hash bucketing pour retrieval efficace
        batch, seq = item_ids.shape
        hash_outputs = []
        for h_layer in self.hash_embeddings:
            bucket_ids = item_ids % self.num_hash_buckets
            h_emb = h_layer(bucket_ids)  # [batch, seq, item_dim//4]
            hash_outputs.append(h_emb.mean(dim=1))  # [batch, item_dim//4]

        hash_repr = torch.cat(hash_outputs, dim=-1)  # [batch, item_dim]

        # Residual connection with actual embeddings for richer representation
        # Hash alone loses fine-grained item semantics; embeddings preserve them
        emb_pool = embeddings.mean(dim=1)  # [batch, item_dim]
        combined = hash_repr + emb_pool  # residual fusion

        return self.proj(combined)


class SIMAttentionUnit(nn.Module):
    """
    Unité d'attention pour scorer la pertinence de chaque item historique
    par rapport à l'item cible (target item).
    """

    def __init__(self, item_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(item_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        target: torch.Tensor,        # [batch, item_dim]
        history: torch.Tensor,       # [batch, seq, item_dim]
    ) -> torch.Tensor:
        """Returns attention weights [batch, seq]"""
        target_expanded = target.unsqueeze(1).expand_as(history)  # [batch, seq, dim]
        interaction = history * target_expanded                     # element-wise
        concat = torch.cat([target_expanded, history, interaction], dim=-1)  # [batch, seq, 3*dim]
        scores = self.fc(concat).squeeze(-1)                        # [batch, seq]
        return F.softmax(scores, dim=-1)


class SIMModel(nn.Module):
    """
    Search-based Interest Model (SIM) — Alibaba 2020.
    Combine un historique long-terme (jusqu'à 1000 items) et court-terme (BST).

    Args:
        num_items:       taille du vocabulaire items
        item_dim:        dimension embedding items (défaut: 64)
        user_dim:        dimension Features utilisateur (défaut: 64)
        short_seq_len:   longueur de la séquence court-terme (défaut: 50)
        long_seq_len:    longueur max de l'historique long-terme (défaut: 1000)
        top_k_hard:      nombre d'items retenus par hard search (défaut: 50)
    """

    def __init__(
        self,
        num_items: int,
        item_dim: int = 64,
        user_dim: int = 64,
        short_seq_len: int = 50,
        long_seq_len: int = 1000,
        top_k_hard: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_dim = item_dim
        self.top_k_hard = top_k_hard
        self.short_seq_len = short_seq_len
        self.long_seq_len = long_seq_len

        # Item embeddings partagés
        self.item_embedding = nn.Embedding(num_items + 1, item_dim, padding_idx=0)

        # Target item projection
        self.target_proj = nn.Linear(item_dim, item_dim)

        # Hard Search: attention unit pour retrieval de l'historique long
        self.hard_attention = SIMAttentionUnit(item_dim)

        # Soft Search: SDIM module
        self.sdim = SDIMModule(item_dim)

        # BST pour le court-terme (séquence récente des 50 interactions)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=item_dim,
            nhead=4,
            dim_feedforward=item_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.bst_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion et prédiction finale
        fusion_input_dim = item_dim * 3 + user_dim  # hard + soft + bst + user
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def hard_search(
        self,
        target_emb: torch.Tensor,     # [batch, item_dim]
        long_hist_ids: torch.Tensor,  # [batch, long_seq]
    ) -> torch.Tensor:
        """
        Hard search: retriever les top_k_hard items de l'historique long-terme
        les plus similaires à l'item cible.
        Returns: [batch, item_dim] — représentation agrégée top-K
        """
        long_hist_emb = self.item_embedding(long_hist_ids)  # [batch, long_seq, item_dim]
        attn_weights = self.hard_attention(target_emb, long_hist_emb)  # [batch, long_seq]

        # Top-K sur les poids
        top_k = min(self.top_k_hard, long_hist_ids.size(1))
        top_weights, top_indices = torch.topk(attn_weights, top_k, dim=-1)
        top_weights = F.softmax(top_weights, dim=-1)

        # Gather les embeddings top-K
        top_indices_exp = top_indices.unsqueeze(-1).expand(-1, -1, self.item_dim)
        top_embs = torch.gather(long_hist_emb, 1, top_indices_exp)  # [B, top_k, dim]

        # Weighted pooling
        weighted = (top_weights.unsqueeze(-1) * top_embs).sum(dim=1)  # [batch, dim]
        return weighted

    def forward(
        self,
        user_features: torch.Tensor,      # [batch, user_dim]
        target_item_id: torch.Tensor,     # [batch]
        short_hist_ids: torch.Tensor,     # [batch, short_seq]
        long_hist_ids: torch.Tensor,      # [batch, long_seq]
    ) -> torch.Tensor:
        """
        Returns: [batch] — logit prédiction (CTR / CVR)
        """
        # Target embedding
        target_emb = self.target_proj(self.item_embedding(target_item_id))  # [B, dim]

        # Hard Search sur historique long
        hard_repr = self.hard_search(target_emb, long_hist_ids)             # [B, dim]

        # Soft Search via SDIM
        soft_repr = self.sdim(long_hist_ids, self.item_embedding(long_hist_ids))  # [B, dim]

        # BST sur historique court-terme
        short_emb = self.item_embedding(short_hist_ids)                     # [B, short, dim]
        bst_out = self.bst_encoder(short_emb)                               # [B, short, dim]
        bst_repr = bst_out.mean(dim=1)                                      # [B, dim]

        # Fusion
        combined = torch.cat([user_features, hard_repr, soft_repr, bst_repr], dim=-1)
        logit = self.fusion(combined).squeeze(-1)                           # [B]
        return logit

    def predict_proba(
        self,
        user_features: torch.Tensor,
        target_item_id: torch.Tensor,
        short_hist_ids: torch.Tensor,
        long_hist_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Returns proba in [0,1]"""
        with torch.no_grad():
            logit = self.forward(user_features, target_item_id, short_hist_ids, long_hist_ids)
            return torch.sigmoid(logit)
