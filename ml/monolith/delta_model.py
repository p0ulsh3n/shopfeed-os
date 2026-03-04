"""
Delta Model — learns real-time adjustments to batch scores (Section 14)
========================================================================
Small MLP that learns adjustments from streaming events.
Score_final = Score_batch(V3) + Delta(V2).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeltaModel(nn.Module):
    """Online delta model: Score_final = Score_batch(V3) + Delta(V2).

    Small MLP that learns adjustments from streaming events.
    This captures changes since the last batch training run.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, item_emb: torch.Tensor, user_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([item_emb, user_emb], dim=-1)
        return self.net(combined)
