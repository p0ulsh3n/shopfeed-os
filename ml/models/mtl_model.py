"""Multi-Task Learning (MTL) Model — Section 02 / 04.

Architecture: Progressive Layered Extraction (PLE) with 7 simultaneous tasks.
This is the CORE scoring model that predicts ALL commercial signals at once.

Tasks (Section 02 — Scoring Commerce):
    1. P(buy_now)       × 12  — instant purchase
    2. P(purchase)      × 10  — cart purchase
    3. P(add_to_cart)   ×  8  — cart addition
    4. P(save_wishlist) ×  6  — wishlist save
    5. P(share)         ×  5  — social share
    6. E(watch_time)    ×  3  — expected watch time (regression)
    7. P(negative)      × -8  — skip <1s or "not interested"

The final commerce score is the weighted sum:
    Score_Contenu = Σ (task_weight × task_prediction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Task weights from Section 02 (Scoring Commerce formula)
# ──────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "buy_now":       {"weight": 12.0, "type": "binary"},
    "purchase":      {"weight": 10.0, "type": "binary"},
    "add_to_cart":   {"weight":  8.0, "type": "binary"},
    "save_wishlist": {"weight":  6.0, "type": "binary"},
    "share":         {"weight":  5.0, "type": "binary"},
    "watch_time":    {"weight":  3.0, "type": "regression"},
    "negative":      {"weight": -8.0, "type": "binary"},
}

TASK_NAMES = list(TASK_CONFIGS.keys())
NUM_TASKS = len(TASK_NAMES)


class ExpertNetwork(nn.Module):
    """Single expert network — shared or task-specific."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatingNetwork(nn.Module):
    """Gating network for expert selection — learns which experts matter
    for each task dynamically based on the input.
    """

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, num_experts) soft attention weights."""
        return self.gate(x)


class PLELayer(nn.Module):
    """Single PLE extraction layer.

    Contains:
        - `num_shared` shared expert networks (used by ALL tasks)
        - `num_task_specific` task-specific experts PER task
        - 1 gating network PER task that selects from all available experts
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_tasks: int = NUM_TASKS,
        num_shared_experts: int = 4,
        num_task_experts: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_shared = num_shared_experts
        self.num_task = num_task_experts
        total_experts_per_task = num_shared_experts + num_task_experts

        # Shared experts
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, dropout)
            for _ in range(num_shared_experts)
        ])

        # Task-specific experts: task_experts[task_idx][expert_idx]
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertNetwork(input_dim, hidden_dim, dropout)
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])

        # Gating per task: selects from (shared + task_specific) experts
        self.gates = nn.ModuleList([
            GatingNetwork(input_dim, total_experts_per_task)
            for _ in range(num_tasks)
        ])

        self.output_dim = hidden_dim

    def forward(self, task_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            task_inputs: list of NUM_TASKS tensors, each (B, input_dim)
                         For the first layer, all inputs are the same shared input.

        Returns:
            list of NUM_TASKS tensors, each (B, hidden_dim)
        """
        # Compute shared expert outputs (computed once, reused)
        shared_outputs = [expert(task_inputs[0]) for expert in self.shared_experts]

        task_outputs = []
        for task_idx in range(self.num_tasks):
            # Task-specific expert outputs
            task_specific = [
                expert(task_inputs[task_idx])
                for expert in self.task_experts[task_idx]
            ]

            # Stack all expert outputs: (B, total_experts, hidden_dim)
            all_expert_outputs = torch.stack(shared_outputs + task_specific, dim=1)

            # Gating weights: (B, total_experts)
            gate_weights = self.gates[task_idx](task_inputs[task_idx])

            # Weighted sum: (B, hidden_dim)
            gated_output = (all_expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=1)
            task_outputs.append(gated_output)

        return task_outputs


class MTLModel(nn.Module):
    """Multi-Task Learning model with PLE architecture.

    The full architecture:
        Input features
            → PLE Layer 1 (shared + task-specific experts with gating)
            → PLE Layer 2 (deeper extraction)
            → Task-specific towers (one per task)
            → 7 task heads (6 binary + 1 regression)

    Commerce Score (Section 02):
        Score = P(buy_now)×12 + P(purchase)×10 + P(add_to_cart)×8
              + P(save)×6 + P(share)×5 + E(watch_time)×3 - P(negative)×8
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_shared_experts: int = 4,
        num_task_experts: int = 2,
        num_ple_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Stacked PLE layers
        self.ple_layers = nn.ModuleList()
        for i in range(num_ple_layers):
            layer_input_dim = hidden_dim
            self.ple_layers.append(
                PLELayer(
                    input_dim=layer_input_dim,
                    hidden_dim=hidden_dim,
                    num_tasks=NUM_TASKS,
                    num_shared_experts=num_shared_experts,
                    num_task_experts=num_task_experts,
                    dropout=dropout,
                )
            )

        # Task-specific towers: final per-task refinement
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(),
            )
            for _ in range(NUM_TASKS)
        ])

        # Task heads
        self.task_heads = nn.ModuleDict()
        for name, config in TASK_CONFIGS.items():
            self.task_heads[name] = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        # M-01 FIX: Kaiming uniform (He init) est conçu pour ReLU/GELU.
        # Les task heads (output sigmoid) doivent utiliser Xavier/Glorot
        # qui suppose une activation symétrique (Glorot & Bengio, 2010).
        # Appliquer Kaiming sur les couches sigmoid fausse les gradients initiaux.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Vérifier si c'est un task head (output dim=1) → Xavier
                # Les task heads ont out_features=1 et sont dans self.task_heads
                is_task_head = any(
                    module is head for head in self.task_heads.values()
                )
                if is_task_head:
                    # Xavier/Glorot uniform: pour les couches de sortie sigmoid
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                else:
                    # Kaiming (He) pour les couches intermédiaires ReLU/GELU
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            features: (B, input_dim) — concatenated user+item+context features

        Returns:
            dict mapping task_name → (B,) predictions
        """
        # Project input
        h = self.input_proj(features)  # (B, hidden_dim)

        # Initial PLE input: same features shared across all tasks
        task_inputs = [h] * NUM_TASKS

        # Pass through stacked PLE layers
        for ple_layer in self.ple_layers:
            task_inputs = ple_layer(task_inputs)

        # Task towers + heads
        predictions: dict[str, torch.Tensor] = {}
        for i, task_name in enumerate(TASK_NAMES):
            tower_out = self.task_towers[i](task_inputs[i])  # (B, 64)
            raw_logit = self.task_heads[task_name](tower_out).squeeze(-1)  # (B,)

            config = TASK_CONFIGS[task_name]
            if config["type"] == "binary":
                predictions[task_name] = torch.sigmoid(raw_logit)
            else:
                # Regression: ReLU to ensure non-negative watch time
                predictions[task_name] = F.relu(raw_logit)

        return predictions

    def compute_commerce_score(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Section 02 — Commerce Score formula.

        Score = Σ (task_weight × task_prediction)
        Negative task has negative weight → automatically subtracted.
        """
        score = torch.zeros_like(predictions[TASK_NAMES[0]])
        for task_name, config in TASK_CONFIGS.items():
            score = score + config["weight"] * predictions[task_name]
        return score

    def compute_loss(
        self,
        features: torch.Tensor,
        labels: dict[str, torch.Tensor],
        task_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Multi-task loss — weighted sum of per-task losses.

        Args:
            features: (B, input_dim)
            labels:   dict mapping task_name → (B,) ground truth
            task_weights: optional loss weights per task (for curriculum learning)

        Returns:
            (total_loss, per_task_losses_dict)
        """
        predictions = self.forward(features)
        per_task_losses: dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=features.device)

        for task_name in TASK_NAMES:
            if task_name not in labels:
                continue

            config = TASK_CONFIGS[task_name]
            pred = predictions[task_name]
            target = labels[task_name].float()

            if config["type"] == "binary":
                task_loss = F.binary_cross_entropy(pred, target, reduction="mean")
            else:
                # Regression (watch_time): Huber loss for robustness to outliers
                task_loss = F.huber_loss(pred, target, reduction="mean", delta=30.0)

            per_task_losses[task_name] = task_loss

            # Apply curriculum or uniform weighting
            w = task_weights.get(task_name, 1.0) if task_weights else 1.0
            total_loss = total_loss + w * task_loss

        return total_loss, per_task_losses
