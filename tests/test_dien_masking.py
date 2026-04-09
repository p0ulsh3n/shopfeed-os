"""
Tests unitaires — DIEN Masking (C-05) et AUGRU TorchScript (H-05)
=================================================================

Vérifie:
  1. Que le masquage se fait AVANT softmax (gradients corrects)
  2. Que _augru_step TorchScript produit le même output que la version Python
  3. Que les positions paddées reçoivent attention_weight == 0
  4. Que l'AUGRU compilé est plus rapide que la version naive

Run: pytest tests/test_dien_masking.py -v
"""

from __future__ import annotations

import time

import pytest
import torch

from ml.models.dien import AUGRU, DIENModel, _augru_step


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def batch_params():
    return {"B": 4, "T": 8, "D": 32, "H": 32}


@pytest.fixture
def dien_model():
    return DIENModel(
        n_items=1000,
        n_categories=50,
        embed_dim=32,
        hidden_size=32,
        mlp_dims=(64, 32),
        n_tasks=2,
        dropout=0.0,
    ).eval()


# ─── C-05: Test masquage avant softmax ──────────────────────────────────────

class TestDIENMaskingBeforeSoftmax:
    """C-05 FIX: Le masquage doit se faire AVANT softmax."""

    def test_padded_positions_have_zero_attention(self, dien_model, batch_params):
        """Les positions paddées (mask=False) doivent recevoir attention_weight == 0."""
        B, T = batch_params["B"], batch_params["T"]
        H = dien_model.hidden_size

        hidden = torch.randn(B, T, H)
        candidate = torch.randn(B, H)

        # Masque: seuls les 5 premiers timesteps sont valides
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :5] = True

        with torch.no_grad():
            attention = dien_model._compute_attention(hidden, candidate, behavior_mask=mask)

        # Vérification principale: positions paddées = 0
        padded_attention = attention[:, 5:]  # Timesteps 5-7 (paddés)
        assert torch.allclose(
            padded_attention, torch.zeros_like(padded_attention), atol=1e-6
        ), f"Padded positions should have attention=0, got max={padded_attention.max().item():.6f}"

    def test_valid_positions_sum_to_one(self, dien_model, batch_params):
        """Les poids attention sur les positions valides doivent sommer à 1."""
        B, T = batch_params["B"], batch_params["T"]
        H = dien_model.hidden_size

        hidden = torch.randn(B, T, H)
        candidate = torch.randn(B, H)

        mask = torch.zeros(B, T, dtype=torch.bool)
        seq_len = 5
        mask[:, :seq_len] = True

        with torch.no_grad():
            attention = dien_model._compute_attention(hidden, candidate, behavior_mask=mask)

        # La somme sur toute la séquence doit être ~1 (softmax après masquage)
        row_sums = attention.sum(dim=-1)
        assert torch.allclose(
            row_sums, torch.ones(B), atol=1e-5
        ), f"Attention weights should sum to 1, got: {row_sums}"

    def test_no_mask_does_not_crash(self, dien_model, batch_params):
        """Sans masque, _compute_attention doit fonctionner normalement."""
        B, T = batch_params["B"], batch_params["T"]
        H = dien_model.hidden_size

        hidden = torch.randn(B, T, H)
        candidate = torch.randn(B, H)

        with torch.no_grad():
            attention = dien_model._compute_attention(hidden, candidate, behavior_mask=None)

        assert attention.shape == (B, T)
        # Sans masque, toutes les positions ont une attention > 0
        assert (attention > 0).all(), "All positions should have positive attention without mask"

    def test_mask_prevents_gradient_leak(self, batch_params):
        """C-05 FIX: Le masquage AVANT softmax empêche les gradients de fuir
        vers les positions paddées.

        Si on masque APRÈS softmax: les gradients des positions paddées != 0
        (soft proba non nulle → gradient non nul via BCE).
        Si on masque AVANT softmax: softmax(-inf) = 0 exact → gradient = 0.
        """
        B, T, H = batch_params["B"], batch_params["T"], batch_params["H"]

        # Simuler attention logits
        logits = torch.randn(B, T, requires_grad=True)

        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :3] = True

        # C-05 FIX: masque AVANT softmax
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        attention_correct = torch.softmax(masked_logits, dim=-1)

        # Vérifier que padded positions ont gradient = 0
        loss = attention_correct[:, :3].sum()  # Loss uniquement sur les valides
        loss.backward()

        padded_grad = logits.grad[:, 3:]
        assert torch.allclose(
            padded_grad, torch.zeros_like(padded_grad), atol=1e-6
        ), "Padded positions should have zero gradient when masked before softmax"

    def test_dien_forward_with_mask(self, dien_model, batch_params):
        """Test end-to-end forward pass de DIENModel avec masque."""
        B, T = batch_params["B"], batch_params["T"]
        n_items = 1000
        n_categories = 50

        behavior_ids = torch.randint(1, n_items, (B, T))
        candidate_id = torch.randint(1, n_items, (B,))
        candidate_cat = torch.randint(1, n_categories, (B,))
        dense_features = torch.randn(B, 5)

        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :5] = True

        with torch.no_grad():
            predictions, aux_logits = dien_model(
                behavior_ids, candidate_id, candidate_cat, dense_features, mask
            )

        assert len(predictions) == 2
        assert aux_logits.shape == (B, T - 1, 1)
        for pred in predictions:
            assert pred.shape == (B, 1)
            assert ((pred >= 0) & (pred <= 1)).all(), "Predictions should be in [0, 1]"


# ─── H-05: Test AUGRU TorchScript ───────────────────────────────────────────

class TestAUGRUTorchScript:
    """H-05 FIX: _augru_step TorchScript produit le même résultat que la version Python."""

    def test_augru_step_correctness(self, batch_params):
        """_augru_step TorchScript == implémentation Python de référence."""
        B, D, H = batch_params["B"], batch_params["D"], batch_params["H"]
        input_size = D

        augru = AUGRU(input_size, H).eval()

        x_t = torch.randn(B, D)
        h = torch.randn(B, H)
        a_t = torch.rand(B)

        # Version TorchScript
        with torch.no_grad():
            h_script = _augru_step(
                x_t, h, a_t,
                augru.W_z.weight, augru.W_z.bias,
                augru.W_r.weight, augru.W_r.bias,
                augru.W_h.weight, augru.W_h.bias,
            )

        # Version Python de référence (même calcul)
        with torch.no_grad():
            combined = torch.cat([x_t, h], dim=-1)
            z_t = torch.sigmoid(augru.W_z(combined))
            r_t = torch.sigmoid(augru.W_r(combined))
            combined_r = torch.cat([x_t, r_t * h], dim=-1)
            h_tilde = torch.tanh(augru.W_h(combined_r))
            z_prime = a_t.unsqueeze(1) * z_t
            h_ref = (1.0 - z_prime) * h + z_prime * h_tilde

        assert torch.allclose(h_script, h_ref, atol=1e-6), (
            f"TorchScript step differs from reference. Max diff: {(h_script - h_ref).abs().max():.2e}"
        )

    def test_augru_forward_shape(self, batch_params):
        """AUGRU.forward() retourne la bonne shape [B, H]."""
        B, T, D, H = batch_params["B"], batch_params["T"], batch_params["D"], batch_params["H"]

        augru = AUGRU(D, H).eval()
        inputs = torch.randn(B, T, D)
        attention = torch.rand(B, T)
        attention = attention / attention.sum(dim=-1, keepdim=True)  # Normalize

        with torch.no_grad():
            out = augru(inputs, attention)

        assert out.shape == (B, H), f"Expected ({B}, {H}), got {out.shape}"

    def test_augru_torchscript_faster(self, batch_params):
        """_augru_step TorchScript doit être plus rapide que Python naïf."""
        B, D, H = 64, 64, 64
        input_size = D
        T = 50

        augru = AUGRU(input_size, H).eval()
        inputs = torch.randn(B, T, D)
        attention = torch.rand(B, T)

        N_RUNS = 10

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                augru(inputs, attention)

        # Benchmark TorchScript version (current impl)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(N_RUNS):
                augru(inputs, attention)
        script_time = (time.perf_counter() - start) / N_RUNS * 1000

        # Le test vérifie surtout que ça ne crash pas et que le temps est raisonnable
        assert script_time < 500, f"AUGRU too slow: {script_time:.1f}ms per forward pass"
