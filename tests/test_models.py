"""
Tests for ml/training/dien.py and ml/training/bst.py
Covers BUG #11 (DIEN cand_proj) and BUG #15 (BST embed_dim%n_heads guard).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ml.training.dien import DIENModel
from ml.training.bst import BSTModel


# ──────────────────────────────────────────────────────────────
# BUG #11 — DIEN: cand_proj must be a learned nn.Linear
# ──────────────────────────────────────────────────────────────

def test_dien_cand_proj_is_linear_when_dims_differ():
    """BUG #11 FIX: when embed_dim != hidden_size, DIEN must register an
    nn.Linear (not recreate a torch.eye on every forward pass)."""
    model = DIENModel(n_items=100, n_categories=10, embed_dim=32, hidden_size=64)

    # cand_proj must be a proper registered submodule
    assert hasattr(model, "cand_proj"), "DIENModel must have a 'cand_proj' attribute"
    assert isinstance(model.cand_proj, nn.Linear), (
        f"BUG #11 NOT FIXED: cand_proj is {type(model.cand_proj).__name__}, expected nn.Linear "
        "when embed_dim (32) != hidden_size (64)"
    )
    assert model.cand_proj.in_features == 32
    assert model.cand_proj.out_features == 64


def test_dien_cand_proj_is_identity_when_dims_equal():
    """When embed_dim == hidden_size, cand_proj should be nn.Identity (no-op)."""
    model = DIENModel(n_items=100, n_categories=10, embed_dim=64, hidden_size=64)
    assert isinstance(model.cand_proj, nn.Identity), (
        f"Expected nn.Identity when dims match, got {type(model.cand_proj).__name__}"
    )


def test_dien_cand_proj_is_in_state_dict():
    """BUG #11 FIX: learned projection must appear in state_dict so it's saved."""
    model = DIENModel(n_items=100, n_categories=10, embed_dim=32, hidden_size=64)
    state_keys = list(model.state_dict().keys())
    cand_proj_keys = [k for k in state_keys if "cand_proj" in k]
    assert len(cand_proj_keys) > 0, (
        "BUG #11: cand_proj weight not found in state_dict \u2014 "
        "projection will not be saved/loaded with checkpoints"
    )


def test_dien_forward_with_mismatched_dims():
    """End-to-end: DIEN forward pass must not crash when embed_dim != hidden_size."""
    model = DIENModel(n_items=100, n_categories=10, embed_dim=32, hidden_size=64)
    model.eval()

    B, T = 2, 10
    behavior_ids = torch.randint(0, 100, (B, T))
    candidate_id = torch.randint(0, 100, (B,))
    candidate_cat = torch.randint(0, 10, (B,))
    dense = torch.randn(B, 5)

    with torch.no_grad():
        preds, aux_logits = model(behavior_ids, candidate_id, candidate_cat, dense)

    assert len(preds) == 3  # 3 task heads
    assert preds[0].shape == (B, 1)


# ──────────────────────────────────────────────────────────────
# BUG #15 — BST: embed_dim % n_heads must be validated at __init__
# ──────────────────────────────────────────────────────────────

def test_bst_raises_on_bad_n_heads():
    """BUG #15 FIX: BSTModel must raise ValueError when embed_dim % n_heads != 0."""
    # 64 % 5 != 0  →  invalid
    with pytest.raises(ValueError, match="embed_dim"):
        BSTModel(n_items=100, n_categories=10, embed_dim=64, n_heads=5)


def test_bst_raises_gives_valid_options():
    """The error message must list valid n_heads values."""
    with pytest.raises(ValueError) as excinfo:
        BSTModel(n_items=100, n_categories=10, embed_dim=64, n_heads=5)
    # Must mention valid divisors of 64 (1, 2, 4, 8, 16, 32, 64)
    assert "8" in str(excinfo.value) or "4" in str(excinfo.value), (
        "Error message should list valid n_heads options"
    )


def test_bst_valid_n_heads_does_not_raise():
    """64 % 8 == 0 → should work fine."""
    model = BSTModel(n_items=100, n_categories=10, embed_dim=64, n_heads=8)
    assert model.embed_dim == 64


def test_bst_forward_with_valid_config():
    """Sanity: BST forward pass works with valid config."""
    model = BSTModel(n_items=100, n_categories=10, embed_dim=64, n_heads=4, n_layers=1)
    model.eval()

    B, T = 2, 10
    behavior_ids = torch.randint(0, 100, (B, T))
    candidate_id = torch.randint(0, 100, (B,))
    candidate_cat = torch.randint(0, 10, (B,))
    dense = torch.randn(B, 5)

    with torch.no_grad():
        preds = model(behavior_ids, candidate_id, candidate_cat, dense)

    assert len(preds) == 3
    assert preds[0].shape == (B, 1)
