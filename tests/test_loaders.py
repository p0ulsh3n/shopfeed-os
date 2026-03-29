"""
Tests for ml/datasets/loaders.py
Covers BUG #7, #9, #10 fixes.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from ml.datasets.loaders import BehaviorSequenceDataset


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def _make_interactions(n_users: int = 3, events_per_user: int = 6) -> list[dict]:
    """Synthetic interaction list for testing."""
    now = time.time()
    interactions = []
    for u in range(n_users):
        for i in range(events_per_user):
            interactions.append({
                "user_id": f"user_{u}",
                "item_id": (u * 100) + i,
                "category_id": i % 5,
                "behavior_type": "buy" if i == events_per_user - 1 else "pv",
                "timestamp": now - (events_per_user - i) * 3600,  # hourly spacing
                "price": 10.0 + i,
                "cv_score": 0.8,
                "stock": 50,
                "account_weight": 2.5,
            })
    return interactions


# ──────────────────────────────────────────────────────────────
# BUG #7 — dense_features must be non-zero with real values
# ──────────────────────────────────────────────────────────────

def test_dense_features_nonzero():
    """BUG #7 FIX: dense_features must contain real values, not zeros."""
    ds = BehaviorSequenceDataset(_make_interactions(), min_seq_len=1)
    assert len(ds) > 0, "Dataset should not be empty"

    sample = ds[0]
    dense = sample["dense_features"]

    assert dense.shape == (5,), f"Expected 5 dense features, got {dense.shape}"
    assert not torch.all(dense == 0.0), (
        "BUG #7 NOT FIXED: dense_features are all zeros — "
        "real features (price, freshness, cv_score, stock, weight) not computed"
    )

    # Price feature: log1p(10) / 10 ≈ 0.24 — must be > 0
    assert dense[0] > 0.0, "Price dense feature should be > 0 for price=10"
    # Freshness: exp(-age/168) for recent item ≈ 1.0 — must be in (0, 1]
    assert 0.0 < dense[1] <= 1.0, f"Freshness out of range: {dense[1]}"
    # cv_score: 0.8 from fixture
    assert abs(dense[2].item() - 0.8) < 1e-4, f"cv_score mismatch: {dense[2]}"


# ──────────────────────────────────────────────────────────────
# BUG #9 — pos_weight property must reflect true class imbalance
# ──────────────────────────────────────────────────────────────

def test_pos_weight_correct_ratio():
    """BUG #9 FIX: pos_weight must equal n_neg / n_pos."""
    ds = BehaviorSequenceDataset(_make_interactions(n_users=5, events_per_user=6), min_seq_len=1)
    assert len(ds) > 0

    n_pos = sum(1 for i in range(len(ds)) if ds.samples[i]["label"] == 1)
    n_neg = len(ds) - n_pos

    if n_pos == 0:
        pytest.skip("No positive samples generated — adjust fixture")

    expected = n_neg / n_pos
    assert abs(ds.pos_weight - expected) < 1e-6, (
        f"BUG #9: pos_weight={ds.pos_weight:.2f}, expected={expected:.2f}"
    )
    # For our fixture: 1 buy per user (last event) → pos_rate ≈ 16%
    # pos_weight should be > 1 (negatives outweigh positives)
    assert ds.pos_weight > 1.0, "pos_weight should be > 1 with typical class imbalance"


# ──────────────────────────────────────────────────────────────
# BUG #10 — cold-start users (1–4 interactions) must appear in samples
# ──────────────────────────────────────────────────────────────

def test_cold_start_users_included():
    """BUG #10 FIX: users with fewer than 5 interactions must be included."""
    # 2 cold-start users with only 2 interactions each
    cold_interactions = [
        {"user_id": "cold_A", "item_id": 1, "behavior_type": "pv", "timestamp": 1000},
        {"user_id": "cold_A", "item_id": 2, "behavior_type": "buy", "timestamp": 2000},
        {"user_id": "cold_B", "item_id": 3, "behavior_type": "pv", "timestamp": 1000},
        {"user_id": "cold_B", "item_id": 4, "behavior_type": "buy", "timestamp": 2000},
    ]

    # With the OLD code (min_seq_len=5): no samples generated for 2-event users
    ds_old = BehaviorSequenceDataset(cold_interactions, min_seq_len=5)
    assert len(ds_old) == 0, "Old behavior: cold-start users excluded (min_seq_len=5)"

    # With the FIX (min_seq_len=1): at least 1 sample per 2-event user
    ds_fixed = BehaviorSequenceDataset(cold_interactions, min_seq_len=1)
    assert len(ds_fixed) > 0, (
        "BUG #10 NOT FIXED: cold-start users (2 interactions) still excluded"
    )

    # Default constructor must also include cold-start (min_seq_len default=1)
    ds_default = BehaviorSequenceDataset(cold_interactions)
    assert len(ds_default) > 0, (
        "BUG #10 NOT FIXED: default min_seq_len should be 1, not 5"
    )
