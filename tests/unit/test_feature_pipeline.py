"""
Unit tests — Feature Engineering Pipeline
"""

from __future__ import annotations
import pytest
import torch
import numpy as np
from ml.feature_store.pipeline import (
    product_to_features,
    user_to_features,
    session_to_features,
    N_CATEGORIES,
    VISUAL_EMB_DIM,
    VENDOR_EMB_DIM,
)


# ── product_to_features ────────────────────────────────────────────────────────

def test_product_basic_shape():
    row = {
        "id": "prod-001",
        "clip_embedding": [0.1] * VISUAL_EMB_DIM,
        "base_price": 29.99,
        "category_id": 5,
        "published_at": "2026-02-01T00:00:00+00:00",
        "vendor_id": "vendor-001",
        "cv_score": 0.8,
        "base_stock": 50,
    }
    features = product_to_features(row)
    expected_dim = VISUAL_EMB_DIM + 1 + N_CATEGORIES + 1 + 1 + VENDOR_EMB_DIM + 1
    assert isinstance(features, torch.Tensor)
    assert features.shape == (expected_dim,)
    assert not torch.isnan(features).any()


def test_product_zero_price():
    """base_price=0 ne doit pas générer NaN."""
    row = {"base_price": 0.0, "clip_embedding": None, "category_id": 0}
    features = product_to_features(row)
    assert not torch.isnan(features).any()


def test_product_category_onehot():
    """Vérifie que le category one-hot est correct."""
    cat_id = 10
    row = {
        "clip_embedding": [0.0] * VISUAL_EMB_DIM,
        "base_price": 10.0,
        "category_id": cat_id,
        "published_at": None,
        "vendor_id": None,
        "cv_score": 0.5,
        "base_stock": 10,
    }
    features = product_to_features(row)
    # Category vector starts at index VISUAL_EMB_DIM + 1
    start_idx = VISUAL_EMB_DIM + 1
    cat_vec = features[start_idx: start_idx + N_CATEGORIES]
    assert cat_vec[cat_id].item() == 1.0
    assert cat_vec.sum().item() == 1.0


def test_product_freshness_old():
    """Produit vieux → freshness proche de 0."""
    row = {
        "clip_embedding": None,
        "base_price": 10.0,
        "category_id": 0,
        "published_at": "2020-01-01T00:00:00+00:00",  # très vieux
        "vendor_id": None,
        "cv_score": 0.5,
        "base_stock": 1,
    }
    features = product_to_features(row)
    freshness_idx = VISUAL_EMB_DIM + 1 + N_CATEGORIES
    assert features[freshness_idx].item() < 0.01


# ── user_to_features ───────────────────────────────────────────────────────────

def test_user_basic_shape():
    profile = {
        "category_prefs": {"1": 0.9, "5": 0.5},
        "price_ranges": {"fashion": {"avg": 50}},
        "persona": "active_buyer",
        "country": "CI",
    }
    history = [
        {"item_id": "p1", "action": "purchase", "timestamp": ""},
        {"item_id": "p2", "action": "view", "timestamp": ""},
    ]
    features = user_to_features(profile, history)
    from ml.feature_store.pipeline import PERSONA_LIST
    expected_dim = N_CATEGORIES + 1 + 1 + len(PERSONA_LIST) + 1 + N_CATEGORIES
    assert isinstance(features, torch.Tensor)
    assert features.shape == (expected_dim,)
    assert not torch.isnan(features).any()


def test_user_empty_profile():
    """Profile vide → pas de NaN."""
    features = user_to_features({}, [])
    assert not torch.isnan(features).any()


def test_user_purchase_frequency():
    """Plusieurs achats → log(1+n) > 0."""
    history = [{"action": "purchase"} for _ in range(5)]
    features = user_to_features({}, history)
    freq_idx = N_CATEGORIES + 1
    assert features[freq_idx].item() > 0.0


# ── session_to_features ────────────────────────────────────────────────────────

def test_session_shape():
    actions = [
        {"type": "view", "product_id": "p1", "category": 1, "price": 50.0,
         "dwell_ms": 3000, "watch_pct": 0.5},
        {"type": "add_to_cart", "product_id": "p1", "category": 1, "price": 50.0,
         "dwell_ms": 0, "watch_pct": 0.0},
    ]
    features = session_to_features(actions, max_seq_len=50)
    from ml.feature_store.pipeline import ACTION_TYPES
    expected_feature_dim = len(ACTION_TYPES) + N_CATEGORIES + 4
    assert isinstance(features, torch.Tensor)
    assert features.shape == (50, expected_feature_dim)


def test_session_empty():
    """Session vide → tensor de zéros."""
    features = session_to_features([], max_seq_len=50)
    assert features.sum().item() == 0.0


def test_session_negative_action_weight():
    """Action skip → poids < neutre."""
    actions = [{"type": "skip", "product_id": "p1", "category": 1,
                "price": 10.0, "dwell_ms": 0, "watch_pct": 0.0}]
    features = session_to_features(actions, max_seq_len=10)
    from ml.feature_store.pipeline import ACTION_TYPES
    weight_idx = len(ACTION_TYPES) + N_CATEGORIES + 3
    # skip weight = -2 → normalized = (−2+8)/20 = 0.3 < 0.5 (neutre)
    assert features[0, weight_idx].item() < 0.5
