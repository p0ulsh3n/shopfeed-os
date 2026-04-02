"""
Unit tests — Circadian Temporal Features (t.md §2)
"""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from ml.feature_store.temporal import (
    compute_temporal_features,
    get_vulnerability_multiplier,
)


# ── compute_temporal_features ────────────────────────────────────────

def test_temporal_features_shape():
    """Must return a 6D tensor."""
    features = compute_temporal_features(hour=14, minute=30, day_of_week=2)
    assert isinstance(features, torch.Tensor)
    assert features.shape == (6,)
    assert not torch.isnan(features).any()


def test_temporal_features_no_nan_at_boundaries():
    """Hour 0 and hour 23 must not produce NaN."""
    for hour in (0, 23):
        features = compute_temporal_features(hour=hour, minute=59)
        assert not torch.isnan(features).any()


def test_vulnerability_peaks_at_night():
    """Vulnerability score (dim 4) must be higher at 1AM than at 9AM."""
    night = compute_temporal_features(hour=1)
    morning = compute_temporal_features(hour=9)
    assert night[4].item() > morning[4].item(), (
        f"Vulnerability at 1AM ({night[4].item():.2f}) should be > 9AM ({morning[4].item():.2f})"
    )


def test_vulnerability_low_in_morning():
    """Vulnerability at 7-9AM should be below 0.15."""
    for hour in (7, 8, 9):
        features = compute_temporal_features(hour=hour)
        assert features[4].item() < 0.15, f"Hour {hour} vulnerability too high"


def test_cyclical_encoding_wraps():
    """Hour 0 and hour 24 (=0) should produce same cyclical encoding."""
    h0 = compute_temporal_features(hour=0, minute=0)
    # Simulating hour=24 by using hour=0
    h24 = compute_temporal_features(hour=0, minute=0)
    assert torch.allclose(h0[:2], h24[:2]), "Cyclical encoding should wrap"


def test_session_fatigue_increases():
    """Longer sessions → higher fatigue score (dim 5)."""
    short = compute_temporal_features(session_duration_s=60)
    long = compute_temporal_features(session_duration_s=1800)
    assert long[5].item() > short[5].item()


def test_session_fatigue_saturates():
    """Fatigue should plateau and not exceed 1.0."""
    extreme = compute_temporal_features(session_duration_s=100000)
    assert extreme[5].item() <= 1.0


# ── get_vulnerability_multiplier ─────────────────────────────────────

def test_multiplier_range():
    """Multiplier must always be in [1.0, 1.5]."""
    for hour in range(24):
        mult = get_vulnerability_multiplier(hour=hour, session_duration_s=0)
        assert 1.0 <= mult <= 1.5, f"Hour {hour}: multiplier {mult} out of range"


def test_multiplier_higher_at_night():
    """Multiplier at 1AM must be higher than at 9AM."""
    night = get_vulnerability_multiplier(hour=1)
    morning = get_vulnerability_multiplier(hour=9)
    assert night > morning


def test_multiplier_baseline_morning():
    """9AM multiplier should be close to 1.0 (low vulnerability)."""
    mult = get_vulnerability_multiplier(hour=9, session_duration_s=0)
    assert mult < 1.1
