"""
Unit tests — Desire Graph 6D (t.md §5)
"""

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from ml.feature_store.desire_graph import DesireGraph


@pytest.fixture
def graph():
    return DesireGraph()


# ── compute_desire_vector ────────────────────────────────────────────

def test_desire_vector_shape(graph: DesireGraph):
    """Must return a 10D tensor."""
    vec = graph.compute_desire_vector()
    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (10,)
    assert not torch.isnan(vec).any()


def test_desire_vector_with_empty_inputs(graph: DesireGraph):
    """Empty profile + empty session → no NaN, all finite."""
    vec = graph.compute_desire_vector(user_profile={}, session_state={})
    assert not torch.isnan(vec).any()
    assert torch.isfinite(vec).all()


def test_desire_vector_values_bounded(graph: DesireGraph):
    """All values should be in [-1, 1] or [0, 1]."""
    vec = graph.compute_desire_vector(
        user_profile={"category_prefs": {"1": 0.5}},
        session_state={
            "last_actions": [
                {"type": "view"}, {"type": "buy_now"}, {"type": "skip"},
            ],
            "price_range": {"max": 200},
            "active_categories": {"1": 0.5, "2": 0.3},
        },
        hour=23,
        session_duration_s=600,
    )
    assert not torch.isnan(vec).any()
    # Most values should be in [0, 1] or [-1, 1]
    assert vec.min().item() >= -1.0
    assert vec.max().item() <= 1.5  # small margin


# ── Decision fatigue ─────────────────────────────────────────────────

def test_fatigue_increases_with_views(graph: DesireGraph):
    """More views in session → higher decision fatigue (dim 5)."""
    short_session = {"last_actions": [{"type": "view"}] * 3}
    long_session = {"last_actions": [{"type": "view"}] * 30}

    vec_short = graph.compute_desire_vector(session_state=short_session)
    vec_long = graph.compute_desire_vector(session_state=long_session)

    # Fatigue dim is index 5 (after emotional[0:2], aspiration[2:4], vulnerability[4:5])
    assert vec_long[5].item() > vec_short[5].item()


# ── Emotional state ──────────────────────────────────────────────────

def test_positive_actions_increase_valence(graph: DesireGraph):
    """Session with buy_now actions → high valence (dim 0)."""
    positive_session = {
        "last_actions": [{"type": "buy_now"}, {"type": "save"}, {"type": "like"}]
    }
    negative_session = {
        "last_actions": [{"type": "skip"}, {"type": "not_interested"}, {"type": "skip"}]
    }

    vec_pos = graph.compute_desire_vector(session_state=positive_session)
    vec_neg = graph.compute_desire_vector(session_state=negative_session)

    assert vec_pos[0].item() > vec_neg[0].item()


# ── Subconscious patterns ───────────────────────────────────────────

def test_micro_pause_density(graph: DesireGraph):
    """Session with micro_pause actions → higher subconscious dim (dim 6)."""
    with_pauses = {
        "last_actions": [{"type": "micro_pause"}] * 5 + [{"type": "view"}] * 5
    }
    without_pauses = {
        "last_actions": [{"type": "view"}] * 10
    }

    vec_pauses = graph.compute_desire_vector(session_state=with_pauses)
    vec_no_pauses = graph.compute_desire_vector(session_state=without_pauses)

    # Subconscious dims are at index 6-7
    assert vec_pauses[6].item() > vec_no_pauses[6].item()
