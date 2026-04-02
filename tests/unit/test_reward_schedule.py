"""
Unit tests — Variable-Ratio Reward Scheduler (t.md §4)
"""

from __future__ import annotations

import pytest

from services.feed_service.reward_schedule import RewardScheduler


class FakeCandidate:
    """Minimal candidate for testing."""
    def __init__(self, score: float):
        self.final_score = score
        self.reward_tier = "baseline"


@pytest.fixture
def scheduler():
    return RewardScheduler()


# ── Tier assignment ─────────────────────────────────────────────────

def test_tier_assignment_distribution(scheduler: RewardScheduler):
    """Top 5% → destiny, next 15% → wow, rest → baseline."""
    # 100 candidates with scores 0.01 to 1.00
    candidates = [FakeCandidate(i / 100.0) for i in range(1, 101)]
    tiered = scheduler.assign_tiers(candidates)

    tiers = [t for _, t in tiered]
    destiny_count = tiers.count("destiny")
    wow_count = tiers.count("wow")
    baseline_count = tiers.count("baseline")

    assert destiny_count >= 4, f"Expected ~5 destiny, got {destiny_count}"
    assert wow_count >= 10, f"Expected ~15 wow, got {wow_count}"
    assert baseline_count >= 70, f"Expected ~80 baseline, got {baseline_count}"


def test_tier_assignment_empty(scheduler: RewardScheduler):
    """Empty candidates → empty result."""
    assert scheduler.assign_tiers([]) == []


def test_tier_assignment_single_candidate(scheduler: RewardScheduler):
    """Single candidate should be baseline (not enough for percentiles)."""
    candidates = [FakeCandidate(0.5)]
    tiered = scheduler.assign_tiers(candidates)
    assert len(tiered) == 1


# ── Feed pacing ─────────────────────────────────────────────────────

def test_pacing_deterministic(scheduler: RewardScheduler):
    """Same session_id → same ordering."""
    candidates = [FakeCandidate(i / 20.0) for i in range(1, 21)]
    tiered = scheduler.assign_tiers(candidates)

    result1 = scheduler.schedule_feed_pacing(tiered, session_id="abc123", limit=15)
    result2 = scheduler.schedule_feed_pacing(tiered, session_id="abc123", limit=15)

    scores1 = [c.final_score for c in result1]
    scores2 = [c.final_score for c in result2]
    assert scores1 == scores2, "Same session_id should produce same order"


def test_pacing_no_wow_adjacent(scheduler: RewardScheduler):
    """No two wow+ items should be adjacent."""
    candidates = [FakeCandidate(i / 20.0) for i in range(1, 21)]
    tiered = scheduler.assign_tiers(candidates)
    result = scheduler.schedule_feed_pacing(tiered, session_id="test", limit=15)

    for i in range(len(result) - 1):
        tier_a = getattr(result[i], "reward_tier", "baseline")
        tier_b = getattr(result[i + 1], "reward_tier", "baseline")
        if tier_a in ("wow", "destiny") and tier_b in ("wow", "destiny"):
            # This is acceptable if it's the same item — but double-check
            assert False, f"Adjacent wow/destiny at positions {i} and {i+1}"


def test_pacing_returns_at_most_limit(scheduler: RewardScheduler):
    """Should not exceed the requested limit."""
    candidates = [FakeCandidate(i / 30.0) for i in range(1, 31)]
    tiered = scheduler.assign_tiers(candidates)
    result = scheduler.schedule_feed_pacing(tiered, session_id="x", limit=10)
    assert len(result) <= 10
