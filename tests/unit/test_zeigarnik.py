"""
Unit tests — Zeigarnik Collection Tracker (t.md §3)
"""

from __future__ import annotations

import asyncio
import pytest

from services.feed_service.zeigarnik import ZeigarnikTracker


@pytest.fixture
def tracker():
    return ZeigarnikTracker(redis_client=None)


# ── compute_zeigarnik_boost ──────────────────────────────────────────

def test_boost_at_zero_completion():
    """0% complete → no boost (no investment = no tension)."""
    assert ZeigarnikTracker.compute_zeigarnik_boost(0.0) == 0.0


def test_boost_at_full_completion():
    """100% complete → no boost (satisfaction, tension released)."""
    assert ZeigarnikTracker.compute_zeigarnik_boost(1.0) == 0.0


def test_boost_peaks_in_tension_zone():
    """40-70% completion should produce the highest boost."""
    boost_40 = ZeigarnikTracker.compute_zeigarnik_boost(0.40)
    boost_55 = ZeigarnikTracker.compute_zeigarnik_boost(0.55)
    boost_70 = ZeigarnikTracker.compute_zeigarnik_boost(0.70)
    boost_10 = ZeigarnikTracker.compute_zeigarnik_boost(0.10)
    boost_90 = ZeigarnikTracker.compute_zeigarnik_boost(0.90)

    # Peak should be higher than edges
    assert boost_55 > boost_10
    assert boost_55 > boost_90
    # Tension zone should all be reasonably high
    assert boost_40 > 0.15
    assert boost_55 > 0.20
    assert boost_70 > 0.15


def test_boost_max_capped_at_03():
    """Max boost should not exceed 0.3."""
    for ratio in [i / 20.0 for i in range(21)]:
        boost = ZeigarnikTracker.compute_zeigarnik_boost(ratio)
        assert boost <= 0.3, f"Boost at {ratio:.2f} = {boost} exceeds 0.3"


def test_boost_symmetric_around_center():
    """Boost at 0.35 and 0.75 should be similar (both ~same distance from center)."""
    boost_35 = ZeigarnikTracker.compute_zeigarnik_boost(0.35)
    boost_75 = ZeigarnikTracker.compute_zeigarnik_boost(0.75)
    assert abs(boost_35 - boost_75) < 0.05


# ── In-memory tracker operations ─────────────────────────────────────

@pytest.mark.asyncio
async def test_record_purchase_in_memory(tracker: ZeigarnikTracker):
    """Recording purchase should track items."""
    await tracker.record_purchase("user1", "coll1", "item_a", collection_total=5)
    ratio = await tracker.get_completion_ratio("user1", "coll1")
    assert ratio == 1 / 5


@pytest.mark.asyncio
async def test_incomplete_collections(tracker: ZeigarnikTracker):
    """get_incomplete_collections returns only partial collections."""
    # Add 2/5 items in collection A
    await tracker.record_purchase("user1", "collA", "i1", 5)
    await tracker.record_purchase("user1", "collA", "i2", 5)

    incomplete = await tracker.get_incomplete_collections("user1")
    assert len(incomplete) == 1
    assert incomplete[0]["collection_id"] == "collA"
    assert incomplete[0]["items_owned"] == 2
    assert incomplete[0]["ratio"] == 2 / 5


@pytest.mark.asyncio
async def test_unknown_user_returns_empty(tracker: ZeigarnikTracker):
    """Unknown user should return ratio 0.0 and no collections."""
    ratio = await tracker.get_completion_ratio("unknown", "what")
    assert ratio == 0.0
