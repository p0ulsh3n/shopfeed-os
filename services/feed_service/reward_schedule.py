"""
Variable-Ratio Reward Scheduler (t.md §4)
============================================
Implements TikTok-style variable-ratio reinforcement in the feed.

The distribution:
    80% baseline — decent content, maintains session
    15% wow      — unexpectedly good match, dopamine spike
     5% destiny  — "c'est exactement ce que je voulais" — massive spike

The UNPREDICTABILITY is the key: rewards arrive at variable intervals,
keeping anticipatory dopamine high (like a slot machine, but with
personalized content instead of symbols).

This module is ADDITIVE: it assigns tiers and reorders the final feed
AFTER the existing re-ranking. It does not modify any scores.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RewardScheduler:
    """Variable-ratio reward scheduler for feed pacing.

    Assigns a reward tier (baseline / wow / destiny) to each candidate
    based on their affinity score relative to the user's personal
    score distribution. Then schedules "wow" and "destiny" items
    at unpredictable positions in the feed.

    Usage:
        scheduler = RewardScheduler()
        tiered = scheduler.assign_tiers(candidates, scores)
        paced  = scheduler.schedule_feed_pacing(tiered, session_id, limit=15)
    """

    # Target tier distribution
    DESTINY_PERCENTILE = 0.95   # Top 5% of scores → destiny
    WOW_PERCENTILE = 0.80       # 80th–95th percentile → wow
    # Below 80th → baseline

    def assign_tiers(
        self,
        candidates: list[Any],
        score_fn=None,
    ) -> list[tuple[Any, str]]:
        """Assign reward tiers based on score distribution.

        Args:
            candidates: list of FeedCandidate objects
            score_fn: callable that extracts score from candidate
                      (defaults to .final_score property)

        Returns:
            List of (candidate, tier) tuples, same order as input.
        """
        if not candidates:
            return []

        if score_fn is None:
            score_fn = lambda c: getattr(c, "final_score", 0.0)

        scores = [score_fn(c) for c in candidates]
        sorted_scores = sorted(scores)
        n = len(sorted_scores)

        if n == 0:
            return [(c, "baseline") for c in candidates]

        # Compute percentile thresholds
        destiny_idx = max(0, int(n * self.DESTINY_PERCENTILE) - 1)
        wow_idx = max(0, int(n * self.WOW_PERCENTILE) - 1)

        destiny_threshold = sorted_scores[destiny_idx]
        wow_threshold = sorted_scores[wow_idx]

        result = []
        for c, s in zip(candidates, scores):
            if s >= destiny_threshold and destiny_threshold > 0:
                tier = "destiny"
            elif s >= wow_threshold and wow_threshold > 0:
                tier = "wow"
            else:
                tier = "baseline"
            result.append((c, tier))

        return result

    def schedule_feed_pacing(
        self,
        tiered_candidates: list[tuple[Any, str]],
        session_id: str = "",
        limit: int = 15,
    ) -> list[Any]:
        """Reorder candidates to place wow/destiny items at variable positions.

        The key principle: wow/destiny items should NOT appear at predictable
        intervals. Instead, they're placed at pseudo-random positions that
        vary per session (but are reproducible for the same session).

        Rules:
            - First 2 items are always baseline (build expectation)
            - Wow items placed at positions determined by session hash
            - Destiny item (if any) placed between positions 4-10
            - No two wow+ items adjacent (preserve surprise)

        Args:
            tiered_candidates: output of assign_tiers()
            session_id: used as seed for position randomization
            limit: max items to return

        Returns:
            List of candidates in paced order (tiers assigned to .reward_tier)
        """
        if not tiered_candidates:
            return []

        # Separate by tier
        baseline = [c for c, t in tiered_candidates if t == "baseline"]
        wow = [c for c, t in tiered_candidates if t == "wow"]
        destiny = [c for c, t in tiered_candidates if t == "destiny"]

        # Generate pseudo-random insertion positions from session_id
        seed = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)

        # Build the paced feed
        result: list[Any] = []
        wow_positions = self._generate_wow_positions(seed, limit, len(wow))
        destiny_pos = self._generate_destiny_position(seed, limit)

        baseline_iter = iter(baseline)
        wow_iter = iter(wow)
        destiny_iter = iter(destiny)

        for pos in range(limit):
            candidate = None

            if pos == destiny_pos:
                candidate = next(destiny_iter, None)
                if candidate:
                    if hasattr(candidate, "reward_tier"):
                        candidate.reward_tier = "destiny"

            if candidate is None and pos in wow_positions:
                candidate = next(wow_iter, None)
                if candidate:
                    if hasattr(candidate, "reward_tier"):
                        candidate.reward_tier = "wow"

            if candidate is None:
                candidate = next(baseline_iter, None)
                if candidate and hasattr(candidate, "reward_tier"):
                    candidate.reward_tier = "baseline"

            if candidate is not None:
                result.append(candidate)

        return result

    @staticmethod
    def _generate_wow_positions(seed: int, limit: int, n_wow: int) -> set[int]:
        """Generate pseudo-random positions for wow items.

        Avoids first 2 and last position. No adjacent wow items.
        """
        positions: set[int] = set()
        rng = seed
        attempts = 0
        while len(positions) < n_wow and attempts < 50:
            rng = (rng * 6364136223846793005 + 1) & 0xFFFFFFFF  # LCG
            pos = 2 + (rng % max(limit - 3, 1))  # skip first 2
            # No adjacent
            if pos not in positions and (pos - 1) not in positions and (pos + 1) not in positions:
                positions.add(pos)
            attempts += 1
        return positions

    @staticmethod
    def _generate_destiny_position(seed: int, limit: int) -> int:
        """Destiny item lands between positions 4–10 (the 'golden zone')."""
        golden_start = min(4, limit - 1)
        golden_end = min(10, limit - 1)
        if golden_end <= golden_start:
            return golden_start
        return golden_start + (seed % (golden_end - golden_start + 1))
