"""
Neural Ad Fatigue — beyond frequency caps
============================================
Simple frequency capping (show ad max N times) is what Facebook did in 2015.
EPSILON models fatigue as a continuous attention decay function:

    fatigue(user, ad, t) = 1 - σ(
        frequency_penalty +
        semantic_saturation +
        category_overload +
        recency_decay
    )

Where σ is sigmoid normalization to [0, 1].

Components:
    1. Frequency penalty: impressions of this exact creative to this user
    2. Semantic saturation: impressions of visually similar creatives (via CLIP)
    3. Category overload: total ad impressions in this category this session
    4. Recency decay: how long since last impression of this ad

Recovery modeling:
    After T hours without seeing the ad, fatigue recovers:
    recovery = 1 - exp(-time_since_last / recovery_half_life)
    Default recovery_half_life = 24h (creative feels fresh after ~3 days)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FatigueState:
    """Fatigue state for a (user, ad_creative) pair."""
    user_id: str
    creative_id: str
    impression_count: int = 0
    last_impression_ts: float = 0.0
    # Computed
    fatigue_score: float = 0.0     # 0 = fresh, 1 = fully fatigued
    freshness: float = 1.0         # = 1 - fatigue_score
    should_suppress: bool = False  # True if too fatigued to show


class AdFatigueManager:
    """Neural ad fatigue manager with recovery modeling.

    Usage:
        fatigue = AdFatigueManager()

        # Before serving an ad, check fatigue
        state = fatigue.get_fatigue(user_id, creative_id, ad_meta)
        if state.should_suppress:
            skip_this_ad()

        # After serving, record impression
        fatigue.record_impression(user_id, creative_id)
    """

    def __init__(
        self,
        max_impressions_per_user: int = 8,      # Hard cap per creative per user
        session_category_cap: int = 3,           # Max ads per category per session
        recovery_half_life_hours: float = 24.0,  # Creative recovery time
        suppression_threshold: float = 0.75,     # Suppress if fatigue > 0.75
        redis_client: Any = None,
    ):
        self.max_impressions = max_impressions_per_user
        self.session_category_cap = session_category_cap
        self.recovery_half_life = recovery_half_life_hours * 3600  # Convert to seconds
        self.suppression_threshold = suppression_threshold
        self.redis = redis_client

    def get_fatigue(
        self,
        user_id: str,
        creative_id: str,
        ad_meta: dict[str, Any] | None = None,
        session_ad_categories: dict[int, int] | None = None,
        similar_creatives_shown: int = 0,
    ) -> FatigueState:
        """Calculate fatigue state for a (user, creative) pair.

        Args:
            user_id: User identifier
            creative_id: Ad creative identifier
            ad_meta: Ad metadata (category_id, etc.)
            session_ad_categories: {category_id: count} of ads shown this session
            similar_creatives_shown: Number of visually similar ads shown recently

        Returns:
            FatigueState with fatigue_score and should_suppress decision
        """
        state = FatigueState(user_id=user_id, creative_id=creative_id)

        # Load impression history from Redis
        history = self._load_history(user_id, creative_id)
        state.impression_count = history.get("count", 0)
        state.last_impression_ts = history.get("last_ts", 0.0)

        # Hard cap: always suppress if over max impressions
        if state.impression_count >= self.max_impressions:
            state.fatigue_score = 1.0
            state.freshness = 0.0
            state.should_suppress = True
            return state

        # Component 1: Frequency penalty (diminishing returns)
        # Each additional impression has less marginal impact
        freq_penalty = 1.0 - math.exp(-state.impression_count / 3.0)

        # Component 2: Semantic saturation
        # Similar creatives contribute to fatigue even if this exact one is new
        semantic_sat = min(1.0, similar_creatives_shown * 0.15)

        # Component 3: Category overload
        category_overload = 0.0
        if session_ad_categories and ad_meta:
            cat_id = ad_meta.get("category_id", -1)
            cat_count = session_ad_categories.get(cat_id, 0)
            if cat_count >= self.session_category_cap:
                category_overload = 1.0
            else:
                category_overload = cat_count / self.session_category_cap

        # Component 4: Recency decay (how long since last impression)
        recency_penalty = 0.0
        if state.last_impression_ts > 0:
            time_since = time.time() - state.last_impression_ts
            # Recovery: fatigue decreases over time
            recovery = 1.0 - math.exp(-time_since / max(self.recovery_half_life, 1))
            recency_penalty = (1.0 - recovery) * freq_penalty

        # Weighted combination
        raw_fatigue = (
            0.40 * freq_penalty +
            0.20 * semantic_sat +
            0.25 * category_overload +
            0.15 * recency_penalty
        )

        # Sigmoid normalization to [0, 1]
        state.fatigue_score = 1.0 / (1.0 + math.exp(-6 * (raw_fatigue - 0.5)))
        state.freshness = 1.0 - state.fatigue_score
        state.should_suppress = state.fatigue_score > self.suppression_threshold

        return state

    def record_impression(self, user_id: str, creative_id: str) -> None:
        """Record that a user saw a creative."""
        key = f"epsilon:fatigue:{user_id}:{creative_id}"
        now = time.time()

        if self.redis:
            try:
                pipe = self.redis.pipeline()
                pipe.hincrby(key, "count", 1)
                pipe.hset(key, "last_ts", str(now))
                # Auto-expire after 30 days (no need to track ancient impressions)
                pipe.expire(key, 30 * 24 * 3600)
                pipe.execute()
            except Exception as e:
                logger.warning("Failed to record ad impression: %s", e)
        else:
            logger.debug("No Redis — fatigue impression not persisted")

    def record_session_ad(
        self,
        session_categories: dict[int, int],
        category_id: int,
    ) -> dict[int, int]:
        """Track ad category counts within a session."""
        session_categories[category_id] = session_categories.get(category_id, 0) + 1
        return session_categories

    def _load_history(self, user_id: str, creative_id: str) -> dict:
        """Load impression history from Redis."""
        if not self.redis:
            return {"count": 0, "last_ts": 0.0}

        key = f"epsilon:fatigue:{user_id}:{creative_id}"
        try:
            data = self.redis.hgetall(key)
            if data:
                return {
                    "count": int(data.get(b"count", data.get("count", 0))),
                    "last_ts": float(data.get(b"last_ts", data.get("last_ts", 0.0))),
                }
        except Exception:
            pass
        return {"count": 0, "last_ts": 0.0}

    def apply_fatigue_discount(
        self,
        ad_scores: list,
        user_id: str,
        session_ad_categories: dict[int, int] | None = None,
    ) -> list:
        """Apply fatigue discount to a list of AdScore objects.

        Modifies each ad's eCPM based on fatigue state.
        Fully fatigued ads get eCPM = 0 (effectively removed).
        """
        for ad_score in ad_scores:
            state = self.get_fatigue(
                user_id=user_id,
                creative_id=ad_score.ad_id,
                ad_meta={"category_id": getattr(ad_score, "category_id", -1)},
                session_ad_categories=session_ad_categories,
            )
            if state.should_suppress:
                ad_score.ecpm = 0.0
            else:
                # Reduce eCPM proportionally to fatigue
                ad_score.ecpm *= state.freshness

        # Remove suppressed ads
        return [a for a in ad_scores if a.ecpm > 0]
