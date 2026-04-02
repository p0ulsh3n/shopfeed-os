"""
Audience Targeting — Multi-signal user matching for EPSILON
=============================================================
Combines 5 targeting strategies to find the ideal audience for each ad.
Strategies are unlocked by subscription tier:

    STARTER:  local + behavioral
    GROWTH:   + lookalike + interest
    PREMIUM:  + desire_graph (dopaminergic targeting — unique to EPSILON)

Each strategy produces a (user_id, match_score) and the final audience
is the union of all strategies, scored by the weighted sum.

Store traffic amplification:
    The targeting module specifically optimizes for vendor store visits
    by identifying users most likely to browse the vendor's full catalog
    (not just click the ad). This drives the 2-3× traffic guarantee.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TargetedUser:
    """A user matched by the targeting engine."""
    user_id: str
    match_score: float = 0.0          # Overall match quality (0-1)
    match_reasons: list[str] = field(default_factory=list)
    # Per-strategy scores
    local_score: float = 0.0
    behavioral_score: float = 0.0
    lookalike_score: float = 0.0
    interest_score: float = 0.0
    desire_score: float = 0.0
    # Store visit prediction
    store_visit_probability: float = 0.0


# Strategy weights (tuned via A/B testing)
STRATEGY_WEIGHTS = {
    "local": 0.15,
    "behavioral": 0.30,
    "lookalike": 0.20,
    "interest": 0.15,
    "desire_graph": 0.20,
}


class AudienceTargeting:
    """Multi-strategy audience targeting engine.

    Usage:
        targeting = AudienceTargeting()
        audience = targeting.find_audience(
            vendor_id="v123",
            ad_category=42,
            ad_embedding=np.array([...]),
            available_strategies=["local", "behavioral", "lookalike"],
            max_audience=10000,
        )
    """

    def __init__(
        self,
        redis_client: Any = None,
        faiss_index: Any = None,
    ):
        self.redis = redis_client
        self.faiss_index = faiss_index

    def find_audience(
        self,
        vendor_id: str,
        ad_category: int,
        ad_embedding: np.ndarray | None = None,
        ad_creative_embedding: np.ndarray | None = None,
        available_strategies: list[str] | None = None,
        seed_user_ids: list[str] | None = None,
        geo_zone: str | None = None,
        max_audience: int = 10000,
    ) -> list[TargetedUser]:
        """Find the target audience for an ad campaign.

        Args:
            vendor_id: Advertiser's vendor ID
            ad_category: Ad product category
            ad_embedding: Ad item embedding (256D from Two-Tower)
            ad_creative_embedding: Ad creative embedding (512D from CLIP)
            available_strategies: Enabled strategies for this plan tier
            seed_user_ids: Seed audience for lookalike expansion
            geo_zone: Geographic targeting zone
            max_audience: Maximum audience size

        Returns:
            List of TargetedUser sorted by match_score
        """
        strategies = available_strategies or ["local", "behavioral"]
        all_matches: dict[str, TargetedUser] = {}

        # Run each enabled strategy
        if "local" in strategies and geo_zone:
            self._target_local(all_matches, geo_zone, ad_category)

        if "behavioral" in strategies:
            self._target_behavioral(all_matches, ad_category, vendor_id)

        if "lookalike" in strategies and seed_user_ids:
            self._target_lookalike(all_matches, seed_user_ids, ad_embedding)

        if "interest" in strategies:
            self._target_interest(all_matches, ad_category)

        if "desire_graph" in strategies:
            self._target_desire_graph(all_matches, ad_category)

        # Compute store visit probability for each user
        for user in all_matches.values():
            user.store_visit_probability = self._predict_store_visit(
                user, vendor_id, ad_category
            )

        # Compute final match score (weighted combination)
        for user in all_matches.values():
            user.match_score = sum(
                STRATEGY_WEIGHTS.get(strategy, 0) * getattr(user, f"{strategy}_score", 0)
                for strategy in strategies
            )
            # Boost by store visit probability (drives traffic amplification)
            user.match_score *= (1.0 + user.store_visit_probability * 0.3)

        # Sort by match score and limit
        audience = sorted(all_matches.values(), key=lambda u: u.match_score, reverse=True)
        audience = audience[:max_audience]

        logger.info(
            "EPSILON targeting: %d users matched for vendor=%s cat=%d (strategies=%s)",
            len(audience), vendor_id, ad_category, strategies,
        )
        return audience

    # ── Strategy 1: Local targeting ────────────────────────────────

    def _target_local(
        self, matches: dict[str, TargetedUser], geo_zone: str, category: int
    ) -> None:
        """Target users in the same geographic zone.

        Uses Redis geo sorted set: geo:{zone}:active_users
        """
        if not self.redis:
            return
        try:
            user_ids = self.redis.smembers(f"geo:{geo_zone}:active_users")
            for uid in (user_ids or []):
                uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                if uid_str not in matches:
                    matches[uid_str] = TargetedUser(user_id=uid_str)
                matches[uid_str].local_score = 0.7
                matches[uid_str].match_reasons.append("local_geo")
        except Exception as e:
            logger.warning("Local targeting failed: %s", e)

    # ── Strategy 2: Behavioral targeting ───────────────────────────

    def _target_behavioral(
        self, matches: dict[str, TargetedUser], category: int, vendor_id: str
    ) -> None:
        """Target users who have interacted with similar categories or vendor.

        Uses Redis: category:{cat_id}:active_users (sorted set by recency)
        and vendor:{vendor_id}:visitors (users who visited this vendor)
        """
        if not self.redis:
            return
        try:
            # Users who browsed this category recently
            cat_users = self.redis.zrevrange(
                f"category:{category}:active_users", 0, 5000
            )
            for uid in (cat_users or []):
                uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                if uid_str not in matches:
                    matches[uid_str] = TargetedUser(user_id=uid_str)
                matches[uid_str].behavioral_score = 0.8
                matches[uid_str].match_reasons.append("category_active")

            # Users who already visited this vendor (retargeting)
            vendor_visitors = self.redis.smembers(f"vendor:{vendor_id}:visitors")
            for uid in (vendor_visitors or []):
                uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                if uid_str not in matches:
                    matches[uid_str] = TargetedUser(user_id=uid_str)
                matches[uid_str].behavioral_score = max(
                    matches[uid_str].behavioral_score, 0.9
                )
                matches[uid_str].match_reasons.append("vendor_retarget")

        except Exception as e:
            logger.warning("Behavioral targeting failed: %s", e)

    # ── Strategy 3: Lookalike targeting ────────────────────────────

    def _target_lookalike(
        self,
        matches: dict[str, TargetedUser],
        seed_user_ids: list[str],
        ad_embedding: np.ndarray | None,
    ) -> None:
        """Find users similar to a seed audience using Two-Tower embeddings.

        Uses FAISS/Milvus to find nearest neighbors in embedding space.
        """
        if not self.faiss_index or ad_embedding is None:
            return
        try:
            # Use ad embedding as query (finds users interested in this type)
            query = ad_embedding.reshape(1, -1).astype(np.float32)
            neighbor_ids, distances = self.faiss_index.search(query, k=5000)

            for i, uid in enumerate(neighbor_ids[0]):
                if uid == -1:
                    continue
                uid_str = str(uid)
                if uid_str not in matches:
                    matches[uid_str] = TargetedUser(user_id=uid_str)
                # Score decays with distance
                sim = 1.0 / (1.0 + float(distances[0][i]))
                matches[uid_str].lookalike_score = max(
                    matches[uid_str].lookalike_score, sim
                )
                matches[uid_str].match_reasons.append("lookalike")

        except Exception as e:
            logger.warning("Lookalike targeting failed: %s", e)

    # ── Strategy 4: Interest targeting ─────────────────────────────

    def _target_interest(
        self, matches: dict[str, TargetedUser], category: int
    ) -> None:
        """Target users with high interest in the ad's category.

        Uses Redis: user:{uid}:interests (hash of category → affinity score)
        Queries the category's interest set directly.
        """
        if not self.redis:
            return
        try:
            # Users with explicit interest in this category
            interest_users = self.redis.zrevrangebyscore(
                f"interest:{category}:users", "+inf", "0.3", start=0, num=5000
            )
            for uid in (interest_users or []):
                uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                if uid_str not in matches:
                    matches[uid_str] = TargetedUser(user_id=uid_str)
                matches[uid_str].interest_score = 0.75
                matches[uid_str].match_reasons.append("interest_affinity")

        except Exception as e:
            logger.warning("Interest targeting failed: %s", e)

    # ── Strategy 5: Desire Graph targeting (PREMIUM only) ──────────

    def _target_desire_graph(
        self, matches: dict[str, TargetedUser], category: int
    ) -> None:
        """Target users via dopaminergic desire graph — UNIQUE to EPSILON.

        Uses the desire graph from ml/feature_store/desire_graph.py:
        - Users with active desire nodes for this category
        - Users in "craving" or "anticipation" phase (not "satiation")
        - Weighted by vulnerability score (circadian timing)

        This targets users at the OPTIMAL moment in their desire cycle,
        dramatically increasing conversion probability.
        """
        if not self.redis:
            return
        try:
            # Users with active desire for this category
            desire_users = self.redis.zrevrangebyscore(
                f"desire:{category}:active_users", "+inf", "0.4",
                start=0, num=3000,
            )
            for uid in (desire_users or []):
                uid_str = uid.decode() if isinstance(uid, bytes) else str(uid)
                if uid_str not in matches:
                    matches[uid_str] = TargetedUser(user_id=uid_str)

                # Get desire phase and strength
                desire_data = self.redis.hgetall(f"desire:{uid_str}:{category}")
                if desire_data:
                    strength = float(desire_data.get(b"strength", desire_data.get("strength", 0.5)))
                    phase = (desire_data.get(b"phase", desire_data.get("phase", b"neutral")))
                    if isinstance(phase, bytes):
                        phase = phase.decode()

                    # Craving & anticipation phases = best ad timing
                    phase_boost = {
                        "craving": 1.0,
                        "anticipation": 0.9,
                        "active": 0.6,
                        "neutral": 0.3,
                        "satiation": 0.1,
                    }.get(phase, 0.3)

                    matches[uid_str].desire_score = strength * phase_boost
                else:
                    matches[uid_str].desire_score = 0.5

                matches[uid_str].match_reasons.append("desire_graph")

        except Exception as e:
            logger.warning("Desire graph targeting failed: %s", e)

    # ── Store Visit Prediction ─────────────────────────────────────

    def _predict_store_visit(
        self,
        user: TargetedUser,
        vendor_id: str,
        category: int,
    ) -> float:
        """Predict probability that this user will visit vendor's store.

        Store visit probability is boosted by:
        - Vendor retarget (already visited → likely to return)
        - Desire graph match (active desire → will explore store)
        - Category active (browsing category → will compare vendors)

        This prediction drives the pStoreVisit signal in the ad ranker,
        which in turn drives the 2-3× traffic amplification guarantee.
        """
        base_prob = 0.05  # 5% baseline store visit rate

        # Retarget boost: users who already visited the vendor
        if "vendor_retarget" in user.match_reasons:
            base_prob *= 3.0  # 3× more likely to revisit

        # Desire boost
        if user.desire_score > 0.5:
            base_prob *= (1.0 + user.desire_score)

        # Category activity boost
        if user.behavioral_score > 0.5:
            base_prob *= 1.5

        # Lookalike users are curious explorers
        if user.lookalike_score > 0.5:
            base_prob *= 1.3

        return min(0.40, base_prob)  # Cap at 40%
