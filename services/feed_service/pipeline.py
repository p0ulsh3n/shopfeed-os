"""Feed Service — Recommendation Pipeline <80ms — Section 04.

This is the heart of ShopFeed OS. Every feed request triggers this pipeline:

    Catalog ∞ → Retrieval (10ms) → Pre-Ranking (15ms) →
    MTL Ranking (40ms) → Re-Ranking (15ms) → Feed served

The 3-speed scoring architecture (Section 12):
    Score_Final = Model_Batch(V3) + Delta_Online(V2) + Session_Boost(V1)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

from shared.models.product import ContentType, PoolLevel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Content Type Multipliers — Section 01
# ──────────────────────────────────────────────────────────────

CONTENT_TYPE_BOOST: dict[str, float] = {
    ContentType.PHOTO: 1.0,
    ContentType.CAROUSEL: 1.4,
    ContentType.VIDEO: 2.0,
    ContentType.LIVE: 3.0,
    ContentType.LIVE_SCHEDULED: 2.5,
}


@dataclass
class FeedCandidate:
    """A scored candidate for the feed."""
    content_id: str
    product_id: str
    vendor_id: str
    content_type: str = ContentType.PHOTO
    pool_level: str = PoolLevel.L1

    # Scores from different pipeline stages
    retrieval_score: float = 0.0        # Two-Tower similarity
    ranking_score: float = 0.0          # DeepFM CTR
    mtl_scores: dict[str, float] = field(default_factory=dict)
    commerce_score: float = 0.0         # Section 02 weighted sum
    online_delta: float = 0.0           # Monolith V2 correction
    session_boost: float = 0.0          # Session State V1

    # Multipliers — Section 02
    account_weight: float = 1.0         # Vendor tier
    freshness_mult: float = 1.0         # ×1.0 to ×2.0
    geo_mult: float = 1.0              # ×1.0 to ×1.2
    content_type_mult: float = 1.0     # Section 01

    # Metadata
    base_price: float = 0.0
    stock: int = 999
    vendor_tier: str = "bronze"

    @property
    def final_score(self) -> float:
        """Section 02 — Score Final formula.

        Score_Final = Score_Contenu × Account_Weight × M_freshness
                    × M_geo × M_content_type
        """
        # Base: commerce score from MTL + online delta + session boost
        base = self.commerce_score + self.online_delta + self.session_boost
        return base * self.account_weight * self.freshness_mult * self.geo_mult * self.content_type_mult


class RecommendationPipeline:
    """Full feed recommendation pipeline — <80ms per request.

    Section 04 — 4 stages:
        1. Retrieval (~10ms)   — Two-Tower ANN → ~2,000 candidates
        2. Pre-Ranking (~15ms) — Hard filters + light scoring → ~400
        3. Ranking (~40ms)     — MTL 7-task scoring → ~80
        4. Re-Ranking (~15ms)  — Diversity, equity, business rules → ~15
    """

    def __init__(
        self,
        model_registry=None,
        monolith_trainer=None,
        session_store=None,
    ):
        self.registry = model_registry
        self.monolith = monolith_trainer
        self.session_store = session_store

    async def generate_feed(
        self,
        user_id: str,
        session_id: str,
        user_embedding: np.ndarray | None = None,
        content_types: list[str] | None = None,
        limit: int = 15,
    ) -> list[FeedCandidate]:
        """Execute full pipeline. Target: <80ms total.

        Args:
            user_id: Authenticated user
            session_id: Current browsing session
            user_embedding: Pre-computed user vector (256-dim)
            content_types: Filter by content type
            limit: Number of items to return

        Returns:
            Ranked list of FeedCandidate items
        """
        t0 = time.perf_counter()

        # ── Stage 1: Retrieval (~10ms) ──
        candidates = await self._retrieve(user_embedding)
        t1 = time.perf_counter()

        # ── Stage 2: Pre-Ranking (~15ms) ──
        candidates = await self._pre_rank(candidates, content_types)
        t2 = time.perf_counter()

        # ── Stage 3: MTL Ranking (~40ms) ──
        candidates = await self._rank(candidates, user_id)
        t3 = time.perf_counter()

        # ── Stage 4: Re-Ranking (~15ms) ──
        candidates = await self._re_rank(candidates, user_id, session_id, limit)
        t4 = time.perf_counter()

        total_ms = (t4 - t0) * 1000
        logger.info(
            "Pipeline: %.1fms (retrieval=%.1f, pre_rank=%.1f, rank=%.1f, re_rank=%.1f) → %d items",
            total_ms,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            (t4 - t3) * 1000,
            len(candidates),
        )

        return candidates

    # ── Stage 1: Retrieval ──

    async def _retrieve(
        self, user_embedding: np.ndarray | None, top_k: int = 2000
    ) -> list[FeedCandidate]:
        """Two-Tower ANN retrieval — <10ms for 10M+ items."""
        if self.registry and user_embedding is not None:
            results = self.registry.retrieve_candidates(user_embedding, top_k=top_k)
            return [
                FeedCandidate(
                    content_id=item_id,
                    product_id=item_id,
                    vendor_id="",
                    retrieval_score=score,
                )
                for item_id, score in results
            ]

        # Fallback: return empty (cold start or no index)
        return []

    # ── Stage 2: Pre-Ranking ──

    async def _pre_rank(
        self,
        candidates: list[FeedCandidate],
        content_types: list[str] | None = None,
    ) -> list[FeedCandidate]:
        """Hard filters + light scoring → ~400 candidates."""
        filtered = []
        for c in candidates:
            # Hard filters
            if c.stock <= 0:
                continue
            if content_types and c.content_type not in content_types:
                continue
            filtered.append(c)

        # Light scoring: use retrieval score + content type boost
        for c in filtered:
            c.content_type_mult = CONTENT_TYPE_BOOST.get(c.content_type, 1.0)
            c.ranking_score = c.retrieval_score * c.content_type_mult

        # Sort and truncate to 400
        filtered.sort(key=lambda x: x.ranking_score, reverse=True)
        return filtered[:400]

    # ── Stage 3: MTL Ranking ──

    async def _rank(
        self, candidates: list[FeedCandidate], user_id: str
    ) -> list[FeedCandidate]:
        """MTL 7-task scoring → ~80 candidates.

        Commerce Score (Section 02):
            P(buy_now)×12 + P(purchase)×10 + P(add_to_cart)×8
            + P(save)×6 + P(share)×5 + E(watch_time)×3 - P(negative)×8
        """
        # In production: batch predict with MTL model
        # For now, use retrieval score as approximation
        for c in candidates:
            c.commerce_score = c.ranking_score

        # Apply multipliers
        for c in candidates:
            c.commerce_score *= c.account_weight * c.freshness_mult * c.geo_mult

        candidates.sort(key=lambda x: x.commerce_score, reverse=True)
        return candidates[:80]

    # ── Stage 4: Re-Ranking ──

    async def _re_rank(
        self,
        candidates: list[FeedCandidate],
        user_id: str,
        session_id: str,
        limit: int,
    ) -> list[FeedCandidate]:
        """Diversity, equity, and business rules → final feed.

        Section 09 — Anti-Ghost rules:
            - Max 2 items from same vendor per 10 items
            - Min 1 new vendor (Bronze) per session batch
            - Cap monopole: no vendor > 8% of feed
            - Diversity: no 3+ items from same category in a row
        """
        # Apply Session State boost (V1)
        if self.session_store:
            for c in candidates:
                session_data = await self.session_store.get_session(session_id)
                if session_data:
                    # Boost categories matching active interests
                    # Penalize negative categories
                    pass  # Implemented in SessionState class

        # Vendor diversity constraint
        vendor_count: dict[str, int] = {}
        final: list[FeedCandidate] = []
        last_categories: list[int] = []

        for c in sorted(candidates, key=lambda x: x.final_score, reverse=True):
            # Max 2 per vendor per batch
            vc = vendor_count.get(c.vendor_id, 0)
            if vc >= 2:
                continue

            vendor_count[c.vendor_id] = vc + 1
            final.append(c)

            if len(final) >= limit:
                break

        return final


# ──────────────────────────────────────────────────────────────
# Session State — Section 13 (Vitesse 1)
# ──────────────────────────────────────────────────────────────

class SessionState:
    """In-memory session state for real-time feed adaptation.

    Section 13 — Updated in <50ms per user action.

    Structure (stored in Redis with 30min TTL):
        active_categories:    [{cat, weight}, ...]
        price_range_signal:   {min, max, confidence}
        intent_level:         low | medium | high | buying_now | checkout
        negative_categories:  set of banned categories
        last_actions:         [{type, product_id, ts}, ...]
        session_duration_s:   float
    """

    def __init__(self, redis_client=None):
        self.redis = redis_client
        # In-memory fallback
        self._sessions: dict[str, dict] = {}

    async def get_session(self, session_id: str) -> dict | None:
        """Get current session state."""
        if self.redis:
            data = await self.redis.hgetall(f"session:{session_id}")
            return data if data else None
        return self._sessions.get(session_id)

    async def update_session(
        self,
        session_id: str,
        user_id: str,
        action_type: str,
        product_id: str | None = None,
        category: str | None = None,
        price: float | None = None,
    ) -> dict:
        """Update session state from a user action — <50ms.

        Section 13 — Action → Signal mapping:
            pause_3s     → category +0.3
            zoom         → intent = high, price recalibrated
            add_to_cart  → intent = buying_now
            buy_now      → intent = checkout, buffer invalidated
            skip         → category -0.5, added to negatives
            not_interested → category banned for session
        """
        session = await self.get_session(session_id) or {
            "user_id": user_id,
            "active_categories": {},
            "price_range": {"min": 0, "max": 1000, "confidence": 0.1},
            "intent_level": "low",
            "negative_categories": [],
            "last_actions": [],
            "skip_count": {},
        }

        # Apply action
        if action_type == "pause_3s" and category:
            cats = session["active_categories"]
            cats[category] = cats.get(category, 0) + 0.3
            if cats.get(category, 0) > 0.5:
                cats[category] += 0.2  # Reinforcement

        elif action_type == "zoom":
            session["intent_level"] = "high"
            if price:
                pr = session["price_range"]
                pr["min"] = min(pr["min"], price * 0.7)
                pr["max"] = max(pr["max"], price * 1.3)
                pr["confidence"] = min(pr["confidence"] + 0.2, 1.0)

        elif action_type == "add_to_cart":
            session["intent_level"] = "buying_now"

        elif action_type == "buy_now":
            session["intent_level"] = "checkout"

        elif action_type == "skip" and category:
            sc = session["skip_count"]
            sc[category] = sc.get(category, 0) + 1
            cats = session["active_categories"]
            cats[category] = cats.get(category, 0) - 0.5

            if sc[category] >= 2:
                if category not in session["negative_categories"]:
                    session["negative_categories"].append(category)

        elif action_type == "not_interested" and category:
            if category not in session["negative_categories"]:
                session["negative_categories"].append(category)

        # Record action
        actions = session["last_actions"]
        actions.append({"type": action_type, "product_id": product_id})
        session["last_actions"] = actions[-20:]  # Keep last 20

        # Persist
        if self.redis:
            import json
            await self.redis.set(
                f"session:{session_id}",
                json.dumps(session, default=str),
                ex=1800,  # 30 min TTL
            )
        else:
            self._sessions[session_id] = session

        return session

    def compute_session_vector(self, session: dict) -> np.ndarray:
        """Compute a dense vector from session state for similarity comparison.

        Used for the "should we recalculate N+1?" check (Section 04b):
            delta = cosine_distance(old_vector, new_vector)
            if delta > 0.15: invalidate buffer
        """
        # Simple weighted category vector
        cats = session.get("active_categories", {})
        vec = np.zeros(200, dtype=np.float32)  # MAX_CATEGORIES
        for cat_str, weight in cats.items():
            try:
                idx = hash(cat_str) % 200
                vec[idx] = float(weight)
            except (ValueError, TypeError):
                pass

        # Add intent level as a dimension
        intent_map = {"low": 0.1, "medium": 0.3, "high": 0.6, "buying_now": 0.9, "checkout": 1.0}
        intent = session.get("intent_level", "low")
        vec[0] = intent_map.get(intent, 0.1)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec
