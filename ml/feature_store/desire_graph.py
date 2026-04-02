"""
Desire Graph 6D — Psychographic User Modeling (t.md §5)
==========================================================
Extends the user feature vector with 6 psychographic dimensions
that capture desire state beyond simple category preferences.

The 6 dimensions:
    1. Emotional triggers (2D)     — valence + arousal from recent interactions
    2. Aspirational identity (2D)  — gap between current and ideal self
    3. Temporal vulnerability (1D) — from temporal.py
    4. Decision fatigue (1D)       — decays with session length
    5. Subconscious patterns (2D)  — micro-pause & scroll-reverse aggregates
    6. Narrative arc position (2D) — progress in active story/collection journey

Total: 10D vector that captures WHERE the user is in their desire cycle.

This is ADDITIVE: the desire vector is concatenated to existing user features
as an additional signal. It does not replace or modify any existing features.
"""

from __future__ import annotations

import math
import logging
from typing import Any

import torch

from .temporal import compute_temporal_features

logger = logging.getLogger(__name__)


class DesireGraph:
    """Computes a 10D desire vector for a user.

    This vector captures the user's current psychographic state,
    making the ML model aware of WHEN and WHY a user is most
    susceptible to specific content — not just WHAT they like.

    Usage:
        graph = DesireGraph()
        desire_vec = graph.compute_desire_vector(
            user_profile=user_profile,
            session_state=session_state,
            hour=23, minute=30,
        )
        # → torch.Tensor[10]
    """

    def compute_desire_vector(
        self,
        user_profile: dict[str, Any] | None = None,
        session_state: dict[str, Any] | None = None,
        hour: int = 12,
        minute: int = 0,
        day_of_week: int = 0,
        session_duration_s: float = 0.0,
    ) -> torch.Tensor:
        """Compute the 10D desire vector.

        Args:
            user_profile: user data from DB (category_prefs, etc.)
            session_state: current session (from SessionState)
            hour/minute: current time
            day_of_week: 0=Monday
            session_duration_s: how long the user has been browsing

        Returns:
            Tensor of shape (10,) — all values in [0, 1] or [-1, 1]
        """
        user_profile = user_profile or {}
        session_state = session_state or {}

        parts = []

        # ── 1. Emotional triggers (2D): valence + arousal ──
        parts.append(self._compute_emotional_state(session_state))

        # ── 2. Aspirational identity (2D): current vs ideal gap ──
        parts.append(self._compute_aspiration_gap(user_profile, session_state))

        # ── 3. Temporal vulnerability (1D) ──
        temporal = compute_temporal_features(hour, minute, day_of_week, session_duration_s)
        vulnerability = temporal[4:5]  # vulnerability score
        parts.append(vulnerability)

        # ── 4. Decision fatigue (1D) ──
        parts.append(self._compute_decision_fatigue(session_state))

        # ── 5. Subconscious patterns (2D) ──
        parts.append(self._compute_subconscious_signals(session_state))

        # ── 6. Narrative arc position (2D) ──
        parts.append(self._compute_narrative_position(user_profile, session_state))

        return torch.cat(parts, dim=0)

    def _compute_emotional_state(self, session: dict) -> torch.Tensor:
        """Emotional valence + arousal from recent session actions.

        Valence: positive actions (save, buy) → high; negative (skip) → low
        Arousal: speed and density of actions → high; slow browsing → low
        """
        actions = session.get("last_actions", [])
        if not actions:
            return torch.tensor([0.5, 0.3], dtype=torch.float32)

        positive_actions = {"buy_now", "purchase", "add_to_cart", "save", "like", "share"}
        negative_actions = {"skip", "not_interested"}

        pos_count = sum(1 for a in actions if a.get("type") in positive_actions)
        neg_count = sum(1 for a in actions if a.get("type") in negative_actions)
        total = max(len(actions), 1)

        # Valence: ratio of positive to total
        valence = pos_count / total

        # Arousal: action density (more actions = higher arousal)
        arousal = min(1.0, total / 20.0)

        return torch.tensor([valence, arousal], dtype=torch.float32)

    def _compute_aspiration_gap(
        self, user_profile: dict, session: dict
    ) -> torch.Tensor:
        """Gap between current purchasing behavior and aspirational targets.

        Dim 1: price aspiration — browsing items above usual price range
        Dim 2: category expansion — exploring new categories vs sticking to known
        """
        # Price aspiration: session price interest vs historical average
        price_range = session.get("price_range", {})
        session_max = float(price_range.get("max", 100))

        hist_price_ranges = user_profile.get("price_ranges", {})
        hist_avgs = [
            v.get("avg", 50) for v in hist_price_ranges.values()
            if isinstance(v, dict)
        ]
        hist_avg = sum(hist_avgs) / max(len(hist_avgs), 1) if hist_avgs else 50.0

        # How much above their normal range are they browsing?
        price_aspiration = min(1.0, max(0.0, (session_max - hist_avg) / max(hist_avg, 1)))

        # Category expansion: new categories in session vs historical prefs
        known_cats = set(str(k) for k in (user_profile.get("category_prefs", {}) or {}))
        active_cats = set(str(k) for k in (session.get("active_categories", {}) or {}))
        new_cats = active_cats - known_cats
        cat_expansion = min(1.0, len(new_cats) / max(len(active_cats), 1)) if active_cats else 0.0

        return torch.tensor([price_aspiration, cat_expansion], dtype=torch.float32)

    def _compute_decision_fatigue(self, session: dict) -> torch.Tensor:
        """Decision fatigue: increases with number of items viewed.

        After ~40 items viewed, decision quality drops significantly.
        This is a log-saturating curve.
        """
        actions = session.get("last_actions", [])
        n_views = sum(1 for a in actions if a.get("type") in ("view", "zoom", "pause"))

        # Log-saturating fatigue: plateaus at ~40 views
        fatigue = min(1.0, math.log1p(n_views / 10.0) / math.log1p(4.0))

        return torch.tensor([fatigue], dtype=torch.float32)

    def _compute_subconscious_signals(self, session: dict) -> torch.Tensor:
        """Aggregate micro-pause and scroll-reverse frequency.

        Dim 1: micro-pause density (ratio of micro_pause actions to total)
        Dim 2: scroll-reverse density (how often user scrolled back)
        """
        actions = session.get("last_actions", [])
        total = max(len(actions), 1)

        micro_pauses = sum(1 for a in actions if a.get("type") == "micro_pause")
        scroll_reverses = sum(1 for a in actions if a.get("type") == "scroll_reverse")

        pause_density = min(1.0, micro_pauses / total)
        reverse_density = min(1.0, scroll_reverses / total)

        return torch.tensor([pause_density, reverse_density], dtype=torch.float32)

    def _compute_narrative_position(
        self, user_profile: dict, session: dict
    ) -> torch.Tensor:
        """Position in active narrative arc (collection/brand journey).

        Dim 1: collection progress — average completion across active collections
        Dim 2: journey momentum — are they making progress this session?

        If no active collections/journeys, returns [0, 0] (neutral).
        """
        # Collection progress from user profile
        collections = user_profile.get("active_collections", [])
        if collections:
            ratios = [c.get("ratio", 0) for c in collections if isinstance(c, dict)]
            avg_progress = sum(ratios) / max(len(ratios), 1)
        else:
            avg_progress = 0.0

        # Journey momentum: purchases in this session
        actions = session.get("last_actions", [])
        purchases = sum(
            1 for a in actions
            if a.get("type") in ("buy_now", "purchase", "add_to_cart")
        )
        momentum = min(1.0, purchases / 5.0)

        return torch.tensor([avg_progress, momentum], dtype=torch.float32)
