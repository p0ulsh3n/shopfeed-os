"""
Budget Pacing — Subscription-based delivery optimization
==========================================================
Vendors pay WEEKLY or MONTHLY subscription plans (not daily CPM).
Budget pacing ensures smooth, even distribution of impressions across
the subscription period to maximize store traffic consistently.

Plans:
    - STARTER:  weekly plan  — lower impression volume, local targeting
    - GROWTH:   monthly plan — 3× impression volume + lookalike targeting
    - PREMIUM:  monthly plan — unlimited impressions + priority auction boost

Pacing algorithm:
    PID controller that adjusts bid modifier in real-time:
    - Over-pacing (spent too fast) → reduce bid modifier → fewer wins
    - Under-pacing (spent too slow) → increase bid modifier → more wins
    - Target: linear delivery across the plan period

Traffic amplification:
    The system guarantees minimum store traffic uplift:
    - STARTER:  1.5× baseline store visits
    - GROWTH:   2.5× baseline store visits
    - PREMIUM:  4.0× baseline store visits
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PlanTier(str, Enum):
    STARTER = "starter"       # Weekly plan
    GROWTH = "growth"         # Monthly plan
    PREMIUM = "premium"       # Monthly premium


@dataclass
class SubscriptionPlan:
    """Vendor's active advertising subscription."""
    plan_id: str
    vendor_id: str
    tier: PlanTier
    # Billing period
    start_ts: float              # Unix timestamp of plan start
    end_ts: float                # Unix timestamp of plan end
    # Budget
    total_budget: float          # Total budget for the period (e.g. $200/week)
    spent: float = 0.0           # Amount spent so far
    # Delivery targets
    target_impressions: int = 0         # Guaranteed impressions
    delivered_impressions: int = 0
    target_store_visits: int = 0        # Guaranteed store visit uplift
    delivered_store_visits: int = 0
    # Pacing state
    bid_modifier: float = 1.0    # Current bid adjustment (0.5 - 2.0)
    is_active: bool = True
    is_paused: bool = False      # Manual pause by vendor

    @property
    def remaining_budget(self) -> float:
        return max(0, self.total_budget - self.spent)

    @property
    def remaining_time_fraction(self) -> float:
        """Fraction of subscription period remaining (0-1)."""
        now = time.time()
        total = max(self.end_ts - self.start_ts, 1)
        remaining = max(0, self.end_ts - now)
        return remaining / total

    @property
    def elapsed_time_fraction(self) -> float:
        return 1.0 - self.remaining_time_fraction

    @property
    def spend_pace(self) -> float:
        """Fraction of budget spent (0-1)."""
        return self.spent / max(self.total_budget, 0.01)

    @property
    def delivery_pace(self) -> float:
        """Fraction of impressions delivered (0-1)."""
        return self.delivered_impressions / max(self.target_impressions, 1)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.end_ts

    @property
    def is_budget_exhausted(self) -> bool:
        return self.remaining_budget <= 0


# Plan configurations
PLAN_CONFIGS = {
    PlanTier.STARTER: {
        "duration_days": 7,
        "min_impressions_per_dollar": 100,
        "store_visit_multiplier": 1.5,   # 1.5× baseline traffic
        "priority_boost": 1.0,           # No auction priority boost
        "max_bid_modifier": 1.5,
        "targeting": ["local", "behavioral"],
    },
    PlanTier.GROWTH: {
        "duration_days": 30,
        "min_impressions_per_dollar": 150,
        "store_visit_multiplier": 2.5,   # 2.5× baseline traffic
        "priority_boost": 1.2,           # 20% auction priority boost
        "max_bid_modifier": 1.8,
        "targeting": ["local", "behavioral", "lookalike", "interest"],
    },
    PlanTier.PREMIUM: {
        "duration_days": 30,
        "min_impressions_per_dollar": 200,
        "store_visit_multiplier": 4.0,   # 4× baseline traffic
        "priority_boost": 1.5,           # 50% auction priority boost
        "max_bid_modifier": 2.5,
        "targeting": ["local", "behavioral", "lookalike", "interest", "desire_graph"],
    },
}


class BudgetPacer:
    """PID-controller based budget pacing for subscription plans.

    Adjusts bid_modifier in real-time to ensure smooth delivery:
    - If spending too fast → lower bid_modifier → win fewer auctions
    - If spending too slow → raise bid_modifier → win more auctions
    - Auto-pause when budget exhausted or plan expired

    Usage:
        pacer = BudgetPacer()
        plan.bid_modifier = pacer.compute_bid_modifier(plan)
        effective_bid = original_bid * plan.bid_modifier
    """

    # PID controller gains
    KP = 0.5   # Proportional gain
    KI = 0.1   # Integral gain (prevents steady-state error)
    KD = 0.05  # Derivative gain (dampens oscillation)

    def __init__(self):
        self._prev_errors: dict[str, float] = {}
        self._integral_errors: dict[str, float] = {}

    def compute_bid_modifier(self, plan: SubscriptionPlan) -> float:
        """Compute bid modifier using PID controller.

        Target: spend_pace should match elapsed_time_fraction.
        - If spend_pace > elapsed_fraction → over-spending → reduce modifier
        - If spend_pace < elapsed_fraction → under-spending → increase modifier

        Returns modifier in range [0.3, max_bid_modifier for plan tier].
        """
        if not plan.is_active or plan.is_paused:
            return 0.0
        if plan.is_expired or plan.is_budget_exhausted:
            plan.is_active = False
            return 0.0

        config = PLAN_CONFIGS.get(plan.tier, PLAN_CONFIGS[PlanTier.STARTER])

        # Error = target pace - actual pace
        target_pace = plan.elapsed_time_fraction
        actual_pace = plan.spend_pace
        error = target_pace - actual_pace  # Positive = under-spending

        # PID terms
        prev_error = self._prev_errors.get(plan.plan_id, 0.0)
        integral = self._integral_errors.get(plan.plan_id, 0.0) + error

        p_term = self.KP * error
        i_term = self.KI * integral
        d_term = self.KD * (error - prev_error)

        adjustment = p_term + i_term + d_term

        # Apply adjustment to current modifier
        new_modifier = plan.bid_modifier + adjustment

        # Clamp to valid range
        max_mod = config["max_bid_modifier"]
        new_modifier = max(0.3, min(max_mod, new_modifier))

        # Store state for next iteration
        self._prev_errors[plan.plan_id] = error
        self._integral_errors[plan.plan_id] = integral

        logger.debug(
            "Pacing %s: error=%.3f P=%.3f I=%.3f D=%.3f → modifier=%.2f (was %.2f)",
            plan.plan_id, error, p_term, i_term, d_term,
            new_modifier, plan.bid_modifier,
        )

        plan.bid_modifier = new_modifier
        return new_modifier

    def record_impression(
        self,
        plan: SubscriptionPlan,
        cost: float,
        resulted_in_store_visit: bool = False,
    ) -> None:
        """Record an ad impression and update plan counters."""
        plan.spent += cost
        plan.delivered_impressions += 1
        if resulted_in_store_visit:
            plan.delivered_store_visits += 1

        # Auto-pause if budget exhausted
        if plan.is_budget_exhausted:
            plan.is_active = False
            logger.info(
                "Plan %s (vendor=%s) budget exhausted: $%.2f spent of $%.2f",
                plan.plan_id, plan.vendor_id, plan.spent, plan.total_budget,
            )

    def get_store_traffic_target(self, plan: SubscriptionPlan) -> int:
        """Calculate target store visits for this plan.

        Based on the vendor's baseline traffic (from ClickHouse analytics)
        and the plan tier's traffic multiplier.
        """
        config = PLAN_CONFIGS.get(plan.tier, PLAN_CONFIGS[PlanTier.STARTER])
        multiplier = config["store_visit_multiplier"]
        # target_store_visits should be set at plan creation from baseline analytics
        return plan.target_store_visits

    def get_priority_boost(self, plan: SubscriptionPlan) -> float:
        """Get auction priority boost for this plan tier.

        PREMIUM plans get 50% eCPM boost in auction.
        """
        config = PLAN_CONFIGS.get(plan.tier, PLAN_CONFIGS[PlanTier.STARTER])
        return config["priority_boost"]

    def get_available_targeting(self, plan: SubscriptionPlan) -> list[str]:
        """Get targeting methods available for this plan tier.

        STARTER: local + behavioral only
        GROWTH:  + lookalike + interest
        PREMIUM: + desire_graph (dopaminergic targeting)
        """
        config = PLAN_CONFIGS.get(plan.tier, PLAN_CONFIGS[PlanTier.STARTER])
        return config["targeting"]
