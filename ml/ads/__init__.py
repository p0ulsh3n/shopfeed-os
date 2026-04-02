"""
EPSILON — Hyper-Targeted Ads Engine for ShopFeed OS
=====================================================
Superior to Meta's ANDROMEDA and TikTok's ad system.

Architecture:
    6-stage pipeline per ad request (<15ms total on A100):
    1. Audience Matching    → eligible ads for this user
    2. Multi-Modal Retrieval → ANN similarity (CLIP + behavior + desire + context)
    3. Ad Ranking           → 11-task PLE scoring (pCTR, pCVR, pROAS, pEngagement)
    4. Uplift Filtering     → keep only ads with positive incremental ROAS
    5. GSP Auction          → quality-score weighted second-price auction
    6. Placement            → organic feed injection with fatigue-aware spacing

Key advantages over ANDROMEDA:
    - Multi-modal retrieval (4 signals vs ANDROMEDA's 1 semantic signal)
    - Causal uplift modeling (incremental ROAS, not just CTR correlation)
    - Neural fatigue (attention decay curves, not frequency caps)
    - Desire graph targeting (dopaminergic user interest mapping)
    - Vendor traffic amplification (2-3× store visit uplift)

Subscription model:
    - Vendors pay weekly or monthly plans
    - Budget pacing distributes impressions evenly across the plan period
    - Auto-optimization maximizes store traffic + conversions per budget unit
"""

from .epsilon import EpsilonEngine
from .ad_ranker import AdRanker, AdScore
from .ad_retrieval import MultiModalAdRetrieval
from .auction import GSPAuction, AuctionResult
from .budget_pacing import BudgetPacer, SubscriptionPlan
from .uplift import UpliftModel
from .fatigue import AdFatigueManager
from .targeting import AudienceTargeting

__all__ = [
    "EpsilonEngine",
    "AdRanker",
    "AdScore",
    "MultiModalAdRetrieval",
    "GSPAuction",
    "AuctionResult",
    "BudgetPacer",
    "SubscriptionPlan",
    "UpliftModel",
    "AdFatigueManager",
    "AudienceTargeting",
]
