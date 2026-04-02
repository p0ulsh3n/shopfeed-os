"""
Ad Ranker — Multi-task ad scoring (extends MTL/PLE with ad-specific tasks)
============================================================================
Extends the existing 7-task MTL/PLE ranking with 4 ad-specific objectives:

    Organic tasks (reused from MTL/PLE):
        1. p_click       — probability of click
        2. p_save         — probability of save/bookmark
        3. p_share        — probability of share
        4. e_watch_time   — expected watch duration

    Ad-specific tasks (EPSILON additions):
        5. pCTR           — probability of ad click (calibrated)
        6. pCVR           — probability of conversion given click
        7. pROAS          — predicted return on ad spend
        8. pStoreVisit    — probability user visits vendor store after ad

    eCPM Formula (EPSILON's core ranking formula):
        eCPM = bid × pCTR × (α·pCVR + β·pStoreVisit) × quality_score × (1 + uplift_bonus)

    Where:
        α = 0.6 (conversion weight)
        β = 0.4 (store traffic weight — user requested 2-3× traffic boost)
        quality_score = creative_quality × landing_page_score × relevance
        uplift_bonus = causal incremental ROAS from uplift model

    This formula prioritizes ads that BOTH convert AND drive store traffic,
    which is the key differentiator from ANDROMEDA (conversion-only) and
    TikTok (engagement-only).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AdScore:
    """Complete scoring output for a single ad candidate."""
    ad_id: str
    campaign_id: str
    vendor_id: str

    # Core ad predictions
    pCTR: float = 0.0           # Calibrated click-through rate
    pCVR: float = 0.0           # Conversion rate (given click)
    pROAS: float = 0.0          # Predicted return on ad spend
    pStoreVisit: float = 0.0    # Probability of vendor store visit

    # Quality signals
    creative_quality: float = 0.5    # CV quality score of ad creative
    landing_page_score: float = 0.5  # Vendor store quality (load time, UX)
    relevance: float = 0.5           # Ad-user relevance from retrieval
    freshness: float = 1.0           # Creative freshness (decays over time)

    # Computed values
    quality_score: float = 0.0
    ecpm: float = 0.0
    bid_amount: float = 0.0
    uplift_bonus: float = 0.0

    # Metadata
    target_url: str = ""
    creative_type: str = "image"

    def compute_ecpm(self, bid: float, uplift_bonus: float = 0.0) -> float:
        """Compute eCPM using EPSILON's hybrid formula.

        Uniquely weights BOTH conversion AND store traffic, unlike:
        - ANDROMEDA: eCPM = bid × pCTR × pCVR (conversion-only)
        - TikTok:    eCPM = bid × pCTR × engagement (engagement-only)
        """
        CONVERSION_WEIGHT = 0.6
        STORE_TRAFFIC_WEIGHT = 0.4   # Drives 2-3× store visits

        self.bid_amount = bid
        self.uplift_bonus = uplift_bonus

        # Quality score = geometric mean (more robust than arithmetic)
        self.quality_score = (
            self.creative_quality *
            self.landing_page_score *
            self.relevance *
            self.freshness
        ) ** 0.25  # 4th root = geometric mean of 4 factors

        # Hybrid value signal: conversion + store traffic
        value_signal = (
            CONVERSION_WEIGHT * self.pCVR +
            STORE_TRAFFIC_WEIGHT * self.pStoreVisit
        )

        # eCPM = bid × pCTR × hybrid_value × quality × (1 + uplift)
        self.ecpm = (
            bid *
            self.pCTR *
            value_signal *
            self.quality_score *
            (1.0 + uplift_bonus) *
            1000  # Convert to per-mille
        )

        return self.ecpm


class AdRanker:
    """Multi-task ad ranking engine.

    Extends existing MTL/PLE with ad-specific tasks.
    Integrates with existing models via Triton gRPC.

    Usage:
        ranker = AdRanker()
        scores = ranker.rank(user_features, ad_candidates)
        # scores: list[AdScore] sorted by eCPM (highest first)
    """

    def __init__(
        self,
        triton_url: str = "localhost:8001",
        store_traffic_weight: float = 0.4,
    ):
        self.triton_url = triton_url
        self.store_traffic_weight = store_traffic_weight
        self._triton = None

    def _get_triton(self):
        """Lazy-load Triton client."""
        if self._triton is None:
            try:
                from ml.serving.triton_client import TritonClient
                self._triton = TritonClient(base_url=f"http://{self.triton_url}")
            except ImportError:
                logger.warning("Triton client not available — using rule-based scoring")
        return self._triton

    def rank(
        self,
        user_features: dict[str, Any],
        ad_candidates: list[dict[str, Any]],
        session_features: dict[str, Any] | None = None,
    ) -> list[AdScore]:
        """Score and rank ad candidates by eCPM.

        Args:
            user_features: User embedding + demographics + behavior history
            ad_candidates: List of ad candidate dicts from retrieval
            session_features: Current session signals (intent, temporal, etc.)

        Returns:
            List of AdScore sorted by eCPM (highest first)
        """
        scores = []
        triton = self._get_triton()

        for ad in ad_candidates:
            score = AdScore(
                ad_id=ad.get("ad_id", ""),
                campaign_id=ad.get("campaign_id", ""),
                vendor_id=ad.get("vendor_id", ""),
                target_url=ad.get("target_url", ""),
                creative_type=ad.get("creative_type", "image"),
                creative_quality=ad.get("creative_quality", 0.5),
                landing_page_score=ad.get("landing_page_score", 0.5),
                relevance=ad.get("fused_score", 0.5),
            )

            # Predict ad-specific tasks via Triton or fallback
            if triton:
                score = self._predict_triton(score, user_features, ad, session_features)
            else:
                score = self._predict_fallback(score, user_features, ad, session_features)

            # Compute eCPM
            score.compute_ecpm(
                bid=ad.get("bid_amount", 0.0),
                uplift_bonus=ad.get("uplift_bonus", 0.0),
            )
            scores.append(score)

        # Sort by eCPM descending
        scores.sort(key=lambda s: s.ecpm, reverse=True)
        return scores

    def _predict_triton(
        self,
        score: AdScore,
        user_features: dict,
        ad: dict,
        session_features: dict | None,
    ) -> AdScore:
        """Predict ad tasks via Triton inference server.

        Uses the existing MTL/PLE model extended with ad-specific heads.
        """
        try:
            import numpy as np

            # Build feature vector: user + ad + session context
            features = self._build_feature_vector(user_features, ad, session_features)

            # Triton inference (reuses existing model with ad-specific output heads)
            result = self._triton.infer(
                model_name="mtl_ple",
                inputs={"features": np.array([features], dtype=np.float32)},
            )

            # Extract predictions from MTL output
            if result is not None:
                preds = result.get("commerce_scores", [])
                if len(preds) > 0:
                    p = preds[0] if isinstance(preds[0], (list, np.ndarray)) else preds
                    score.pCTR = float(p[0]) if len(p) > 0 else 0.02
                    score.pCVR = float(p[1]) if len(p) > 1 else 0.01
                    score.pROAS = float(p[2]) if len(p) > 2 else 1.0
                    score.pStoreVisit = float(p[3]) if len(p) > 3 else 0.05

        except Exception as e:
            logger.warning(f"Triton ad scoring failed for {score.ad_id}: {e}")
            score = self._predict_fallback(score, user_features, ad, session_features)

        return score

    def _predict_fallback(
        self,
        score: AdScore,
        user_features: dict,
        ad: dict,
        session_features: dict | None,
    ) -> AdScore:
        """Rule-based ad scoring fallback.

        Uses heuristics from user features + ad metadata when Triton is unavailable.
        NOT random — uses real signals to produce meaningful predictions.
        """
        # Base rates from industry benchmarks (social commerce)
        base_ctr = 0.025   # 2.5% average CTR
        base_cvr = 0.015   # 1.5% average CVR
        base_roas = 2.0    # 2× ROAS baseline
        base_store_visit = 0.08  # 8% store visit rate

        # Adjust by ad relevance (from retrieval fusion score)
        relevance = ad.get("fused_score", 0.5)

        # Adjust by user intent
        intent_map = {"buying_now": 2.5, "high": 2.0, "medium": 1.2, "low": 0.7}
        intent = session_features.get("intent_level", "low") if session_features else "low"
        intent_mult = intent_map.get(intent, 0.7)

        # Adjust by creative type (video > carousel > image)
        creative_mult = {"video": 1.4, "carousel": 1.2, "image": 1.0}
        c_mult = creative_mult.get(ad.get("creative_type", "image"), 1.0)

        # Category affinity (from desire graph)
        desire_strength = ad.get("desire_score", 0.0)
        desire_mult = 1.0 + desire_strength * 0.5

        # Compute predictions
        score.pCTR = min(0.15, base_ctr * relevance * intent_mult * c_mult * desire_mult)
        score.pCVR = min(0.10, base_cvr * relevance * intent_mult * desire_mult)
        score.pROAS = base_roas * relevance * intent_mult
        score.pStoreVisit = min(0.25, base_store_visit * relevance * intent_mult * 1.5)

        # Freshness decay
        import time
        created_ts = ad.get("created_at", time.time())
        age_hours = max(0, (time.time() - created_ts) / 3600)
        score.freshness = math.exp(-age_hours / (24 * 7))  # Half-life = 1 week

        return score

    def _build_feature_vector(
        self,
        user_features: dict,
        ad: dict,
        session_features: dict | None,
    ) -> list[float]:
        """Build dense feature vector for Triton MTL inference.

        Combines user, ad, and session features into a single vector
        compatible with the existing MTL/PLE model input.
        """
        features = []

        # User features (256D embedding + demographics)
        user_emb = user_features.get("embedding", [0.0] * 256)
        features.extend(user_emb[:256])

        # Ad features (item embedding + metadata)
        ad_emb = ad.get("item_embedding", [0.0] * 256)
        features.extend(ad_emb[:256])

        # Session context (intent, temporal)
        if session_features:
            features.append(
                {"buying_now": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2}.get(
                    session_features.get("intent_level", "low"), 0.2
                )
            )
            features.append(session_features.get("vulnerability", 0.5))
            features.append(float(session_features.get("is_weekend", False)))
        else:
            features.extend([0.2, 0.5, 0.0])

        return features
