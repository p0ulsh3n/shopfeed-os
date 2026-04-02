"""
Uplift Modeling — Causal incremental ROAS (what ANDROMEDA can't do)
====================================================================
Standard ad ML predicts: "Will this user click/convert?"
Uplift modeling predicts: "Will this user convert BECAUSE of the ad?"

The difference is critical:
    - User A was going to buy anyway → ad is wasted money (ROAS = 0)
    - User B wouldn't have bought without the ad → ad is valuable (ROAS > 0)

ANDROMEDA and TikTok both maximize pCTR × pCVR (correlational).
EPSILON maximizes INCREMENTAL conversions (causal).

Algorithm: T-Learner (Two-Model approach)
    1. Train model_treated(X) → P(convert | saw ad, features X)
    2. Train model_control(X) → P(convert | no ad, features X)
    3. Uplift = model_treated(X) - model_control(X)
    4. Only serve ads where uplift > 0

Data collection:
    - Randomly hold out 10% of eligible impressions as control group
    - Track conversions for both treated and control
    - Update T-Learner models daily via Kubeflow pipeline
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UpliftPrediction:
    """Uplift prediction for a single (user, ad) pair."""
    user_id: str
    ad_id: str
    p_convert_treated: float     # P(convert | saw ad)
    p_convert_control: float     # P(convert | no ad)
    uplift: float                # Incremental conversion probability
    incremental_roas: float      # Expected incremental ROAS
    should_serve: bool           # Whether this ad should be served


class UpliftModel:
    """T-Learner uplift model for causal ad impact estimation.

    Uses two separate models:
    - treatment_model: predicts conversion probability for users who see the ad
    - control_model: predicts conversion probability for users who DON'T see the ad

    The uplift = P(convert|treated) - P(convert|control) measures the
    CAUSAL impact of showing the ad.

    Usage:
        uplift = UpliftModel()
        predictions = uplift.predict(
            user_features=[...],
            ad_features=[...],
            avg_order_value=50.0,
            ad_cost=0.05,
        )
        for pred in predictions:
            if pred.should_serve:
                # Serve this ad — it creates incremental value
                pass
    """

    def __init__(
        self,
        model_dir: str = "checkpoints/uplift",
        min_uplift_threshold: float = 0.001,  # Minimum 0.1% incremental conversion
        holdout_rate: float = 0.10,            # 10% random holdout for control
    ):
        self.model_dir = model_dir
        self.min_uplift_threshold = min_uplift_threshold
        self.holdout_rate = holdout_rate
        self._treatment_model = None
        self._control_model = None
        self._load_models()

    def _load_models(self) -> None:
        """Load T-Learner models (treatment + control)."""
        try:
            import lightgbm as lgb
            from pathlib import Path

            treatment_path = Path(self.model_dir) / "treatment_model.txt"
            control_path = Path(self.model_dir) / "control_model.txt"

            if treatment_path.exists():
                self._treatment_model = lgb.Booster(model_file=str(treatment_path))
                logger.info("Uplift treatment model loaded: %s", treatment_path)
            if control_path.exists():
                self._control_model = lgb.Booster(model_file=str(control_path))
                logger.info("Uplift control model loaded: %s", control_path)

        except ImportError:
            logger.info(
                "LightGBM not installed — uplift model uses heuristic fallback. "
                "pip install lightgbm>=4.3"
            )

    def predict(
        self,
        user_features: list[dict[str, Any]],
        ad_features: list[dict[str, Any]],
        avg_order_value: float = 50.0,
        ad_cost_per_impression: float = 0.05,
    ) -> list[UpliftPrediction]:
        """Predict uplift for each (user, ad) pair.

        Args:
            user_features: List of user feature dicts
            ad_features: List of corresponding ad feature dicts
            avg_order_value: Average order value for ROAS calculation
            ad_cost_per_impression: Cost per impression for ROAS calculation

        Returns:
            List of UpliftPrediction with should_serve recommendations
        """
        predictions = []

        for uf, af in zip(user_features, ad_features):
            if self._treatment_model and self._control_model:
                pred = self._predict_lgb(uf, af, avg_order_value, ad_cost_per_impression)
            else:
                pred = self._predict_heuristic(uf, af, avg_order_value, ad_cost_per_impression)
            predictions.append(pred)

        served = sum(1 for p in predictions if p.should_serve)
        logger.debug(
            "Uplift: %d/%d ads recommended (%.1f%% filtered as non-incremental)",
            served, len(predictions),
            100 * (1 - served / max(len(predictions), 1)),
        )
        return predictions

    def _predict_lgb(
        self,
        user_features: dict,
        ad_features: dict,
        avg_order_value: float,
        ad_cost: float,
    ) -> UpliftPrediction:
        """Predict using trained LightGBM T-Learner models."""
        features = self._build_features(user_features, ad_features)
        feature_array = np.array([features], dtype=np.float64)

        p_treated = float(self._treatment_model.predict(feature_array)[0])
        p_control = float(self._control_model.predict(feature_array)[0])

        # Clamp to valid probability range
        p_treated = max(0.0, min(1.0, p_treated))
        p_control = max(0.0, min(1.0, p_control))

        uplift = p_treated - p_control
        incremental_roas = (uplift * avg_order_value) / max(ad_cost, 0.001)

        return UpliftPrediction(
            user_id=user_features.get("user_id", ""),
            ad_id=ad_features.get("ad_id", ""),
            p_convert_treated=p_treated,
            p_convert_control=p_control,
            uplift=uplift,
            incremental_roas=incremental_roas,
            should_serve=uplift > self.min_uplift_threshold,
        )

    def _predict_heuristic(
        self,
        user_features: dict,
        ad_features: dict,
        avg_order_value: float,
        ad_cost: float,
    ) -> UpliftPrediction:
        """Heuristic uplift estimation when LightGBM models not available.

        Uses user engagement signals to estimate incremental impact:
        - New users → high uplift (discovery-driven, wouldn't find vendor otherwise)
        - High-intent users → low uplift (would convert anyway)
        - Medium-intent, relevant category → highest uplift sweet spot
        """
        # User signals
        interaction_count = user_features.get("interaction_count", 0)
        intent_level = user_features.get("intent_level", "low")
        category_affinity = user_features.get("category_affinity", 0.0)

        # Ad signals
        relevance = ad_features.get("relevance", 0.5)
        creative_quality = ad_features.get("creative_quality", 0.5)

        # Heuristic treatment probability
        intent_map = {"buying_now": 0.15, "high": 0.08, "medium": 0.04, "low": 0.02}
        p_treated = intent_map.get(intent_level, 0.02) * (1 + relevance) * creative_quality

        # Control probability (conversion without ad)
        # High intent users would convert anyway → high control rate
        control_map = {"buying_now": 0.12, "high": 0.05, "medium": 0.01, "low": 0.002}
        p_control = control_map.get(intent_level, 0.002)

        # New users have virtually zero control rate (wouldn't discover vendor)
        if interaction_count < 20:
            p_control *= 0.1  # New users = pure uplift opportunity

        # Category affinity boosts treatment effect
        p_treated *= (1 + category_affinity * 0.5)

        p_treated = min(0.20, p_treated)
        p_control = min(0.15, p_control)

        uplift = max(0, p_treated - p_control)
        incremental_roas = (uplift * avg_order_value) / max(ad_cost, 0.001)

        return UpliftPrediction(
            user_id=user_features.get("user_id", ""),
            ad_id=ad_features.get("ad_id", ""),
            p_convert_treated=p_treated,
            p_convert_control=p_control,
            uplift=uplift,
            incremental_roas=incremental_roas,
            should_serve=uplift > self.min_uplift_threshold,
        )

    def _build_features(self, user_features: dict, ad_features: dict) -> list[float]:
        """Build feature vector for LightGBM prediction."""
        return [
            user_features.get("interaction_count", 0),
            {"buying_now": 4, "high": 3, "medium": 2, "low": 1}.get(
                user_features.get("intent_level", "low"), 1
            ),
            user_features.get("category_affinity", 0.0),
            user_features.get("vulnerability", 0.5),
            user_features.get("session_duration_s", 0),
            user_features.get("days_since_last_purchase", 30),
            ad_features.get("relevance", 0.5),
            ad_features.get("creative_quality", 0.5),
            ad_features.get("bid_amount", 0.0),
            ad_features.get("category_match", 0.0),
        ]

    def should_holdout(self, user_id: str) -> bool:
        """Deterministic control group assignment (10% holdout).

        Uses hash-based assignment so the same user is always in the
        same group — essential for valid causal inference.
        """
        hash_val = hash(f"epsilon_uplift_{user_id}") % 1000
        return hash_val < int(self.holdout_rate * 1000)
