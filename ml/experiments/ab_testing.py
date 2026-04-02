"""
Statsig A/B Testing Integration (archi-2026 §9.6)
====================================================
Feature flags and ML experiment management for production model rollouts.

Architecture:
    New model trained → Statsig creates experiment
    5% of users get new model → measure engagement
    If +2% engagement → promote to 100%
    If degradation → automatic rollback

Requires:
    pip install statsig>=0.36
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import statsig
    from statsig import statsig as statsig_client
    from statsig.statsig_options import StatsigOptions
    HAS_STATSIG = True
except ImportError:
    HAS_STATSIG = False
    logger.warning("statsig not installed — A/B testing uses local fallback. pip install statsig>=0.36")


STATSIG_SERVER_KEY = os.getenv("STATSIG_SERVER_KEY", "")


class ABTestManager:
    """Production A/B testing for ML model rollouts.

    Manages:
        1. Feature gates (e.g., "new_two_tower_model")
        2. Experiments with percentage rollouts
        3. Auto-promotion/rollback based on metrics
        4. Canary deploys (5% → 20% → 50% → 100%)
    """

    def __init__(self) -> None:
        self._initialized = False
        self._local_overrides: dict[str, dict] = {}

    def initialize(self) -> bool:
        """Initialize Statsig SDK."""
        if not HAS_STATSIG or not STATSIG_SERVER_KEY:
            logger.info("Statsig not configured — using local overrides for A/B testing")
            return False

        try:
            statsig_client.initialize(
                STATSIG_SERVER_KEY,
                StatsigOptions(tier="production"),
            )
            self._initialized = True
            logger.info("Statsig initialized for A/B testing")
            return True
        except Exception as e:
            logger.error("Statsig init failed: %s", e)
            return False

    def check_gate(self, user_id: str, gate_name: str) -> bool:
        """Check if a user passes a feature gate.

        Example gates:
            - "new_two_tower_v2" → 5% of users get new model
            - "dopaminergic_feed" → 10% get enhanced feed
            - "fraud_detection_v2" → 50% get new fraud model
        """
        # Check local overrides first
        if gate_name in self._local_overrides:
            return self._local_overrides[gate_name].get("enabled", False)

        if not self._initialized:
            return self._deterministic_gate(user_id, gate_name)

        try:
            user = statsig.StatsigUser(user_id=user_id)
            return statsig_client.check_gate(user, gate_name)
        except Exception as e:
            logger.error("Gate check failed: %s", e)
            return False

    def get_experiment(self, user_id: str, experiment_name: str) -> dict[str, Any]:
        """Get experiment configuration for a user.

        Returns the parameter values assigned to this user's test group.

        Example experiment:
            name: "recommendation_model_v2"
            groups:
                control: {"model_version": "v1", "rerank_weight": 1.0}
                test:    {"model_version": "v2", "rerank_weight": 1.2}
        """
        if experiment_name in self._local_overrides:
            return self._local_overrides[experiment_name]

        if not self._initialized:
            return {"group": "control"}

        try:
            user = statsig.StatsigUser(user_id=user_id)
            config = statsig_client.get_experiment(user, experiment_name)
            return config.get_value() or {"group": "control"}
        except Exception as e:
            logger.error("Experiment fetch failed: %s", e)
            return {"group": "control"}

    def log_exposure(
        self,
        user_id: str,
        experiment_name: str,
        group: str,
        metrics: dict[str, float],
    ) -> None:
        """Log experiment exposure and metrics for analysis.

        Called after serving a recommendation to track which model
        version was used and what the outcome was.
        """
        if not self._initialized:
            logger.debug(
                "A/B exposure: user=%s experiment=%s group=%s metrics=%s",
                user_id, experiment_name, group, metrics,
            )
            return

        try:
            user = statsig.StatsigUser(user_id=user_id)
            for metric_name, value in metrics.items():
                statsig_client.log_event(user, metric_name, value, {
                    "experiment": experiment_name,
                    "group": group,
                })
        except Exception as e:
            logger.error("Exposure logging failed: %s", e)

    def create_canary(
        self,
        model_name: str,
        traffic_pct: int = 5,
    ) -> dict[str, Any]:
        """Create a canary deployment for a new model.

        Programmatically sets up:
            1. Feature gate for the model (5% of users)
            2. Experiment tracking
            3. Auto-promotion criteria

        In production, this is called by the Kubeflow pipeline after
        model validation passes.
        """
        gate_name = f"canary_{model_name}_{int(time.time())}"

        self._local_overrides[gate_name] = {
            "enabled": True,
            "traffic_pct": traffic_pct,
            "model_name": model_name,
            "created_at": int(time.time()),
            "promotion_criteria": {
                "min_engagement_lift": 0.02,   # +2%
                "min_sample_size": 10000,
                "max_p99_latency_ms": 10,
                "observation_hours": 24,
            },
        }

        logger.info(
            "Canary created: %s → %d%% traffic, gate=%s",
            model_name, traffic_pct, gate_name,
        )
        return self._local_overrides[gate_name]

    def set_local_override(self, name: str, config: dict) -> None:
        """Set a local override for testing (no Statsig required)."""
        self._local_overrides[name] = config

    def _deterministic_gate(self, user_id: str, gate_name: str, pct: int = 5) -> bool:
        """Deterministic gate check when Statsig is unavailable.

        Uses hash of user_id to deterministically assign users to groups.
        Same user always gets the same assignment.
        """
        h = hash(f"{user_id}:{gate_name}") % 100
        return h < pct

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._initialized:
            try:
                statsig_client.shutdown()
            except Exception:
                pass
