"""
LightGBM Fraud Detection (archi-2026 §9.5)
=============================================
Detects bots, fake likes, spam comments, fake followers, card fraud,
and agentic AI attacks using gradient-boosted trees + rule-based fallback.

Architecture:
    Flink computes real-time fraud features → LightGBM scores each user
    → If score > 0.9 → shadowban + investigation
    → If score > 0.7 → captcha challenge

2026 Best Practices (source: SEON, Sift, Stripe Radar):
    - Behavioral biometrics (touch pressure, scroll speed, session patterns)
    - Advanced device fingerprinting (emulator, VPN, rooted device)
    - Velocity checks (orders/h, cards/week, addresses/week)
    - Graph-based entity resolution (shared devices/IPs across accounts)

Requires:
    pip install lightgbm>=4.3
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("lightgbm not installed — fraud detection uses rule-based fallback. pip install lightgbm>=4.3")


# ── Feature Definitions ───────────────────────────────────────
# Each feature MUST be sent by the app (React Native / Web).
# Features are grouped by signal type for clarity.

FRAUD_FEATURES = [
    # ── Short-term behavior (last 10 minutes, computed by Flink) ──
    "likes_per_minute",             # Normal < 5, bot > 30
    "comments_per_minute",          # Normal < 2, spam bot > 10
    "follows_per_minute",           # Normal < 1, follow farm > 20
    "unique_creators_liked",        # Low diversity = bot
    "action_interval_std",          # Very low std = automated
    "unique_ips_10m",               # Multiple IPs = suspicious

    # ── Account signals ──
    "account_age_hours",            # New accounts are riskier
    "profile_completeness",         # 0.0-1.0, empty = suspicious
    "has_avatar",                   # Binary
    "bio_length",                   # Short bio = less legitimate
    "followers_count",              # Context for behavior rate
    "following_count",              # Follow farms have high following

    # ── Device signals (sent by app) ──
    "device_fingerprint_reuse",     # Same device, multiple accounts
    "accounts_same_ip_24h",         # IP sharing
    "is_emulator",                  # Emulators are suspicious
    "is_vpn_or_proxy",              # VPN/proxy detected (via IP geolocation)
    "is_rooted_jailbroken",         # Rooted/jailbroken device
    "timezone_mismatch",            # GPS timezone ≠ device timezone
    "battery_level",                # Emulators often report 100%
    "app_install_age_hours",        # Very fresh installs are riskier
    "screen_resolution_common",     # Unusual resolutions = emulator

    # ── Behavioral biometrics (sent by app, 2026 best practice) ──
    "touch_pressure_std",           # Bots have zero variance
    "scroll_speed_avg",             # Bots scroll unnaturally fast
    "session_duration_seconds",     # Very short sessions = bot
    "screen_tap_interval_std",      # Too regular = automated
    "typing_speed_wpm",             # Unnaturally fast = paste bot
    "navigation_depth",             # How deep into the app (bots are shallow)

    # ── Content signals ──
    "comment_duplicate_rate",       # Same comment copy-pasted
    "like_without_view_rate",       # Liking without watching = bot

    # ── Transaction/velocity signals (for checkout fraud) ──
    "orders_24h",                   # Rapid ordering = card fraud
    "addresses_7d",                 # Multiple shipping addresses
    "payment_methods_7d",           # Multiple cards = testing stolen cards
    "failed_payments_1h",           # Failed payment velocity
    "avg_order_value_deviation",    # Sudden high-value orders vs history
    "shipping_billing_distance_km", # Far shipping from billing = suspicious
]


# Thresholds for rule-based fallback (when LightGBM not available)
RULE_THRESHOLDS = {
    # Behavior
    "likes_per_minute": 30,
    "comments_per_minute": 10,
    "follows_per_minute": 20,
    "action_interval_std": 0.05,
    # Device
    "device_fingerprint_reuse": 5,
    "accounts_same_ip_24h": 10,
    "is_emulator": 0.5,            # Binary: 1 = emulator
    "is_vpn_or_proxy": 0.5,        # Binary: 1 = VPN
    "is_rooted_jailbroken": 0.5,   # Binary: 1 = rooted
    "battery_level": 99.5,         # Always 100% = emulator
    "app_install_age_hours": 0.5,  # Less than 30 min = very suspicious
    # Behavioral biometrics
    "touch_pressure_std": 0.001,   # Near-zero variance = bot
    "scroll_speed_avg": 5000,      # Pixels/sec — inhuman speed
    "screen_tap_interval_std": 0.01, # Too regular = bot
    # Content
    "comment_duplicate_rate": 0.8,
    "like_without_view_rate": 0.9,
    # Transaction
    "orders_24h": 10,              # 10+ orders in 24h
    "addresses_7d": 5,             # 5+ different addresses in a week
    "payment_methods_7d": 4,       # 4+ cards in a week
    "failed_payments_1h": 5,       # 5+ failed payments in an hour
}


class FraudDetector:
    """Production fraud detection using LightGBM + rule-based fallback.

    In production:
        1. Flink streaming job computes fraud features per user per 10-min window
        2. FraudDetector.predict() scores each user
        3. Score > 0.9 → shadowban (user sees own posts, nobody else sees them)
        4. Score > 0.7 → captcha challenge
        5. Score > 0.5 → flag for human review
    """

    # Fraud action thresholds
    SHADOWBAN_THRESHOLD = 0.9
    CAPTCHA_THRESHOLD = 0.7
    REVIEW_THRESHOLD = 0.5

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[Any] = None
        self._model_path = model_path

        if model_path and HAS_LIGHTGBM:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """Load a trained LightGBM model from disk."""
        if not HAS_LIGHTGBM:
            return

        path = Path(model_path)
        if not path.exists():
            logger.warning("Fraud model not found at %s — using rule-based fallback", model_path)
            return

        try:
            self.model = lgb.Booster(model_file=str(path))
            logger.info("Fraud model loaded: %s", model_path)
        except Exception as e:
            logger.error("Failed to load fraud model: %s", e)

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Score a user for fraud probability.

        Args:
            features: dict of fraud feature values (must match FRAUD_FEATURES).
                      The app (React Native/Web) sends these signals with
                      every session or at key checkpoints.

        Returns:
            {
                "fraud_score": float,  # 0.0 (clean) to 1.0 (definitely bot)
                "action": str,         # "allow" | "captcha" | "shadowban" | "review"
                "triggered_rules": [...],  # which rules fired
                "risk_factors": {...},     # categorized risk breakdown
            }
        """
        if self.model and HAS_LIGHTGBM and HAS_NUMPY:
            return self._predict_lgb(features)
        else:
            return self._predict_rules(features)

    def _predict_lgb(self, features: dict[str, float]) -> dict[str, Any]:
        """LightGBM model prediction."""
        feature_vec = np.array([[features.get(f, 0.0) for f in FRAUD_FEATURES]])
        score = float(self.model.predict(feature_vec)[0])

        action = self._score_to_action(score)
        risk_factors = self._compute_risk_factors(features)

        return {
            "fraud_score": round(score, 4),
            "action": action,
            "method": "lightgbm",
            "triggered_rules": [],
            "risk_factors": risk_factors,
        }

    def _predict_rules(self, features: dict[str, float]) -> dict[str, Any]:
        """Rule-based fallback — no ML model needed."""
        triggered = []
        for rule, threshold in RULE_THRESHOLDS.items():
            value = features.get(rule, 0.0)
            # Special handling for "low value = suspicious" rules
            if rule in ("touch_pressure_std", "screen_tap_interval_std", "action_interval_std", "app_install_age_hours"):
                if value > 0 and value < threshold:
                    triggered.append(f"{rule}={value:.4f} (<{threshold})")
            elif value > threshold:
                triggered.append(f"{rule}={value:.2f} (>{threshold})")

        # Heuristic score based on number of triggered rules
        n = len(triggered)
        if n == 0:
            score = 0.0
        elif n == 1:
            score = 0.3
        elif n == 2:
            score = 0.5
        elif n == 3:
            score = 0.7
        elif n == 4:
            score = 0.85
        else:
            score = min(0.98, 0.85 + n * 0.03)

        # Automatic escalation for extreme values
        if features.get("likes_per_minute", 0) > 100:
            score = max(score, 0.95)
            triggered.append("EXTREME: likes_per_minute > 100")
        if features.get("device_fingerprint_reuse", 0) > 20:
            score = max(score, 0.95)
            triggered.append("EXTREME: device_fingerprint_reuse > 20")
        if features.get("failed_payments_1h", 0) > 10:
            score = max(score, 0.95)
            triggered.append("EXTREME: failed_payments_1h > 10")
        if features.get("is_emulator", 0) > 0.5 and features.get("is_vpn_or_proxy", 0) > 0.5:
            score = max(score, 0.90)
            triggered.append("COMBO: emulator + VPN")
        if features.get("touch_pressure_std", 1.0) == 0.0 and features.get("session_duration_seconds", 999) < 5:
            score = max(score, 0.95)
            triggered.append("BOT: zero touch pressure + instant session")

        risk_factors = self._compute_risk_factors(features)
        action = self._score_to_action(score)

        return {
            "fraud_score": round(score, 4),
            "action": action,
            "method": "rules",
            "triggered_rules": triggered,
            "risk_factors": risk_factors,
        }

    def _score_to_action(self, score: float) -> str:
        if score >= self.SHADOWBAN_THRESHOLD:
            return "shadowban"
        elif score >= self.CAPTCHA_THRESHOLD:
            return "captcha"
        elif score >= self.REVIEW_THRESHOLD:
            return "review"
        else:
            return "allow"

    @staticmethod
    def _compute_risk_factors(features: dict[str, float]) -> dict[str, str]:
        """Categorize risk factors for dashboard display."""
        factors = {}

        # Device risk
        device_risk = 0
        if features.get("is_emulator", 0) > 0.5:
            device_risk += 3
        if features.get("is_vpn_or_proxy", 0) > 0.5:
            device_risk += 2
        if features.get("is_rooted_jailbroken", 0) > 0.5:
            device_risk += 2
        if features.get("device_fingerprint_reuse", 0) > 3:
            device_risk += 3
        factors["device"] = "high" if device_risk >= 5 else "medium" if device_risk >= 2 else "low"

        # Behavior risk
        behavior_risk = 0
        if features.get("likes_per_minute", 0) > 30:
            behavior_risk += 3
        if features.get("action_interval_std", 1.0) < 0.05:
            behavior_risk += 3
        if features.get("touch_pressure_std", 1.0) < 0.01:
            behavior_risk += 3
        factors["behavior"] = "high" if behavior_risk >= 5 else "medium" if behavior_risk >= 2 else "low"

        # Transaction risk
        tx_risk = 0
        if features.get("orders_24h", 0) > 5:
            tx_risk += 2
        if features.get("failed_payments_1h", 0) > 3:
            tx_risk += 3
        if features.get("payment_methods_7d", 0) > 3:
            tx_risk += 3
        factors["transaction"] = "high" if tx_risk >= 5 else "medium" if tx_risk >= 2 else "low"

        return factors

    # ── Training ───────────────────────────────────────────────

    @staticmethod
    def train(
        train_data_path: str,
        output_model_path: str,
        params: Optional[dict] = None,
    ) -> dict[str, float]:
        """Train a LightGBM fraud detection model.

        Args:
            train_data_path: path to parquet/CSV with labeled fraud data
                Columns: FRAUD_FEATURES + "is_fraud" (0/1 label)
            output_model_path: where to save the trained model
            params: LightGBM parameters override

        Returns:
            dict of evaluation metrics
        """
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm required for training: pip install lightgbm>=4.3")

        import pandas as pd

        df = pd.read_parquet(train_data_path) if train_data_path.endswith(".parquet") \
            else pd.read_csv(train_data_path)

        feature_cols = [f for f in FRAUD_FEATURES if f in df.columns]
        X = df[feature_cols]
        y = df["is_fraud"]

        # Temporal split — last 20% for validation (never random split for time-series)
        split_idx = int(len(df) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)

        default_params = {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "scale_pos_weight": len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
            "verbosity": -1,
        }
        if params:
            default_params.update(params)

        model = lgb.train(
            default_params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        # Save model
        Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(output_model_path)

        # Evaluate
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        y_pred = model.predict(X_val)
        y_pred_binary = (y_pred > 0.5).astype(int)

        metrics = {
            "val_auc": float(roc_auc_score(y_val, y_pred)),
            "val_precision": float(precision_score(y_val, y_pred_binary, zero_division=0)),
            "val_recall": float(recall_score(y_val, y_pred_binary, zero_division=0)),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": len(feature_cols),
        }

        logger.info("Fraud model trained: %s", metrics)

        # Log to MLflow if available
        try:
            from ml.tracking import log_training_run
            log_training_run(
                model_name="fraud_lightgbm",
                config=default_params,
                metrics=metrics,
                model_path=output_model_path,
            )
        except Exception:
            pass

        return metrics
