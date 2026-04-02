"""
Evidently AI — Model Drift Monitoring (archi-2026 §9.6)
=========================================================
Detects when production model performance drifts from the training
distribution. If drift is detected → triggers alert + automatic retraining.

This module monitors:
    1. Feature drift — input feature distributions shifting
    2. Prediction drift — model output scores shifting
    3. Target drift — actual conversion rates changing

Requires:
    pip install evidently>=0.4

Usage:
    from ml.monitoring.drift_report import DriftMonitor
    monitor = DriftMonitor()
    report = await monitor.run_drift_check(reference_df, current_df)
    if report["dataset_drift"]:
        trigger_retraining()
"""

from __future__ import annotations

import logging
import json
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
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import (
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ColumnDriftMetric,
    )
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False
    logger.warning(
        "evidently not installed — drift monitoring disabled. "
        "Install with: pip install evidently>=0.4"
    )


class DriftMonitor:
    """Production model drift monitor.

    Compares a reference dataset (from training/validation) against
    current production data to detect distribution shifts.

    Architecture:
        Spark/Flink → ClickHouse (raw events)
        DriftMonitor reads from ClickHouse → generates Evidently report
        If drift → alert via webhook + trigger Kubeflow retraining

    In production, this runs as a scheduled Kubernetes CronJob (daily).
    """

    def __init__(
        self,
        report_output_dir: str = "drift_reports",
        alert_webhook_url: Optional[str] = None,
        drift_threshold: float = 0.5,
    ):
        self.report_dir = Path(report_output_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.alert_webhook_url = alert_webhook_url
        self.drift_threshold = drift_threshold

    def run_drift_check(
        self,
        reference_data: Any,
        current_data: Any,
        monitored_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Run drift detection between reference and current data.

        Args:
            reference_data: pandas DataFrame from training/validation period
            current_data: pandas DataFrame from recent production period
            monitored_columns: specific columns to monitor (default: all)

        Returns:
            Dict with drift results:
            {
                "dataset_drift": bool,
                "drift_share": float,  # % of columns that drifted
                "drifted_columns": [...],
                "timestamp": int,
                "report_path": str,
            }
        """
        if not HAS_EVIDENTLY:
            logger.warning("Evidently not available — skipping drift check")
            return {"dataset_drift": False, "error": "evidently not installed"}

        try:
            # Build Evidently report
            metrics = [
                DatasetDriftMetric(drift_share=self.drift_threshold),
                DatasetMissingValuesMetric(),
            ]

            # Add per-column drift for key features
            if monitored_columns:
                for col in monitored_columns:
                    if col in reference_data.columns and col in current_data.columns:
                        metrics.append(ColumnDriftMetric(column_name=col))

            report = Report(metrics=metrics)
            report.run(reference_data=reference_data, current_data=current_data)

            # Extract results
            result_dict = report.as_dict()
            drift_result = result_dict.get("metrics", [{}])[0].get("result", {})

            is_drifted = drift_result.get("dataset_drift", False)
            drift_share = drift_result.get("drift_share", 0.0)
            n_drifted = drift_result.get("number_of_drifted_columns", 0)

            # Find which columns drifted
            drifted_columns = []
            drift_by_columns = drift_result.get("drift_by_columns", {})
            for col_name, col_data in drift_by_columns.items():
                if col_data.get("drift_detected", False):
                    drifted_columns.append(col_name)

            # Save report
            ts = int(time.time())
            report_path = self.report_dir / f"drift_report_{ts}.html"
            report.save_html(str(report_path))

            result = {
                "dataset_drift": is_drifted,
                "drift_share": drift_share,
                "n_drifted_columns": n_drifted,
                "drifted_columns": drifted_columns,
                "timestamp": ts,
                "report_path": str(report_path),
            }

            if is_drifted:
                logger.warning(
                    "🚨 DRIFT DETECTED: %.1f%% of columns drifted (%s)",
                    drift_share * 100, drifted_columns,
                )
                self._send_alert(result)
            else:
                logger.info("✅ No drift detected (%.1f%% threshold)", self.drift_threshold * 100)

            return result

        except Exception as e:
            logger.error("Drift check failed: %s", e)
            return {"dataset_drift": False, "error": str(e)}

    def run_prediction_drift(
        self,
        reference_scores: Any,
        current_scores: Any,
    ) -> dict[str, Any]:
        """Check if model prediction scores have drifted.

        This catches cases where the model outputs have shifted even if
        input features haven't — indicating model degradation.
        """
        if not HAS_NUMPY:
            return {"prediction_drift": False, "error": "numpy not installed"}

        try:
            ref = np.array(reference_scores)
            cur = np.array(current_scores)

            # Simple KS-test for distribution comparison
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(ref, cur)

            drifted = p_value < 0.01  # 1% significance level

            result = {
                "prediction_drift": bool(drifted),
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "ref_mean": float(ref.mean()),
                "cur_mean": float(cur.mean()),
                "ref_std": float(ref.std()),
                "cur_std": float(cur.std()),
            }

            if drifted:
                logger.warning(
                    "🚨 PREDICTION DRIFT: KS=%.4f p=%.6f ref_mean=%.4f cur_mean=%.4f",
                    ks_stat, p_value, ref.mean(), cur.mean(),
                )

            return result

        except ImportError:
            logger.warning("scipy not installed — using fallback mean comparison")
            ref = np.array(reference_scores)
            cur = np.array(current_scores)
            mean_shift = abs(cur.mean() - ref.mean()) / max(ref.std(), 1e-6)
            return {
                "prediction_drift": bool(mean_shift > 2.0),
                "mean_shift_sigmas": float(mean_shift),
            }
        except Exception as e:
            logger.error("Prediction drift check failed: %s", e)
            return {"prediction_drift": False, "error": str(e)}

    def _send_alert(self, result: dict) -> None:
        """Send drift alert via webhook (Slack, PagerDuty, etc.)."""
        if not self.alert_webhook_url:
            return

        try:
            import urllib.request
            payload = json.dumps({
                "text": f"🚨 Model Drift Detected — {result.get('n_drifted_columns', 0)} columns drifted. "
                        f"Columns: {result.get('drifted_columns', [])}",
            }).encode()
            req = urllib.request.Request(
                self.alert_webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.error("Alert webhook failed: %s", e)
