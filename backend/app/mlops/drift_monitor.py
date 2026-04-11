"""
Data Drift Monitor (Evidently)
===============================
Compares recent prediction inputs against the training distribution
using Evidently's DataDriftPreset.

Public API
----------
DriftMonitor(reference_data, feature_names)
    .check_drift(current_data) -> DriftReport
    .get_cached_report() -> DriftReport | None
"""

from __future__ import annotations

import collections
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Configurable via environment variables ─────────────────────────────────────
CACHE_TTL_SECONDS    = int(os.getenv("DRIFT_CACHE_TTL_SECONDS", "3600"))   # default 1h
DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.20"))  # alert if ≥20% features drift
TREND_WINDOW          = int(os.getenv("DRIFT_TREND_WINDOW", "5"))          # last N reports for trend
_SLACK_WEBHOOK        = os.getenv("SLACK_DRIFT_WEBHOOK_URL", "")


@dataclass
class FeatureDrift:
    feature:        str
    drift_detected: bool
    statistic:      float
    p_value:        float
    method:         str

    def to_dict(self) -> dict:
        return {
            "feature":        self.feature,
            "drift_detected": self.drift_detected,
            "statistic":      round(self.statistic, 6),
            "p_value":        round(self.p_value, 6),
            "method":         self.method,
        }


@dataclass
class DriftReport:
    dataset_drift:       bool
    n_drifted_features:  int
    n_total_features:    int
    drift_share:         float
    feature_drifts:      list[FeatureDrift]
    generated_at:        float = field(default_factory=time.time)
    alert_fired:         bool  = False
    trend_direction:     Optional[str] = None    # "increasing" | "stable" | "decreasing"

    def to_dict(self) -> dict:
        return {
            "dataset_drift":      self.dataset_drift,
            "n_drifted_features": self.n_drifted_features,
            "n_total_features":   self.n_total_features,
            "drift_share":        round(self.drift_share, 4),
            "feature_drifts":     [fd.to_dict() for fd in self.feature_drifts],
            "alert_fired":        self.alert_fired,
            "trend_direction":    self.trend_direction,
        }


AlertHook = Callable[["DriftReport"], None]


def _slack_alert_hook(report: "DriftReport") -> None:
    """Default alert hook — posts to Slack webhook if SLACK_DRIFT_WEBHOOK_URL is set."""
    if not _SLACK_WEBHOOK:
        return
    try:
        import json, urllib.request
        drifted = [
            fd.feature for fd in report.feature_drifts if fd.drift_detected
        ]
        msg = {
            "text": (
                f":warning: *XAI Drift Alert*\n"
                f"drift_share={report.drift_share:.1%}  "
                f"drifted_features={drifted}\n"
                f"trend={report.trend_direction}  "
                f"threshold={DRIFT_SHARE_THRESHOLD:.0%}"
            )
        }
        req = urllib.request.Request(
            _SLACK_WEBHOOK,
            data=json.dumps(msg).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
        log.info("Drift alert sent to Slack")
    except Exception as e:
        log.warning("Slack drift alert failed: %s", e)


class DriftMonitor:
    def __init__(
        self,
        reference_data: np.ndarray | pd.DataFrame,
        feature_names: list[str],
        alert_hook: Optional[AlertHook] = None,
    ):
        if isinstance(reference_data, np.ndarray):
            self.reference_df = pd.DataFrame(reference_data, columns=feature_names)
        else:
            self.reference_df = reference_data.copy()
        self.feature_names   = feature_names
        self._cached_report: Optional[DriftReport] = None
        self._alert_hook     = alert_hook or _slack_alert_hook
        self._report_history: collections.deque[DriftReport] = collections.deque(
            maxlen=TREND_WINDOW
        )
        log.info(
            "DriftMonitor initialised  reference_size=%d  features=%d  "
            "ttl=%ds  alert_threshold=%.0f%%",
            len(self.reference_df), len(feature_names),
            CACHE_TTL_SECONDS, DRIFT_SHARE_THRESHOLD * 100,
        )

    def check_drift(
        self, current_data: np.ndarray | pd.DataFrame
    ) -> DriftReport:
        if isinstance(current_data, np.ndarray):
            current_df = pd.DataFrame(current_data, columns=self.feature_names)
        else:
            current_df = current_data.copy()

        if current_df.empty:
            return DriftReport(
                dataset_drift=False,
                n_drifted_features=0,
                n_total_features=len(self.feature_names),
                drift_share=0.0,
                feature_drifts=[],
            )

        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_df,
            current_data=current_df,
        )

        report_dict = report.as_dict()

        # Parse Evidently's report structure
        feature_drifts = []
        dataset_drift = False
        n_drifted = 0

        results = report_dict.get("metrics", [])
        for metric_result in results:
            metric_data = metric_result.get("result", {})

            # Dataset-level drift
            if "dataset_drift" in metric_data:
                dataset_drift = metric_data["dataset_drift"]
                n_drifted = metric_data.get("number_of_drifted_columns", 0)

            # Per-feature drift
            drift_by_columns = metric_data.get("drift_by_columns", {})
            for fname, fdata in drift_by_columns.items():
                if fname in self.feature_names:
                    feature_drifts.append(FeatureDrift(
                        feature=fname,
                        drift_detected=fdata.get("drift_detected", False),
                        statistic=fdata.get("stattest_value", 0.0) or 0.0,
                        p_value=fdata.get("drift_score", 1.0) or 1.0,
                        method=fdata.get("stattest_name", "unknown"),
                    ))

        drift_share = n_drifted / max(len(self.feature_names), 1)

        trend = self._compute_trend(drift_share)
        result = DriftReport(
            dataset_drift=dataset_drift,
            n_drifted_features=n_drifted,
            n_total_features=len(self.feature_names),
            drift_share=drift_share,
            feature_drifts=feature_drifts,
            trend_direction=trend,
        )

        # ── Alert hook ───────────────────────────────────────────────────────
        if drift_share >= DRIFT_SHARE_THRESHOLD:
            result.alert_fired = True
            try:
                self._alert_hook(result)
            except Exception as e:
                log.error("Drift alert hook error: %s", e)

        self._report_history.append(result)
        self._cached_report = result
        return result

    def _compute_trend(self, current_share: float) -> str:
        """
        Compare current drift share against the rolling window.
        Returns "increasing", "decreasing", or "stable".
        """
        if len(self._report_history) < 2:
            return "stable"
        past_avg = sum(r.drift_share for r in self._report_history) / len(self._report_history)
        delta = current_share - past_avg
        if delta > 0.05:
            return "increasing"
        if delta < -0.05:
            return "decreasing"
        return "stable"

    def trend_summary(self) -> dict:
        """Return drift share history and trend direction for the last TREND_WINDOW reports."""
        shares = [round(r.drift_share, 4) for r in self._report_history]
        return {
            "window":        TREND_WINDOW,
            "drift_shares":  shares,
            "trend":         self._compute_trend(shares[-1] if shares else 0.0),
            "threshold":     DRIFT_SHARE_THRESHOLD,
        }

    def get_cached_report(self) -> Optional[DriftReport]:
        if self._cached_report is None:
            return None
        age = time.time() - self._cached_report.generated_at
        if age > CACHE_TTL_SECONDS:
            return None
        return self._cached_report

    def concept_drift_proxy(
        self,
        recent_risk_scores: list[float],
        recent_followed: list[bool],
    ) -> dict:
        """
        Lightweight concept-drift proxy using recommendation follow rate
        and mean risk score calibration shift.

        Parameters
        ----------
        recent_risk_scores : list of recent predicted risk scores
        recent_followed    : list of bool (did learner follow recommendation?)

        Returns
        -------
        dict with follow_rate, mean_risk, calibration_gap, drift_proxy_flag
        """
        if not recent_risk_scores or not recent_followed:
            return {"drift_proxy_flag": False, "message": "Insufficient feedback data"}

        follow_rate   = sum(recent_followed) / len(recent_followed)
        mean_risk     = float(np.mean(recent_risk_scores))
        # Expected: high risk ↔ low follow rate; low risk ↔ high follow rate
        # Calibration gap: if model says high risk but people follow at high rate,
        # model may be over-predicting risk
        calibration_gap = abs(mean_risk - (1.0 - follow_rate))
        flag = calibration_gap > 0.25

        return {
            "follow_rate":       round(follow_rate, 4),
            "mean_risk":         round(mean_risk, 4),
            "calibration_gap":   round(calibration_gap, 4),
            "drift_proxy_flag":  flag,
            "interpretation":    (
                "Possible concept drift: model risk scores misaligned with observed behaviour"
                if flag else "Calibration within expected range"
            ),
        }
