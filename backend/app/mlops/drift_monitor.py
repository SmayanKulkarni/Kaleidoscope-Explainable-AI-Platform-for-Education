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

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 3600  # 1 hour


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

    def to_dict(self) -> dict:
        return {
            "dataset_drift":      self.dataset_drift,
            "n_drifted_features": self.n_drifted_features,
            "n_total_features":   self.n_total_features,
            "drift_share":        round(self.drift_share, 4),
            "feature_drifts":     [fd.to_dict() for fd in self.feature_drifts],
        }


class DriftMonitor:
    def __init__(
        self,
        reference_data: np.ndarray | pd.DataFrame,
        feature_names: list[str],
    ):
        if isinstance(reference_data, np.ndarray):
            self.reference_df = pd.DataFrame(reference_data, columns=feature_names)
        else:
            self.reference_df = reference_data.copy()
        self.feature_names = feature_names
        self._cached_report: Optional[DriftReport] = None
        log.info("DriftMonitor initialised  reference_size=%d  features=%d",
                 len(self.reference_df), len(feature_names))

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

        result = DriftReport(
            dataset_drift=dataset_drift,
            n_drifted_features=n_drifted,
            n_total_features=len(self.feature_names),
            drift_share=drift_share,
            feature_drifts=feature_drifts,
        )
        self._cached_report = result
        return result

    def get_cached_report(self) -> Optional[DriftReport]:
        if self._cached_report is None:
            return None
        age = time.time() - self._cached_report.generated_at
        if age > CACHE_TTL_SECONDS:
            return None
        return self._cached_report
