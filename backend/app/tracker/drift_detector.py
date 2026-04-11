"""
Explanation Drift Detector
===========================
Detects when a learner's explanation has changed significantly between
consecutive predictions using Jensen-Shannon Divergence on SHAP vectors
and top-feature rank shift analysis.

Public API
----------
ExplanationDriftDetector(threshold_jsd=0.15, threshold_rank_shift=1)
    .compare(prev_shap, curr_shap) -> DriftResult
    .check_learner_drift(learner_id, store) -> DriftResult | None
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax

log = logging.getLogger(__name__)


@dataclass
class DriftResult:
    jsd:             float   # Jensen-Shannon Divergence
    rank_shift:      int     # number of features that changed in top-3
    drift_detected:  bool
    flag:            str     # "none" | "mild" | "severe"
    prev_top3:       list[str]
    curr_top3:       list[str]
    details:         str

    def to_dict(self) -> dict:
        return {
            "jsd":            round(self.jsd, 6),
            "rank_shift":     self.rank_shift,
            "drift_detected": self.drift_detected,
            "flag":           self.flag,
            "prev_top3":      self.prev_top3,
            "curr_top3":      self.curr_top3,
            "details":        self.details,
        }


class ExplanationDriftDetector:
    def __init__(
        self,
        threshold_jsd: float = 0.15,
        threshold_rank_shift: int = 1,
    ):
        self.threshold_jsd        = threshold_jsd
        self.threshold_rank_shift = threshold_rank_shift

    def _shap_to_distribution(self, shap_values: dict[str, float]) -> np.ndarray:
        """Convert SHAP values to a probability distribution via softmax on |SHAP|."""
        vals = np.array([abs(v) for v in shap_values.values()], dtype=np.float64)
        if vals.sum() < 1e-10:
            return np.ones(len(vals)) / len(vals)
        return softmax(vals)

    def _get_top3(self, shap_values: dict[str, float]) -> list[str]:
        return sorted(shap_values, key=lambda k: abs(shap_values[k]), reverse=True)[:3]

    def compare(
        self,
        prev_shap: dict[str, float],
        curr_shap: dict[str, float],
    ) -> DriftResult:
        # Ensure both dicts have the same keys
        all_keys = sorted(set(prev_shap.keys()) | set(curr_shap.keys()))
        p_aligned = {k: prev_shap.get(k, 0.0) for k in all_keys}
        c_aligned = {k: curr_shap.get(k, 0.0) for k in all_keys}

        p_dist = self._shap_to_distribution(p_aligned)
        c_dist = self._shap_to_distribution(c_aligned)

        jsd = float(jensenshannon(p_dist, c_dist))

        prev_top3 = self._get_top3(p_aligned)
        curr_top3 = self._get_top3(c_aligned)
        rank_shift = len(set(curr_top3) - set(prev_top3))

        jsd_drift  = jsd > self.threshold_jsd
        rank_drift = rank_shift > self.threshold_rank_shift
        drift_detected = jsd_drift or rank_drift

        if jsd_drift and rank_drift:
            flag = "severe"
        elif jsd_drift or rank_drift:
            flag = "mild"
        else:
            flag = "none"

        details_parts = []
        if jsd_drift:
            details_parts.append(f"JSD={jsd:.4f} exceeds threshold {self.threshold_jsd}")
        if rank_drift:
            details_parts.append(f"Top-3 rank shift={rank_shift} (changed: {set(curr_top3) - set(prev_top3)})")
        if not details_parts:
            details_parts.append("No significant drift detected")

        return DriftResult(
            jsd=jsd,
            rank_shift=rank_shift,
            drift_detected=drift_detected,
            flag=flag,
            prev_top3=prev_top3,
            curr_top3=curr_top3,
            details="; ".join(details_parts),
        )

    def check_learner_drift(
        self, learner_id: str, store
    ) -> Optional[DriftResult]:
        """
        Retrieve the last 2 explanation records for a learner and compare.
        Returns None if fewer than 2 records exist.
        """
        history = store.get_history(learner_id, n=2)
        if len(history) < 2:
            return None

        curr_shap = history[0]["shap_values"]  # most recent
        prev_shap = history[1]["shap_values"]

        if isinstance(curr_shap, str):
            import json
            curr_shap = json.loads(curr_shap)
        if isinstance(prev_shap, str):
            import json
            prev_shap = json.loads(prev_shap)

        return self.compare(prev_shap, curr_shap)
