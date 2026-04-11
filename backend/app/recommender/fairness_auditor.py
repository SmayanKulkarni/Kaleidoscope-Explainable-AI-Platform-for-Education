"""
Fairness Auditor (Feature 12)
==============================
Checks whether recommendation scores systematically differ across protected
demographic groups (explicit_gender, explicit_age_band, explicit_disability).

A disparity is flagged when any group's mean score deviates > 15 % from the
overall mean score across all scored items.

Public API
----------
FairnessAuditor(feature_columns, protected_features, disparity_threshold)
    .audit(items, scores) -> FairnessReport
    FairnessReport.to_dict()  -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FairnessReport:
    overall_fair:        bool
    group_scores:        Dict[str, Dict[str, float]]  # feature → {group_value → mean_score}
    flagged_disparities: List[Dict]                   # list of flagged group dicts

    def to_dict(self) -> dict:
        return {
            "overall_fair":        self.overall_fair,
            "group_scores":        self.group_scores,
            "flagged_disparities": self.flagged_disparities,
        }


# ──────────────────────────────────────────────────────────────────────────────
# FairnessAuditor
# ──────────────────────────────────────────────────────────────────────────────

class FairnessAuditor:
    """
    Audits recommendation scores for group fairness across protected attributes.

    Parameters
    ----------
    feature_columns      : list of all feature names expected in items
    protected_features   : list of feature names to audit for bias
    disparity_threshold  : max allowed relative deviation from overall mean
                           (default 0.15 = 15 %)
    """

    DEFAULT_PROTECTED = ["explicit_gender", "explicit_age_band", "explicit_disability"]

    def __init__(
        self,
        feature_columns:     Optional[List[str]] = None,
        protected_features:  Optional[List[str]] = None,
        disparity_threshold: float = 0.15,
    ):
        self.feature_columns     = feature_columns or []
        self.protected_features  = protected_features or self.DEFAULT_PROTECTED
        self.disparity_threshold = disparity_threshold
        log.info(
            "FairnessAuditor initialised  protected=%s  threshold=%.0f%%",
            self.protected_features,
            disparity_threshold * 100,
        )

    def audit(self, items: List[Dict], scores: List[float]) -> FairnessReport:
        """
        Audit recommendation scores across protected groups.

        Parameters
        ----------
        items  : list of feature dicts (one per scored item)
        scores : list of float scores aligned index-for-index with items

        Returns
        -------
        FairnessReport
        """
        if not items or not scores or len(items) != len(scores):
            return FairnessReport(
                overall_fair=True,
                group_scores={},
                flagged_disparities=[],
            )

        overall_mean = sum(scores) / len(scores)
        group_scores: Dict[str, Dict[str, float]] = {}
        flagged: List[Dict] = []

        for feat in self.protected_features:
            # Skip features absent from all items
            if not any(feat in it for it in items):
                continue

            # Bucket scores by this feature's value
            groups: Dict[str, List[float]] = {}
            for item, score in zip(items, scores):
                val = item.get(feat)
                if val is None:
                    continue
                key = str(val)
                groups.setdefault(key, []).append(score)

            if not groups:
                continue

            group_means = {g: sum(vs) / len(vs) for g, vs in groups.items()}
            group_scores[feat] = {g: round(v, 5) for g, v in group_means.items()}

            # Flag groups whose mean deviates beyond the threshold
            for group, mean in group_means.items():
                deviation = (
                    (mean - overall_mean) / abs(overall_mean)
                    if overall_mean != 0
                    else 0.0
                )
                if abs(deviation) > self.disparity_threshold:
                    flagged.append({
                        "feature":       feat,
                        "group":         group,
                        "score":         round(mean, 5),
                        "deviation_pct": round(deviation * 100, 2),
                        "direction":     "above_mean" if deviation > 0 else "below_mean",
                    })

        return FairnessReport(
            overall_fair=len(flagged) == 0,
            group_scores=group_scores,
            flagged_disparities=flagged,
        )
