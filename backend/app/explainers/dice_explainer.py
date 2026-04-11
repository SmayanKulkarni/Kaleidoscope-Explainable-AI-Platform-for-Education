"""
DiCE Counterfactual Engine
===========================
Generates diverse counterfactual explanations via dice-ml.
Immutable features are auto-detected from the training data (zero-variance
in actionability terms) and also include domain-locked features.

Public API
----------
DiCEExplainer(model, X_train, feature_names, y_train)
    .get_counterfactuals(features_dict, n_cfs=3, desired_class=0)
        -> CounterfactualResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import dice_ml
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Features the learner cannot change (structural / historical)
IMMUTABLE_FEATURES = [
    "prior_course_completions",
    "current_week_in_course",
]

# Continuous feature ranges are learned from training data.
# These categorical hints are only for features that are truly discrete.
DISCRETE_FEATURES = [
    "forum_posts_count",
    "days_since_last_activity",
    "missed_deadlines_count",
    "help_requests_count",
    "prior_course_completions",
    "current_week_in_course",
]


@dataclass
class PrescriptiveAction:
    feature:       str
    current_value: float
    target_value:  float
    direction:     str        # "increase" | "decrease"
    magnitude:     float      # absolute change
    priority_rank: int = 0

    def to_dict(self) -> dict:
        return {
            "feature":       self.feature,
            "current_value": round(self.current_value, 4),
            "target_value":  round(self.target_value, 4),
            "direction":     self.direction,
            "magnitude":     round(self.magnitude, 4),
            "priority_rank": self.priority_rank,
        }


@dataclass
class CounterfactualResult:
    actions:          list[PrescriptiveAction]
    counterfactuals:  list[dict]
    best_cf:          Optional[dict]
    changed_features: dict[str, dict]   # {feat: {from, to}}
    n_generated:      int

    def to_dict(self) -> dict:
        return {
            "actions":          [a.to_dict() for a in self.actions],
            "counterfactuals":  self.counterfactuals,
            "best_cf":          self.best_cf,
            "changed_features": self.changed_features,
            "n_generated":      self.n_generated,
        }


class DiCEExplainer:
    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: list[str],
        y_train: np.ndarray,
    ):
        self.feature_names = feature_names
        self.model = model

        # Build a training DataFrame for dice-ml
        df_train = pd.DataFrame(X_train, columns=feature_names)
        df_train["dropout_risk"] = y_train.astype(int)

        # Infer continuous vs discrete per feature
        continuous = [f for f in feature_names if f not in DISCRETE_FEATURES]

        self.data_interface = dice_ml.Data(
            dataframe=df_train,
            continuous_features=continuous,
            outcome_name="dropout_risk",
        )

        self.model_interface = dice_ml.Model(
            model=model,
            backend="sklearn",
            model_type="classifier",
        )

        self.dice = dice_ml.Dice(
            data_interface=self.data_interface,
            model_interface=self.model_interface,
            method="random",
        )

        # Build permitted_range from actual training data (5th–95th percentile)
        self.permitted_range = {}
        for f in feature_names:
            if f not in IMMUTABLE_FEATURES:
                col = df_train[f]
                self.permitted_range[f] = [
                    float(np.percentile(col, 5)),
                    float(np.percentile(col, 95)),
                ]

        log.info("DiCEExplainer initialised  features=%d  immutable=%s",
                 len(feature_names), IMMUTABLE_FEATURES)

    def get_counterfactuals(
        self,
        features_dict: dict,
        n_cfs: int = 3,
        desired_class: int = 0,
    ) -> CounterfactualResult:
        """
        Generate diverse counterfactuals that flip the learner to `desired_class`
        (default: 0 = no dropout).
        """
        query = pd.DataFrame([{f: features_dict[f] for f in self.feature_names}])

        try:
            exp = self.dice.generate_counterfactuals(
                query_instances=query,
                total_CFs=n_cfs,
                desired_class=desired_class,
                features_to_vary=[f for f in self.feature_names if f not in IMMUTABLE_FEATURES],
                permitted_range=self.permitted_range,
            )
        except Exception as e:
            log.warning("DiCE generation failed: %s", e)
            return CounterfactualResult(
                actions=[], counterfactuals=[], best_cf=None,
                changed_features={}, n_generated=0,
            )

        cf_list = exp.cf_examples_list[0]
        if cf_list.final_cfs_df is None or cf_list.final_cfs_df.empty:
            return CounterfactualResult(
                actions=[], counterfactuals=[], best_cf=None,
                changed_features={}, n_generated=0,
            )

        cf_df = cf_list.final_cfs_df.drop(columns=["dropout_risk"], errors="ignore")
        cfs = cf_df.to_dict(orient="records")

        # Pick the CF closest to the original (L1 distance on normalised features)
        original = np.array([features_dict[f] for f in self.feature_names], dtype=np.float64)
        ranges = np.array([
            max(self.permitted_range.get(f, [0, 1])[1] - self.permitted_range.get(f, [0, 1])[0], 1e-8)
            for f in self.feature_names
        ])
        best_idx, best_dist = 0, float("inf")
        for i, cf in enumerate(cfs):
            cf_arr = np.array([cf[f] for f in self.feature_names], dtype=np.float64)
            dist = np.sum(np.abs(cf_arr - original) / ranges)
            if dist < best_dist:
                best_idx, best_dist = i, dist

        best_cf = cfs[best_idx]

        # Extract changed features and build actions
        changed_features = {}
        actions = []
        for f in self.feature_names:
            if f in IMMUTABLE_FEATURES:
                continue
            orig_val = float(features_dict[f])
            cf_val   = float(best_cf[f])
            if abs(cf_val - orig_val) > 1e-6:
                changed_features[f] = {"from": round(orig_val, 4), "to": round(cf_val, 4)}
                actions.append(PrescriptiveAction(
                    feature=f,
                    current_value=orig_val,
                    target_value=cf_val,
                    direction="increase" if cf_val > orig_val else "decrease",
                    magnitude=abs(cf_val - orig_val),
                ))

        # Sort by magnitude descending, assign priority ranks
        actions.sort(key=lambda a: a.magnitude, reverse=True)
        for i, a in enumerate(actions):
            a.priority_rank = i + 1

        return CounterfactualResult(
            actions=actions,
            counterfactuals=cfs,
            best_cf=best_cf,
            changed_features=changed_features,
            n_generated=len(cfs),
        )
