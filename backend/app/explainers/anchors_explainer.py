"""
Anchors Rule Explainer
=======================
Generates IF-THEN rules using alibi's AnchorTabular.
Fits a discretizer on X_train at init; each `explain()` call produces
a human-readable anchor rule with precision and coverage metrics.

Public API
----------
AnchorsExplainer(model, X_train, feature_names)
    .explain(features_dict) -> AnchorResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from alibi.explainers import AnchorTabular

log = logging.getLogger(__name__)


@dataclass
class AnchorResult:
    anchor_rule:    str          # "quiz_avg_score <= 45.2 AND days_since_last_activity > 14"
    precision:      float        # P(same prediction | anchor holds)
    coverage:       float        # fraction of data where anchor applies
    predicates:     list[str]    # individual conditions
    human_readable: str          # natural-language version

    def to_dict(self) -> dict:
        return {
            "anchor_rule":    self.anchor_rule,
            "precision":      round(self.precision, 4),
            "coverage":       round(self.coverage, 4),
            "predicates":     self.predicates,
            "human_readable": self.human_readable,
        }


class AnchorsExplainer:
    def __init__(
        self,
        predict_fn: Callable,
        X_train: np.ndarray,
        feature_names: list[str],
        discretizer: str = "quartile",
        threshold: float = 0.90,
    ):
        """
        Parameters
        ----------
        predict_fn : callable
            Must accept (N, F) ndarray → (N,) integer predictions.
        X_train : ndarray (N, F)
        feature_names : list[str]
        discretizer : str
            'quartile' | 'decile' | 'entropy'
        threshold : float
            Desired precision for the anchor.
        """
        self.feature_names = feature_names
        self.threshold     = threshold

        self.explainer = AnchorTabular(
            predictor=predict_fn,
            feature_names=feature_names,
            seed=42,
        )
        self.explainer.fit(X_train, disc_perc=(25, 50, 75) if discretizer == "quartile"
                           else (10, 20, 30, 40, 50, 60, 70, 80, 90))

        log.info("AnchorsExplainer initialised  discretizer=%s  threshold=%.2f",
                 discretizer, threshold)

    def _predict_label(self, features_dict: dict) -> str:
        """Human label for the predicted class."""
        arr = np.array([features_dict[f] for f in self.feature_names]).reshape(1, -1)
        # We don't call the model here; just used for narration context
        return ""

    def _build_human_readable(self, predicates: list[str], precision: float) -> str:
        if not predicates:
            return "No anchor rule could be found for this instance."
        conditions = " AND ".join(predicates)
        return (
            f"IF {conditions}, THEN this prediction holds "
            f"with {precision:.0%} confidence."
        )

    def explain(self, features_dict: dict) -> AnchorResult:
        X = np.array(
            [features_dict[f] for f in self.feature_names], dtype=np.float64
        ).reshape(1, -1)

        explanation = self.explainer.explain(X, threshold=self.threshold)

        anchor_rule = " AND ".join(explanation.anchor) if explanation.anchor else ""
        predicates  = list(explanation.anchor) if explanation.anchor else []
        precision   = float(explanation.precision)
        coverage    = float(explanation.coverage)

        return AnchorResult(
            anchor_rule    = anchor_rule,
            precision      = precision,
            coverage       = coverage,
            predicates     = predicates,
            human_readable = self._build_human_readable(predicates, precision),
        )
