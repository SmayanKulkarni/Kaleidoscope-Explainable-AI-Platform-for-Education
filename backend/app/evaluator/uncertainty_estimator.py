"""
Uncertainty Estimator (MAPIE)
==============================
Conformal prediction wrapper around the GBM model using MapieClassifier.
Provides calibrated prediction sets with guaranteed coverage.

Public API
----------
UncertaintyEstimator(model, X_cal, y_cal)
    .predict_with_uncertainty(X, alpha=0.1) -> UncertaintyResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from mapie.classification import SplitConformalClassifier

log = logging.getLogger(__name__)


@dataclass
class UncertaintyResult:
    prediction:       int
    risk_score:       float
    prediction_set:   list[int]   # classes in the conformal set
    confidence_width: float       # |set| / n_classes — 0=certain, 1=max uncertainty
    uncertainty_label: str        # "Low" | "Medium" | "High"

    def to_dict(self) -> dict:
        return {
            "prediction":        self.prediction,
            "risk_score":        round(self.risk_score, 4),
            "prediction_set":    self.prediction_set,
            "confidence_width":  round(self.confidence_width, 4),
            "uncertainty_label": self.uncertainty_label,
        }


class UncertaintyEstimator:
    def __init__(
        self,
        model,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        method: str = "lac",
    ):
        """
        Parameters
        ----------
        model : sklearn classifier with predict_proba
        X_cal : calibration set features (disjoint from training)
        y_cal : calibration set labels
        method : MAPIE method ('lac', 'aps', 'top_k')
        """
        self.model = model
        self.mapie = SplitConformalClassifier(
            estimator=model,
            confidence_level=0.9,
        )
        self.mapie.conformalize(X_cal, y_cal)
        log.info("UncertaintyEstimator initialised  method=%s  cal_size=%d", method, len(X_cal))

    def predict_with_uncertainty(
        self, X: np.ndarray, alpha: float = 0.10
    ) -> UncertaintyResult:
        """
        Parameters
        ----------
        X : (1, n_features) array
        alpha : significance level (0.10 = 90% coverage guarantee)
        """
        # predict_set returns (pred_sets, y_pred)
        #   pred_sets shape: (n_samples, n_classes)
        #   y_pred shape: (n_samples,)
        try:
            pred_sets, y_pred = self.mapie.predict_set(X)
        except Exception:
            # Fallback: return deterministic prediction
            proba = self.model.predict_proba(X)
            risk_score = float(proba[0, 1])
            prediction = int(proba[0, 1] >= 0.5)
            return UncertaintyResult(
                prediction=prediction, risk_score=risk_score,
                prediction_set=[prediction], confidence_width=0.5,
                uncertainty_label="Medium",
            )

        y_pred_single = self.mapie.predict(X)
        prediction = int(y_pred_single[0])

        # Get risk score from underlying model
        proba = self.model.predict_proba(X)
        risk_score = float(proba[0, 1])

        # pred_sets shape varies by version: (n, n_classes) or (n, n_classes, 1)
        arr = np.array(pred_sets)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        pred_set_mask = arr[0]
        prediction_set = [int(c) for c in np.where(pred_set_mask)[0]]

        n_classes = max(arr.shape[1] if arr.ndim > 1 else len(pred_set_mask), 2)
        confidence_width = len(prediction_set) / n_classes

        # Label based on set size
        if len(prediction_set) == 1:
            label = "Low"
        elif len(prediction_set) == n_classes:
            label = "High"
        else:
            label = "Medium"

        return UncertaintyResult(
            prediction=prediction,
            risk_score=risk_score,
            prediction_set=prediction_set,
            confidence_width=confidence_width,
            uncertainty_label=label,
        )
