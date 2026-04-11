"""
Trust Scorer
=============
Composite trust score for explanations:
    trust = 0.40 × fidelity + 0.35 × stability + 0.25 × completeness

- **Fidelity:** How well SHAP sum + base_value ≈ predict_proba (measured, not hardcoded)
- **Stability:** From SHAPExplainer.stability_score() (perturbation-based)
- **Completeness:** Fraction of total |SHAP| captured by top-5 features

Public API
----------
TrustScorer()
    .score(shap_values, base_value, risk_score, stability, top_k=5) -> TrustResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

WEIGHT_FIDELITY     = 0.40
WEIGHT_STABILITY    = 0.35
WEIGHT_COMPLETENESS = 0.25


@dataclass
class TrustResult:
    trust_score:   float   # 0-1 composite
    fidelity:      float   # 0-1
    stability:     float   # 0-1
    completeness:  float   # 0-1
    label:         str     # "High" | "Medium" | "Low"
    breakdown:     dict    # detailed explanation of each component

    def to_dict(self) -> dict:
        return {
            "trust_score":  round(self.trust_score, 4),
            "fidelity":     round(self.fidelity, 4),
            "stability":    round(self.stability, 4),
            "completeness": round(self.completeness, 4),
            "label":        self.label,
            "breakdown":    self.breakdown,
        }


class TrustScorer:
    def __init__(
        self,
        w_fidelity: float = WEIGHT_FIDELITY,
        w_stability: float = WEIGHT_STABILITY,
        w_completeness: float = WEIGHT_COMPLETENESS,
    ):
        total = w_fidelity + w_stability + w_completeness
        self.w_fidelity     = w_fidelity / total
        self.w_stability    = w_stability / total
        self.w_completeness = w_completeness / total

    def _compute_fidelity(
        self, shap_values: dict[str, float], base_value: float, risk_score: float
    ) -> float:
        """
        Fidelity = 1 - |sum(SHAP) + base_value - risk_score|
        Perfect SHAP → fidelity = 1.0
        """
        shap_sum = sum(shap_values.values())
        error = abs(shap_sum + base_value - risk_score)
        return float(np.clip(1.0 - error, 0.0, 1.0))

    def _compute_completeness(
        self, shap_values: dict[str, float], top_k: int = 5
    ) -> float:
        """
        Fraction of total |SHAP| captured by the top-k features.
        High completeness → a few features dominate the explanation.
        """
        abs_vals = sorted([abs(v) for v in shap_values.values()], reverse=True)
        total = sum(abs_vals)
        if total < 1e-10:
            return 1.0  # trivial case: all zero
        top_sum = sum(abs_vals[:top_k])
        return float(top_sum / total)

    def _label(self, score: float) -> str:
        if score >= 0.75:
            return "High"
        elif score >= 0.50:
            return "Medium"
        return "Low"

    def score(
        self,
        shap_values: dict[str, float],
        base_value: float,
        risk_score: float,
        stability: float,
        top_k: int = 5,
    ) -> TrustResult:
        fidelity     = self._compute_fidelity(shap_values, base_value, risk_score)
        completeness = self._compute_completeness(shap_values, top_k)

        trust = (
            self.w_fidelity     * fidelity
            + self.w_stability  * stability
            + self.w_completeness * completeness
        )

        label = self._label(trust)

        breakdown = {
            "fidelity": {
                "weight": round(self.w_fidelity, 2),
                "value":  round(fidelity, 4),
                "contribution": round(self.w_fidelity * fidelity, 4),
                "description": "1 - |SHAP_sum + base - risk_score|",
            },
            "stability": {
                "weight": round(self.w_stability, 2),
                "value":  round(stability, 4),
                "contribution": round(self.w_stability * stability, 4),
                "description": "Rank consistency under perturbation",
            },
            "completeness": {
                "weight": round(self.w_completeness, 2),
                "value":  round(completeness, 4),
                "contribution": round(self.w_completeness * completeness, 4),
                "description": f"Top-{top_k} features cover {completeness:.0%} of total attribution",
            },
        }

        return TrustResult(
            trust_score=trust,
            fidelity=fidelity,
            stability=stability,
            completeness=completeness,
            label=label,
            breakdown=breakdown,
        )
