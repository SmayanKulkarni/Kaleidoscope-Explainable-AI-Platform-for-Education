"""
Dual SHAP Explainer
====================
Wraps TreeSHAP (GBM) and DeepSHAP (LSTM) under one interface.

Public API
----------
SHAPExplainer(gbm_model, feature_names, lstm_model=None, X_background=None)
    .explain(features_dict)               -> ExplainResult
    .explain_whatif(features_dict)        -> ExplainResult
    .explain_temporal(X_sequence)         -> TemporalExplainResult
    .stability_score(features_dict, n=20) -> float  0-1
    .global_summary(X_array)             -> dict  {feature: mean_abs_shap}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shap

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Return types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExplainResult:
    shap_values:  dict[str, float]
    base_value:   float
    risk_score:   float
    top_features: list[dict]        # [{name, value, shap, direction}] sorted by |shap|

    def to_dict(self) -> dict:
        return {
            "shap_values":  self.shap_values,
            "base_value":   self.base_value,
            "risk_score":   self.risk_score,
            "top_features": self.top_features,
        }


@dataclass
class TemporalExplainResult:
    temporal_attributions: list[list[float]]  # shape [T, F]
    week_importance:       list[float]         # shape [T]   sum |attr| per week
    feature_importance:    dict[str, float]    # sum |attr| per feature across time
    weeks:                 list[int]

    def to_dict(self) -> dict:
        return {
            "temporal_attributions": self.temporal_attributions,
            "week_importance":       self.week_importance,
            "feature_importance":    self.feature_importance,
            "weeks":                 self.weeks,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class SHAPExplainer:
    WEEK_ORDER = [2, 4, 6, 8, 10, 12]

    def __init__(
        self,
        gbm_model,
        feature_names: list[str],
        lstm_model=None,
        X_background: Optional[np.ndarray] = None,
    ):
        self.feature_names = feature_names
        self.gbm_model     = gbm_model

        # TreeSHAP needs the raw tree model, not CalibratedClassifierCV
        from sklearn.calibration import CalibratedClassifierCV
        base_model = gbm_model
        if isinstance(gbm_model, CalibratedClassifierCV):
            base_model = gbm_model.estimator
            if base_model is None:
                base_model = gbm_model.calibrated_classifiers_[0].estimator
        log.info("Initialising TreeExplainer (base=%s) …", type(base_model).__name__)
        self.tree_explainer = shap.TreeExplainer(base_model)

        self.deep_explainer = None
        self.lstm_model     = lstm_model
        if lstm_model is not None and X_background is not None:
            log.info("Initialising DeepExplainer …")
            import torch
            import torch.nn as _nn
            device = next(lstm_model.parameters()).device
            bg = torch.tensor(X_background, dtype=torch.float32).to(device)

            class _OutputWrapper(_nn.Module):
                """Ensures forward() returns (batch, 1) for SHAP's DeepExplainer."""
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    out = self.m(x)
                    return out.unsqueeze(-1) if out.dim() == 1 else out

            self.deep_explainer = shap.DeepExplainer(_OutputWrapper(lstm_model), bg)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _dict_to_array(self, features_dict: dict) -> np.ndarray:
        return np.array(
            [features_dict[f] for f in self.feature_names], dtype=np.float32
        ).reshape(1, -1)

    def _build_top_features(self, shap_vals: dict) -> list[dict]:
        items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
        return [
            {
                "name":      name,
                "shap":      round(val, 5),
                "direction": "risk" if val > 0 else "protective",
            }
            for name, val in items[:5]
        ]

    # ── GBM explain ────────────────────────────────────────────────────────────

    def explain(self, features_dict: dict) -> ExplainResult:
        X = self._dict_to_array(features_dict)
        sv = self.tree_explainer.shap_values(X)

        if isinstance(sv, list):
            sv = sv[1]
        sv_flat = sv[0].tolist()

        shap_vals   = dict(zip(self.feature_names, sv_flat))
        _ev = self.tree_explainer.expected_value
        if isinstance(_ev, (list, np.ndarray)):
            _ev_arr = np.asarray(_ev).ravel()
            base_value = float(_ev_arr[-1])
        else:
            base_value = float(_ev)
        risk_score  = float(self.gbm_model.predict_proba(X)[0][1])

        return ExplainResult(
            shap_values  = shap_vals,
            base_value   = base_value,
            risk_score   = risk_score,
            top_features = self._build_top_features(shap_vals),
        )

    def explain_whatif(self, features_dict: dict) -> ExplainResult:
        return self.explain(features_dict)

    # ── LSTM temporal explain ──────────────────────────────────────────────────

    def explain_temporal(self, X_sequence: np.ndarray) -> TemporalExplainResult:
        """
        X_sequence: shape (1, T, F) or (T, F) — will be reshaped to (1, T, F)
        Returns per-timestep × per-feature SHAP attributions via DeepSHAP.
        Falls back to zero matrix if DeepExplainer is unavailable.
        """
        if X_sequence.ndim == 2:
            X_sequence = X_sequence[np.newaxis, :]

        if self.deep_explainer is None:
            log.warning("DeepExplainer not available — returning zero attributions")
            attrs = np.zeros((1, len(self.WEEK_ORDER), len(self.feature_names)))
        else:
            import torch
            t = torch.tensor(X_sequence, dtype=torch.float32)
            attrs = np.array(self.deep_explainer.shap_values(t))
            if attrs.ndim == 4:
                attrs = attrs[0]

        attrs_sq = attrs[0]
        week_imp = np.abs(attrs_sq).sum(axis=1).tolist()
        feat_imp = {
            name: round(float(np.abs(attrs_sq[:, i]).sum()), 5)
            for i, name in enumerate(self.feature_names)
        }
        return TemporalExplainResult(
            temporal_attributions = attrs_sq.tolist(),
            week_importance       = [round(v, 5) for v in week_imp],
            feature_importance    = feat_imp,
            weeks                 = self.WEEK_ORDER,
        )

    # ── stability ──────────────────────────────────────────────────────────────

    def stability_score(self, features_dict: dict, n_runs: int = 20) -> float:
        """
        Runs SHAP n times on feature vectors with tiny Gaussian noise.
        Returns 1 - normalised mean rank variance (higher = more stable).
        """
        base = self._dict_to_array(features_dict)
        noise_scale = 1e-4
        rankings = []

        for _ in range(n_runs):
            noisy = base + np.random.normal(0, noise_scale, base.shape).astype(np.float32)
            sv = self.tree_explainer.shap_values(noisy)
            if isinstance(sv, list):
                sv = sv[1]
            order = np.argsort(-np.abs(sv[0]))
            rankings.append(order.tolist())

        rankings = np.array(rankings)
        mean_rank = rankings.mean(axis=0)
        variance  = ((rankings - mean_rank) ** 2).mean()
        n_feat    = len(self.feature_names)
        normalised_var = variance / (n_feat ** 2)
        return round(float(max(0.0, 1.0 - normalised_var * 20)), 4)

    # ── global ─────────────────────────────────────────────────────────────────

    def global_summary(self, X_array: np.ndarray) -> dict[str, float]:
        """Mean absolute SHAP across a dataset. Returns {feature: mean_abs_shap}."""
        sv = self.tree_explainer.shap_values(X_array)
        if isinstance(sv, list):
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        return {name: round(float(v), 5) for name, v in zip(self.feature_names, mean_abs)}
