"""
DL Explainer (Captum — Integrated Gradients)
=============================================
Provides temporal attribution for the LSTM model using Captum's
IntegratedGradients and LayerIntegratedGradients.

Complements DeepSHAP (in shap_explainer.py) with a second attribution
method — the two can be cross-validated for consistency.

Public API
----------
DLExplainer(lstm_model)
    .explain_temporal(sequence)         -> TemporalAttributionResult
    .temporal_attention_summary(sequence) -> dict
    .cross_validate(sequence, shap_result) -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class TemporalAttributionResult:
    """
    Attribution matrix from Integrated Gradients over an LSTM sequence.

    Attributes
    ----------
    temporal_attributions : list[list[float]]
        Shape (n_timesteps, n_features) — per-week × per-feature attribution.
    week_importance : list[float]
        Sum of absolute attributions per week (length = n_timesteps).
    feature_importance : list[float]
        Sum of absolute attributions per feature (length = n_features).
    most_important_week : int
        0-indexed week with highest total attribution.
    most_important_feature_per_week : list[int]
        Index of top feature for each week.
    convergence_delta : float
        Approximation error from IG (lower is better; < 0.05 is good).
    method : str
        "integrated_gradients"
    """

    temporal_attributions:          list[list[float]]
    week_importance:                list[float]
    feature_importance:             list[float]
    most_important_week:            int
    most_important_feature_per_week: list[int]
    convergence_delta:              float
    method:                         str = "integrated_gradients"

    def to_dict(self) -> dict:
        return {
            "temporal_attributions":           self.temporal_attributions,
            "week_importance":                 [round(w, 5) for w in self.week_importance],
            "feature_importance":              [round(f, 5) for f in self.feature_importance],
            "most_important_week":             self.most_important_week,
            "most_important_feature_per_week": self.most_important_feature_per_week,
            "convergence_delta":               round(self.convergence_delta, 6),
            "method":                          self.method,
        }


class DLExplainer:
    """
    Captum-based explainer for the DropoutLSTM model.

    Parameters
    ----------
    lstm_model : DropoutLSTM (nn.Module)
        The trained LSTM model. Must accept (batch, seq_len, n_features) tensors.
    feature_names : list[str]
        Names for the n_features dimension.
    device : str | torch.device
        "cuda" or "cpu"
    """

    def __init__(
        self,
        lstm_model,
        feature_names: list[str],
        device: str = "cpu",
    ):
        try:
            from captum.attr import IntegratedGradients
        except ImportError as e:
            raise ImportError(
                "captum is required for DLExplainer. "
                "Install with: pip install captum"
            ) from e

        self.model        = lstm_model
        self.feature_names = feature_names
        self.device       = torch.device(device)
        self.model.eval()
        self.model.to(self.device)

        # Wrap model to return scalar output for IG
        self._wrapped = _LSTMScalarWrapper(lstm_model)
        self.ig = IntegratedGradients(self._wrapped)
        log.info(
            "DLExplainer initialised  features=%d  device=%s",
            len(feature_names), device,
        )

    def explain_temporal(
        self,
        sequence: np.ndarray,
        target: int = 1,
        n_steps: int = 50,
        baseline: Optional[np.ndarray] = None,
    ) -> TemporalAttributionResult:
        """
        Compute Integrated Gradients attributions over a temporal sequence.

        Parameters
        ----------
        sequence : np.ndarray, shape (1, n_timesteps, n_features)
        target   : int — class index (1 = dropout risk)
        n_steps  : int — IG approximation steps (more = more accurate)
        baseline : optional zero-baseline override, same shape as sequence

        Returns
        -------
        TemporalAttributionResult
        """
        inp = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        if baseline is None:
            base = torch.zeros_like(inp)
        else:
            base = torch.tensor(baseline, dtype=torch.float32, device=self.device)

        attrs, delta = self.ig.attribute(
            inp,
            baselines=base,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=True,
        )
        # attrs shape: (1, n_timesteps, n_features)
        attrs_np = attrs.squeeze(0).detach().cpu().numpy()   # (T, F)

        week_importance    = attrs_np.abs().sum(axis=1).tolist()
        feature_importance = attrs_np.abs().sum(axis=0).tolist()
        most_important_week = int(np.argmax(week_importance))
        top_feat_per_week  = attrs_np.abs().argmax(axis=1).tolist()
        conv_delta = float(delta.abs().mean().item())

        return TemporalAttributionResult(
            temporal_attributions=attrs_np.tolist(),
            week_importance=week_importance,
            feature_importance=feature_importance,
            most_important_week=most_important_week,
            most_important_feature_per_week=top_feat_per_week,
            convergence_delta=conv_delta,
        )

    def temporal_attention_summary(self, sequence: np.ndarray) -> dict:
        """
        Human-readable summary of which weeks and features matter most.

        Returns
        -------
        dict with keys: top_week, top_feature, week_ranking, feature_ranking
        """
        result = self.explain_temporal(sequence)
        week_ranking = sorted(
            range(len(result.week_importance)),
            key=lambda i: result.week_importance[i],
            reverse=True,
        )
        feat_ranking = sorted(
            range(len(result.feature_importance)),
            key=lambda i: result.feature_importance[i],
            reverse=True,
        )
        return {
            "top_week":        result.most_important_week,
            "top_feature":     self.feature_names[feat_ranking[0]] if self.feature_names else feat_ranking[0],
            "week_ranking":    week_ranking,
            "feature_ranking": [self.feature_names[i] if self.feature_names else i for i in feat_ranking],
            "week_importance": {f"week_{i}": round(v, 5) for i, v in enumerate(result.week_importance)},
        }

    def cross_validate(
        self,
        sequence: np.ndarray,
        deep_shap_values: list[float],
        tolerance: float = 0.20,
    ) -> dict:
        """
        Cross-validate IG attributions against DeepSHAP feature importances.

        Compares per-feature total importance (summed over time) between
        the two methods. Returns agreement score and any disagreeing features.

        Parameters
        ----------
        sequence         : (1, T, F) input
        deep_shap_values : list[float] of length F (from shap_explainer.explain_temporal)
        tolerance        : max allowed rank difference to be considered "agreement"
        """
        ig_result    = self.explain_temporal(sequence)
        ig_feat_imp  = np.array(ig_result.feature_importance)
        shap_feat_imp = np.abs(np.array(deep_shap_values))

        ig_rank   = np.argsort(-ig_feat_imp)
        shap_rank = np.argsort(-shap_feat_imp)

        rank_diffs   = []
        disagreements = []
        for feat_idx in range(len(ig_feat_imp)):
            ig_r   = int(np.where(ig_rank == feat_idx)[0][0])
            shap_r = int(np.where(shap_rank == feat_idx)[0][0])
            diff   = abs(ig_r - shap_r)
            rank_diffs.append(diff)
            if diff > tolerance * len(ig_feat_imp):
                name = self.feature_names[feat_idx] if self.feature_names else str(feat_idx)
                disagreements.append({
                    "feature":    name,
                    "ig_rank":    ig_r,
                    "shap_rank":  shap_r,
                    "rank_diff":  diff,
                })

        avg_rank_diff = float(np.mean(rank_diffs))
        agreement = max(0.0, 1.0 - avg_rank_diff / len(ig_feat_imp))

        return {
            "agreement_score":  round(agreement, 4),
            "avg_rank_diff":    round(avg_rank_diff, 3),
            "disagreements":    disagreements,
            "methods_compared": ["integrated_gradients", "deep_shap"],
        }


class _LSTMScalarWrapper(torch.nn.Module):
    """
    Wraps DropoutLSTM to return a scalar probability for class `target`.
    Required by Captum (expects scalar output for attribution).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        # logits shape: (batch, 1)  or (batch,)
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        probs = torch.sigmoid(logits)          # (batch, 1)
        return probs[:, 0]                     # (batch,) scalar per sample
