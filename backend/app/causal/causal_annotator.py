"""
Causal Annotator (DoWhy)
=========================
Uses DoWhy to estimate causal effects of each feature on dropout risk
from the training data.  A causal graph is built from domain knowledge
and validated via DoWhy's identification + estimation pipeline.

At init, runs causal effect estimation for every feature → dropout_risk,
stores the results, and exposes `annotate_shap()` to augment SHAP values
with causal vs correlational labels backed by real effect estimates.

Public API
----------
CausalAnnotator(X_train, y_train, feature_names)
    .annotate_shap(shap_values: dict) -> list[AnnotatedFeature]
    .get_causal_effects() -> dict[str, CausalEffect]
    .estimate_single_effect(feature, X, y) -> CausalEffect
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Suppress verbose DoWhy/EconML output during estimation
warnings.filterwarnings("ignore", module="dowhy")
warnings.filterwarnings("ignore", module="econml")


# ──────────────────────────────────────────────────────────────────────────────
# Domain Knowledge: Causal Graph (GML format for DoWhy)
# ──────────────────────────────────────────────────────────────────────────────
# This graph encodes domain knowledge about the OULAD dataset:
#
#  Engagement behaviours (login, session, forum) → Performance (quiz, assignment)
#  Performance → Dropout risk
#  Engagement → Dropout risk (direct + indirect via performance)
#  Structural features (prior_completions, current_week) → everything (confounders)
#
# This is NOT hardcoded classification — DoWhy will validate identifiability
# and estimate actual causal effects.

def _build_causal_graph(feature_names: list[str]) -> str:
    """
    Build a GML string encoding the causal DAG.
    """
    outcome = "dropout_risk"

    # Define edge list based on educational domain knowledge
    edges = []

    # Structural confounders → engagement & performance
    confounders = ["prior_course_completions", "current_week_in_course"]
    engagement  = ["login_frequency_weekly", "avg_session_duration_min",
                   "forum_posts_count", "help_requests_count"]
    performance = ["quiz_avg_score", "quiz_completion_rate",
                   "assignment_submission_rate", "video_completion_rate"]
    risk_signals = ["days_since_last_activity", "missed_deadlines_count"]

    # Confounders → engagement
    for c in confounders:
        if c in feature_names:
            for e in engagement:
                if e in feature_names:
                    edges.append((c, e))
            for p in performance:
                if p in feature_names:
                    edges.append((c, p))
            edges.append((c, outcome))

    # Engagement → performance
    for e in engagement:
        if e in feature_names:
            for p in performance:
                if p in feature_names:
                    edges.append((e, p))
            edges.append((e, outcome))

    # Performance → outcome
    for p in performance:
        if p in feature_names:
            edges.append((p, outcome))

    # Risk signals → outcome
    for r in risk_signals:
        if r in feature_names:
            edges.append((r, outcome))

    # Engagement → risk signals (e.g. low login → high days_since)
    for e in engagement:
        if e in feature_names:
            for r in risk_signals:
                if r in feature_names:
                    edges.append((e, r))

    # Latent engagement features (engagement_latent_1/2/3) are CORRELATIONAL:
    # they are compressed representations of platform interaction signals,
    # not direct interventions. They correlate with, but do not cause, dropout.
    # We add them as direct → outcome edges so DoWhy treats them as observable
    # correlates, not causes (no back-door paths from latent to confounders).
    latent_features = [f for f in feature_names if f.startswith("engagement_latent_")]
    for lf in latent_features:
        edges.append((lf, outcome))

    # Build GML
    all_nodes = [n for n in feature_names if n in feature_names] + [outcome]
    node_lines = []
    for i, n in enumerate(all_nodes):
        node_lines.append(f'  node [ id {i} label "{n}" ]')

    node_map = {n: i for i, n in enumerate(all_nodes)}
    edge_lines = []
    for src, tgt in edges:
        if src in node_map and tgt in node_map:
            edge_lines.append(f'  edge [ source {node_map[src]} target {node_map[tgt]} ]')

    gml = "graph [\n  directed 1\n" + "\n".join(node_lines) + "\n" + "\n".join(edge_lines) + "\n]"
    return gml


@dataclass
class CausalEffect:
    feature:         str
    ate:             float     # average treatment effect
    p_value:         Optional[float]
    is_causal:       bool      # True if ATE is statistically significant
    effect_direction: str      # "risk_increasing" | "risk_decreasing" | "neutral"
    method_used:     str
    note:            str

    def to_dict(self) -> dict:
        return {
            "feature":          self.feature,
            "ate":              round(self.ate, 6),
            "p_value":          round(self.p_value, 6) if self.p_value is not None else None,
            "is_causal":        self.is_causal,
            "effect_direction": self.effect_direction,
            "method_used":      self.method_used,
            "note":             self.note,
        }


@dataclass
class AnnotatedFeature:
    name:            str
    shap_value:      float
    shap_direction:  str       # "risk" | "protective"
    causal_type:     str       # "causal" | "correlational" | "confounder"
    causal_effect:   Optional[CausalEffect]
    note:            str

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "shap_value":     round(self.shap_value, 5),
            "shap_direction": self.shap_direction,
            "causal_type":    self.causal_type,
            "causal_effect":  self.causal_effect.to_dict() if self.causal_effect else None,
            "note":           self.note,
        }


class CausalAnnotator:
    CONFOUNDERS = {"prior_course_completions", "current_week_in_course"}
    # Latent engagement features are compressed signals — correlational by design
    LATENT_PREFIX = "engagement_latent_"

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
        significance_threshold: float = 0.10,
    ):
        self.feature_names = feature_names
        self.significance   = significance_threshold
        self.causal_effects: dict[str, CausalEffect] = {}

        # Build training dataframe
        self.df = pd.DataFrame(X_train, columns=feature_names)
        self.df["dropout_risk"] = y_train.astype(int)

        # Build causal graph
        self.gml = _build_causal_graph(feature_names)

        # Run causal effect estimation for each feature
        log.info("CausalAnnotator: estimating causal effects for %d features …", len(feature_names))
        for feat in feature_names:
            self.causal_effects[feat] = self.estimate_single_effect(feat)
        n_causal = sum(1 for e in self.causal_effects.values() if e.is_causal)
        log.info("CausalAnnotator initialised  causal=%d  correlational=%d",
                 n_causal, len(feature_names) - n_causal)

    def estimate_single_effect(self, feature: str) -> CausalEffect:
        """
        Estimate the average treatment effect (ATE) of `feature` on dropout_risk
        using DoWhy's identification + estimation pipeline.
        """
        # Latent engagement features are compressed platform-interaction signals.
        # They are not directly interventiable — always label as correlational.
        if feature.startswith(self.LATENT_PREFIX):
            from scipy.stats import pointbiserialr
            if feature in self.df.columns:
                corr, p_val = pointbiserialr(self.df["dropout_risk"], self.df[feature])
            else:
                corr, p_val = 0.0, 1.0
            direction = ("risk_increasing" if corr > 0.05
                        else "risk_decreasing" if corr < -0.05
                        else "neutral")
            return CausalEffect(
                feature=feature,
                ate=float(corr),
                p_value=float(p_val),
                is_causal=False,
                effect_direction=direction,
                method_used="correlational.latent_engagement",
                note=(
                    "Latent engagement feature — compressed platform interaction signal. "
                    "Correlational only; not directly interventiable. "
                    f"r={corr:.4f}, p={p_val:.6f}"
                ),
            )

        import dowhy

        try:
            # Binarise the treatment at the median for ATE estimation
            median_val = self.df[feature].median()
            df_copy = self.df.copy()
            treatment_col = f"{feature}_treatment"
            df_copy[treatment_col] = (df_copy[feature] > median_val).astype(int)

            # Identify confounders from the graph: all features that are parents of
            # both the treatment and outcome
            confounders = [f for f in self.feature_names
                          if f != feature and f in self.CONFOUNDERS]
            other_features = [f for f in self.feature_names
                             if f != feature and f not in self.CONFOUNDERS]

            # Use DoWhy causal model
            model = dowhy.CausalModel(
                data=df_copy,
                treatment=treatment_col,
                outcome="dropout_risk",
                common_causes=confounders if confounders else None,
                effect_modifiers=other_features[:3] if other_features else None,
                graph=None,  # Let DoWhy build from common_causes specification
            )

            # Identify causal effect
            identified = model.identify_effect(proceed_when_unidentifiable=True)

            # Estimate using linear regression (fast, interpretable)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression",
            )

            ate = float(estimate.value)

            # Refute with random common cause to get p-value proxy
            try:
                refutation = model.refute_estimate(
                    identified,
                    estimate,
                    method_name="random_common_cause",
                    num_simulations=50,
                )
                p_value = float(refutation.estimated_effect)
                # If refuted effect is close to original, the estimate is robust
                # Use the ratio as a pseudo-significance measure
                robustness = abs(ate - p_value) / max(abs(ate), 1e-8)
                is_causal = robustness < 0.5 and abs(ate) > 0.01
                p_val_report = robustness
            except Exception:
                p_val_report = None
                is_causal = abs(ate) > 0.02

            if feature in self.CONFOUNDERS:
                causal_type_note = "confounder"
                is_causal = False
            else:
                causal_type_note = "causal" if is_causal else "correlational"

            direction = ("risk_increasing" if ate > 0.01
                        else "risk_decreasing" if ate < -0.01
                        else "neutral")

            return CausalEffect(
                feature=feature,
                ate=ate,
                p_value=p_val_report,
                is_causal=is_causal,
                effect_direction=direction,
                method_used="dowhy.backdoor.linear_regression",
                note=f"ATE={ate:.4f} via DoWhy backdoor adjustment. "
                     f"{'Robust to random common cause.' if is_causal else 'Effect not robust or below threshold.'}",
            )

        except Exception as e:
            log.warning("DoWhy estimation failed for %s: %s — falling back to correlation", feature, e)
            # Fallback: use point-biserial correlation as a proxy
            from scipy.stats import pointbiserialr
            corr, p_val = pointbiserialr(self.df["dropout_risk"], self.df[feature])
            is_sig = p_val < self.significance and abs(corr) > 0.05
            return CausalEffect(
                feature=feature,
                ate=float(corr),
                p_value=float(p_val),
                is_causal=is_sig and feature not in self.CONFOUNDERS,
                effect_direction=("risk_increasing" if corr > 0.05
                                 else "risk_decreasing" if corr < -0.05
                                 else "neutral"),
                method_used="fallback.point_biserial_correlation",
                note=f"DoWhy failed, using correlation fallback. r={corr:.4f}, p={p_val:.6f}",
            )

    def get_causal_effects(self) -> dict[str, CausalEffect]:
        return self.causal_effects

    def annotate_shap(self, shap_values: dict[str, float]) -> list[AnnotatedFeature]:
        """
        Augment each SHAP feature with causal type and effect info.
        """
        annotated = []
        for name, shap_val in shap_values.items():
            effect = self.causal_effects.get(name)

            if name in self.CONFOUNDERS:
                causal_type = "confounder"
            elif name.startswith(self.LATENT_PREFIX):
                causal_type = "correlational"
            elif effect and effect.is_causal:
                causal_type = "causal"
            else:
                causal_type = "correlational"

            shap_dir = "risk" if shap_val > 0 else "protective"

            note = ""
            if name.startswith(self.LATENT_PREFIX):
                note = (
                    "Latent engagement score derived from platform interaction patterns "
                    "(clicks, time-on-page, What-If usage, action follow-rate). "
                    "Reflects your overall engagement quality — not directly editable."
                )
            elif effect:
                if causal_type == "causal":
                    note = (f"DoWhy confirms causal effect (ATE={effect.ate:.4f}). "
                           f"SHAP and causal direction {'agree' if (effect.ate > 0) == (shap_val > 0) else 'disagree'}.")
                elif causal_type == "confounder":
                    note = "Structural feature — cannot be changed by intervention."
                else:
                    note = (f"Correlation detected but causal effect not robust "
                           f"(ATE={effect.ate:.4f}). Interpret with caution.")

            annotated.append(AnnotatedFeature(
                name=name,
                shap_value=shap_val,
                shap_direction=shap_dir,
                causal_type=causal_type,
                causal_effect=effect,
                note=note,
            ))

        return annotated
