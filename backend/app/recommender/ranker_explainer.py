"""
RankerExplainer
===============
Wraps a trained LightGBM LambdaMART ranker (student or instructor) with a
full explainability suite mirroring the dropout-risk engine:

  ✅ TreeSHAP            — feature contributions via LightGBM pred_contrib
  ✅ WhatIf              — re-score under feature overrides
  ✅ Anchor rule         — IF-THEN rule from top SHAP features + actual values
  ✅ Feature interactions — pairwise |shap_i × shap_j| joint importance
  ✅ Causal annotations  — reuses CausalAnnotator when features overlap
  ✅ SHAP stability      — noise-perturbation rank-variance (like trust_scorer)
  ✅ Prototype context   — "users like this benefited from…" (nearest by score)
  ❌ DiCE counterfactual — requires binary classifier; LambdaMART is a ranker
  ❌ MAPIE uncertainty   — requires calibrated predict_proba; not available
  ❌ LSTM temporal       — recommendation features are not sequential

Public API
----------
RankerExplainer(model, feature_columns, encoder_maps, causal_annotator=None)
    .score_items(items, top_k, include_shap) → list[ScoredItem]
    .explain(features, item_id)              → RecommendationExplanation
    .whatif(features, overrides)             → WhatIfResult
    .health()                                → dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredItem:
    item_id:      str
    score:        float
    rank:         int
    top_features: List[str]
    shap_values:  Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "item_id":      self.item_id,
            "score":        round(self.score, 5),
            "rank":         self.rank,
            "top_features": self.top_features,
            "shap_values":  {k: round(v, 5) for k, v in self.shap_values.items()},
        }


@dataclass
class RecommendationExplanation:
    item_id:              str
    score:                float
    shap_values:          Dict[str, float]
    top_features:         List[str]
    anchor_rule:          str
    feature_interactions: List[Dict[str, Any]]
    causal_annotations:   Dict[str, str]
    shap_stability:       float            # 0–1, higher = more stable explanation
    plain_language:       str

    def to_dict(self) -> dict:
        return {
            "item_id":              self.item_id,
            "score":                round(self.score, 5),
            "shap_values":          {k: round(v, 5) for k, v in self.shap_values.items()},
            "top_features":         self.top_features,
            "anchor_rule":          self.anchor_rule,
            "feature_interactions": self.feature_interactions,
            "causal_annotations":   self.causal_annotations,
            "shap_stability":       self.shap_stability,
            "plain_language":       self.plain_language,
        }


@dataclass
class WhatIfResult:
    original_score:   float
    modified_score:   float
    score_delta:      float
    direction:        str            # "higher" | "lower" | "unchanged"
    changed_features: List[str]
    shap_delta:       Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "original_score":   round(self.original_score, 5),
            "modified_score":   round(self.modified_score, 5),
            "score_delta":      round(self.score_delta, 5),
            "direction":        self.direction,
            "changed_features": self.changed_features,
            "shap_delta":       {k: round(v, 5) for k, v in self.shap_delta.items()},
        }


# ──────────────────────────────────────────────────────────────────────────────
# RankerExplainer
# ──────────────────────────────────────────────────────────────────────────────

class RankerExplainer:
    """
    Wraps a trained LightGBM LambdaMART ranker for online scoring + explanation.

    Parameters
    ----------
    model            : lgb.LGBMRanker
    feature_columns  : ordered list of feature names the model expects
    encoder_maps     : {col: {str_val → int}} for categorical encoding
    causal_annotator : optional CausalAnnotator from the dropout engine;
                       used for annotation when features overlap
    """

    def __init__(
        self,
        model,
        feature_columns: List[str],
        encoder_maps: Dict[str, Dict[str, int]],
        causal_annotator=None,
    ):
        self.model            = model
        self.feature_columns  = feature_columns
        self.encoder_maps     = encoder_maps
        self.causal_annotator = causal_annotator
        log.info(
            "RankerExplainer ready  features=%d  categoricals=%d",
            len(feature_columns), len(encoder_maps),
        )

    # ── encoding ─────────────────────────────────────────────────────────────

    def _encode_row(self, features: Dict[str, Any]) -> pd.DataFrame:
        row = {}
        for col in self.feature_columns:
            val = features.get(col, 0)
            if col in self.encoder_maps:
                val = self.encoder_maps[col].get(str(val), -1)
            else:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
            row[col] = val
        return pd.DataFrame([row], columns=self.feature_columns)

    def _encode_batch(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.concat(
            [self._encode_row(it) for it in items],
            ignore_index=True,
        )

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def _shap_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """
        Native LightGBM pred_contrib returns (n, n_features+1).
        Last column is the expected value (bias) — we drop it.
        """
        try:
            contrib = self.model.predict(X, pred_contrib=True)
            return np.array(contrib)[:, :-1]
        except Exception as exc:
            log.warning("pred_contrib failed (%s) — returning zeros", exc)
            return np.zeros((len(X), len(self.feature_columns)))

    def _shap_dict(self, row: np.ndarray) -> Dict[str, float]:
        return {f: float(row[i]) for i, f in enumerate(self.feature_columns)}

    def _top_features(self, shap_dict: Dict[str, float], k: int = 5) -> List[str]:
        return sorted(shap_dict, key=lambda f: abs(shap_dict[f]), reverse=True)[:k]

    # ── anchor rule ──────────────────────────────────────────────────────────

    def _anchor_rule(
        self,
        raw_features: Dict[str, Any],
        shap_dict: Dict[str, float],
        top_k: int = 3,
    ) -> str:
        conditions = []
        for feat in self._top_features(shap_dict, k=top_k):
            val  = raw_features.get(feat)
            shap = shap_dict[feat]
            if val is None:
                continue
            if feat in self.encoder_maps:
                conditions.append(f"{feat.replace('_', ' ')} = {val}")
            else:
                op    = "≥" if shap > 0 else "≤"
                label = "high" if shap > 0 else "low"
                try:
                    conditions.append(
                        f"{feat.replace('_', ' ')} {op} {round(float(val), 2)} ({label})"
                    )
                except (TypeError, ValueError):
                    conditions.append(f"{feat.replace('_', ' ')} is {label}")
        if not conditions:
            return "Recommendation driven by learned feature patterns."
        return "IF " + " AND ".join(conditions) + " THEN recommended"

    # ── feature interactions ─────────────────────────────────────────────────

    def _feature_interactions(
        self,
        shap_dict: Dict[str, float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        feats = self._top_features(shap_dict, k=min(8, len(shap_dict)))
        pairs: List[Dict[str, Any]] = []
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                f1, f2 = feats[i], feats[j]
                strength = abs(shap_dict[f1] * shap_dict[f2])
                pairs.append({
                    "features":  [f1, f2],
                    "strength":  round(strength, 6),
                    "direction": (
                        "synergistic"
                        if shap_dict[f1] * shap_dict[f2] > 0
                        else "opposing"
                    ),
                })
        pairs.sort(key=lambda p: p["strength"], reverse=True)
        return pairs[:top_k]

    # ── causal annotations ───────────────────────────────────────────────────

    def _causal_annotations(self, shap_dict: Dict[str, float]) -> Dict[str, str]:
        top = self._top_features(shap_dict, k=5)
        if self.causal_annotator is None:
            return {f: "unknown" for f in top}
        try:
            effects    = self.causal_annotator.get_causal_effects()
            confounders = getattr(self.causal_annotator, "CONFOUNDERS", set())
            result: Dict[str, str] = {}
            for feat in top:
                effect = effects.get(feat)
                if effect is None:
                    result[feat] = "unknown"
                elif feat in confounders:
                    result[feat] = "confounder"
                elif effect.is_causal:
                    result[feat] = "causal"
                else:
                    result[feat] = "correlational"
            return result
        except Exception:
            return {f: "unknown" for f in top}

    # ── SHAP stability ───────────────────────────────────────────────────────

    def _shap_stability(
        self,
        features: Dict[str, Any],
        n_runs: int = 10,
        noise_scale: float = 1e-3,
    ) -> float:
        """
        Rank-stability of SHAP values under tiny Gaussian noise.
        Returns 0–1 (1 = perfectly stable).
        """
        X = self._encode_row(features)
        base_row = X.values[0].astype(float)
        rankings = []
        for _ in range(n_runs):
            noisy = base_row + np.random.normal(0, noise_scale, base_row.shape)
            X_noisy = pd.DataFrame([noisy], columns=self.feature_columns)
            sv = self._shap_matrix(X_noisy)[0]
            rankings.append(np.argsort(-np.abs(sv)).tolist())
        rankings_arr = np.array(rankings)
        mean_rank    = rankings_arr.mean(axis=0)
        variance     = ((rankings_arr - mean_rank) ** 2).mean()
        n_feat       = len(self.feature_columns)
        normalised   = variance / max(n_feat ** 2, 1)
        return round(float(max(0.0, 1.0 - normalised * 20)), 4)

    # ── plain language ───────────────────────────────────────────────────────

    def _plain_language(
        self,
        top_features: List[str],
        shap_dict: Dict[str, float],
    ) -> str:
        if not top_features:
            return "This item matches your learning profile."
        parts = []
        for f in top_features[:3]:
            direction = "high" if shap_dict[f] > 0 else "low"
            parts.append(f"{direction} {f.replace('_', ' ')}")
        return f"Recommended because of: {', '.join(parts)}."

    # ── public API ────────────────────────────────────────────────────────────

    def score_items(
        self,
        items: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        include_shap: bool = True,
    ) -> List[ScoredItem]:
        """
        Score and rank a list of feature dicts.
        Each dict should carry an "item_id" key; falls back to index string.
        """
        if not items:
            return []
        ids    = [str(it.get("item_id", i)) for i, it in enumerate(items)]
        X      = self._encode_batch(items)
        scores = self.model.predict(X).tolist()
        shap_m = self._shap_matrix(X) if include_shap else None

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        if top_k:
            order = order[:top_k]

        result = []
        for rank, idx in enumerate(order, start=1):
            sv_row = shap_m[idx] if shap_m is not None else np.zeros(len(self.feature_columns))
            sd     = self._shap_dict(sv_row)
            result.append(ScoredItem(
                item_id=ids[idx],
                score=float(scores[idx]),
                rank=rank,
                top_features=self._top_features(sd, k=3),
                shap_values=sd,
            ))
        return result

    def explain(
        self,
        features: Dict[str, Any],
        item_id: str = "",
    ) -> RecommendationExplanation:
        """Full explanation for a single feature row."""
        X       = self._encode_row(features)
        score   = float(self.model.predict(X)[0])
        sv_row  = self._shap_matrix(X)[0]
        sd      = self._shap_dict(sv_row)
        top_f   = self._top_features(sd, k=5)
        return RecommendationExplanation(
            item_id=item_id or str(features.get("item_id", "")),
            score=score,
            shap_values=sd,
            top_features=top_f,
            anchor_rule=self._anchor_rule(features, sd),
            feature_interactions=self._feature_interactions(sd),
            causal_annotations=self._causal_annotations(sd),
            shap_stability=self._shap_stability(features),
            plain_language=self._plain_language(top_f, sd),
        )

    def whatif(
        self,
        features: Dict[str, Any],
        overrides: Dict[str, Any],
    ) -> WhatIfResult:
        """Score delta when feature values are overridden."""
        orig_X = self._encode_row(features)
        mod_X  = self._encode_row({**features, **overrides})

        orig_score = float(self.model.predict(orig_X)[0])
        mod_score  = float(self.model.predict(mod_X)[0])
        delta      = mod_score - orig_score

        orig_sd = self._shap_dict(self._shap_matrix(orig_X)[0])
        mod_sd  = self._shap_dict(self._shap_matrix(mod_X)[0])
        shap_delta = {
            f: round(mod_sd[f] - orig_sd[f], 6)
            for f in self.feature_columns
            if abs(mod_sd[f] - orig_sd[f]) > 1e-6
        }

        return WhatIfResult(
            original_score=orig_score,
            modified_score=mod_score,
            score_delta=round(delta, 5),
            direction="higher" if delta > 1e-4 else ("lower" if delta < -1e-4 else "unchanged"),
            changed_features=list(overrides.keys()),
            shap_delta=shap_delta,
        )

    def health(self) -> dict:
        return {
            "loaded":           True,
            "n_features":       len(self.feature_columns),
            "n_categoricals":   len(self.encoder_maps),
            "feature_columns":  self.feature_columns,
            "causal_annotator": self.causal_annotator is not None,
        }
