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
    # Feature 2: trust score
    trust_score:          Optional[Dict[str, float]] = None
    # Feature 3: Alibi anchor precision
    anchor_precision:     float = 0.0
    # Feature 5: KNN prototypes
    prototypes:           List[Dict] = field(default_factory=list)
    # Feature 1: LLM narration (set by endpoint, not explainer)
    narratives:           Optional[Dict] = None
    # Feature 6: explanation drift (set by endpoint, not explainer)
    explanation_drift:    Optional[Dict] = None

    def to_dict(self) -> dict:
        return {
            "item_id":              self.item_id,
            "score":                round(self.score, 5),
            "shap_values":          {k: round(v, 5) for k, v in self.shap_values.items()},
            "top_features":         self.top_features,
            "anchor_rule":          self.anchor_rule,
            "anchor_precision":     round(self.anchor_precision, 4),
            "feature_interactions": self.feature_interactions,
            "causal_annotations":   self.causal_annotations,
            "shap_stability":       self.shap_stability,
            "plain_language":       self.plain_language,
            "trust_score":          self.trust_score,
            "prototypes":           self.prototypes,
            "narratives":           self.narratives,
            "explanation_drift":    self.explanation_drift,
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
        X_train_sample: Optional[np.ndarray] = None,
        reference_pool: Optional[pd.DataFrame] = None,
    ):
        self.model            = model
        self.feature_columns  = feature_columns
        self.encoder_maps     = encoder_maps
        self.causal_annotator = causal_annotator
        self.reference_pool   = reference_pool
        self._anchors_explainer = None

        # Feature 3: initialise Alibi AnchorsExplainer if training sample is available
        if X_train_sample is not None and len(X_train_sample) > 10:
            try:
                from backend.app.explainers.anchors_explainer import AnchorsExplainer as _AE
                _X_df  = pd.DataFrame(X_train_sample, columns=self.feature_columns)
                _scores = np.array(self.model.predict(_X_df), dtype=float)
                _median = float(np.median(_scores))

                def _pseudo_predict(X_np: np.ndarray) -> np.ndarray:
                    X_df = pd.DataFrame(X_np, columns=self.feature_columns)
                    preds = np.array(self.model.predict(X_df), dtype=float)
                    return (preds >= _median).astype(int)

                self._anchors_explainer = _AE(
                    predict_fn    = _pseudo_predict,
                    X_train       = X_train_sample,
                    feature_names = self.feature_columns,
                )
                log.info("RankerExplainer: AnchorsExplainer initialised")
            except Exception as _exc:
                log.warning(
                    "RankerExplainer: AnchorsExplainer init failed (%s) — template fallback",
                    _exc,
                )

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

    def _feature_interactions_product(
        self,
        shap_dict: Dict[str, float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Product-based |shap_i × shap_j| approximation (fallback)."""
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

    def _feature_interactions(
        self,
        X: pd.DataFrame,
        shap_dict: Dict[str, float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Feature 4: native LightGBM pred_interact with product-based fallback."""
        try:
            interact_raw = self.model.predict(X, pred_interact=True)
            interact_arr = np.array(interact_raw)
            if interact_arr.ndim == 3:
                interact_mat = interact_arr[0]  # (n_features, n_features)
            elif interact_arr.ndim == 2:
                interact_mat = interact_arr
            else:
                raise ValueError(f"Unexpected interact shape: {interact_arr.shape}")
            n = len(self.feature_columns)
            pairs: List[Dict[str, Any]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    strength = abs(float(interact_mat[i, j]))
                    if strength < 1e-12:
                        continue
                    pairs.append({
                        "features":  [self.feature_columns[i], self.feature_columns[j]],
                        "strength":  round(strength, 6),
                        "direction": (
                            "synergistic"
                            if interact_mat[i, j] > 0
                            else "opposing"
                        ),
                    })
            pairs.sort(key=lambda p: p["strength"], reverse=True)
            return pairs[:top_k]
        except Exception as exc:
            log.warning("pred_interact failed (%s) — using product approximation", exc)
            return self._feature_interactions_product(shap_dict, top_k)

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

    # ── anchor rule with Alibi + precision ─────────────────────────────────

    def _anchor_rule_with_precision(
        self,
        raw_features: Dict[str, Any],
        shap_dict: Dict[str, float],
    ) -> tuple:
        """Feature 3: Alibi AnchorTabular with template-based fallback."""
        if self._anchors_explainer is not None:
            try:
                X = self._encode_row(raw_features)
                encoded_dict = {
                    col: float(X.values[0][i])
                    for i, col in enumerate(self.feature_columns)
                }
                anchor_result = self._anchors_explainer.explain(encoded_dict)
                ad = anchor_result.to_dict()
                rule = ad.get("anchor_rule") or ad.get("human_readable") or ""
                return rule, float(ad.get("precision", 0.0))
            except Exception as exc:
                log.warning("Alibi anchor failed (%s) — using template fallback", exc)
        return self._anchor_rule(raw_features, shap_dict), 0.0

    # ── trust score ──────────────────────────────────────────────────────────

    def _compute_trust_score(
        self,
        X: pd.DataFrame,
        score: float,
        shap_row: np.ndarray,
        stability: float,
    ) -> Dict[str, float]:
        """Feature 2: composite trust score (fidelity 40%, stability 35%, completeness 25%)."""
        try:
            contrib_full = np.array(self.model.predict(X, pred_contrib=True))[0]
            bias         = float(contrib_full[-1])
            shap_sum     = float(shap_row.sum())
            fidelity     = float(
                1.0 - min(abs(shap_sum + bias - score) / max(abs(score), 1e-8), 1.0)
            )
            abs_shap     = np.abs(shap_row)
            total_abs    = float(abs_shap.sum())
            top5_abs     = float(np.sort(abs_shap)[::-1][:5].sum())
            completeness = float(top5_abs / max(total_abs, 1e-8))
            trust        = 0.40 * fidelity + 0.35 * stability + 0.25 * completeness
            return {
                "trust_score":  round(trust, 4),
                "fidelity":     round(fidelity, 4),
                "stability":    round(stability, 4),
                "completeness": round(completeness, 4),
            }
        except Exception as exc:
            log.warning("Trust score computation failed (%s)", exc)
            return {
                "trust_score":  0.0,
                "fidelity":     0.0,
                "stability":    round(stability, 4),
                "completeness": 0.0,
            }

    # ── KNN prototypes ───────────────────────────────────────────────────────

    def _prototypes(
        self,
        X: pd.DataFrame,
        K: int = 3,
    ) -> List[Dict[str, Any]]:
        """Feature 5: K nearest neighbours from reference pool by Euclidean distance."""
        if self.reference_pool is None or self.reference_pool.empty:
            return []
        try:
            avail_cols = [c for c in self.feature_columns if c in self.reference_pool.columns]
            if not avail_cols:
                return []
            query_vec = X[avail_cols].values[0].astype(float)
            ref_mat   = self.reference_pool[avail_cols].values.astype(float)
            diffs     = ref_mat - query_vec
            dists     = np.sqrt((diffs ** 2).sum(axis=1))
            top_idx   = np.argsort(dists)[:K]
            result: List[Dict[str, Any]] = []
            for idx in top_idx:
                row        = self.reference_pool.iloc[idx]
                similarity = round(1.0 / (1.0 + float(dists[idx])), 4)
                result.append({
                    "item_id":    str(row.get("item_id", idx)),
                    "score":      round(float(row.get("score", 0.0)), 5),
                    "similarity": similarity,
                    "outcome":    str(row.get("outcome", "unknown")),
                })
            return result
        except Exception as exc:
            log.warning("Prototypes failed (%s)", exc)
            return []

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
        stability                = self._shap_stability(features)
        anchor_rule_str, anchor_prec = self._anchor_rule_with_precision(features, sd)
        trust_dict               = self._compute_trust_score(X, score, sv_row, stability)
        prototypes               = self._prototypes(X)
        return RecommendationExplanation(
            item_id=item_id or str(features.get("item_id", "")),
            score=score,
            shap_values=sd,
            top_features=top_f,
            anchor_rule=anchor_rule_str,
            anchor_precision=anchor_prec,
            feature_interactions=self._feature_interactions(X, sd),
            causal_annotations=self._causal_annotations(sd),
            shap_stability=stability,
            plain_language=self._plain_language(top_f, sd),
            trust_score=trust_dict,
            prototypes=prototypes,
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
