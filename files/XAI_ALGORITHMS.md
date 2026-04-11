# XAI Algorithms — Complete Reference

> This document covers every XAI algorithm used in the system:
> conventional baselines + novel approaches. Includes **why** each
> was chosen, **how** it maps to an MVP component, and **implementation notes**.

---

## Algorithm Stack Overview

```
┌─────────────────────────────────────────────────────────┐
│                   EXPLANATION LAYERS                    │
├──────────────┬──────────────────┬───────────────────────┤
│ PREDICTION   │ FEATURE          │ CONCEPT               │
│ LEVEL        │ LEVEL            │ LEVEL                 │
│              │                  │                       │
│ PDP + ICE    │ TreeSHAP /       │ LIME over concept     │
│ Global SHAP  │ KernelSHAP       │ groups / TCAV         │
│ summary      │ Archipelago      │ Prototype match       │
│              │ (interactions)   │ Anchors rules         │
└──────────────┴──────────────────┴───────────────────────┘
┌─────────────────────────────────────────────────────────┐
│               DIAGNOSIS LAYER                          │
│  CEM (pertinent positives + negatives)                 │
│  MC Dropout uncertainty bounds                         │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│               PRESCRIPTION LAYER                       │
│  DiCE counterfactuals → ranked actions                 │
│  SHAP × actionability → priority scoring               │
└─────────────────────────────────────────────────────────┘
```

---

## 1. SHAP — SHapley Additive exPlanations

**Type:** Post-hoc, Feature Attribution, Local + Global  
**MVP Requirement:** Feature-level explanation (mandatory), Stability indicator  
**Library:** `shap`

### Why SHAP?
SHAP is the theoretical gold standard for feature attribution. It satisfies:
- **Efficiency**: attributions sum to the prediction gap from baseline
- **Symmetry**: equal features get equal attribution
- **Dummy**: irrelevant features get 0
- **Additivity**: attributions across models combine linearly

For tree-based models (GBM, RF), **TreeSHAP** runs in `O(TLD²)` — fast enough for real-time use.

### Implementation
```python
import shap
import numpy as np

class SHAPExplainer(BaseExplainer):
    def __init__(self, model, X_train: np.ndarray):
        # Use TreeExplainer for tree models (fast path)
        if hasattr(model, 'estimators_'):
            self.explainer = shap.TreeExplainer(model)
        else:
            # Fallback: KernelSHAP (slower, model-agnostic)
            self.explainer = shap.KernelExplainer(
                model.predict_proba, shap.sample(X_train, 100)
            )
        self.feature_names = list(LEARNER_FEATURES.keys())

    def explain(self, features: dict, model) -> dict:
        X = np.array([list(features.values())])
        shap_values = self.explainer.shap_values(X)
        # For binary classification, take class-1 (at-risk) SHAP values
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        return {
            "shap_values": dict(zip(self.feature_names, vals.tolist())),
            "base_value": float(self.explainer.expected_value[1]
                                if isinstance(self.explainer.expected_value, list)
                                else self.explainer.expected_value),
        }

    def stability_score(self, features: dict, n_runs: int = 20) -> float:
        """
        Perturb inputs slightly, re-run SHAP, measure variance.
        Returns 1.0 - normalized std of top feature ranks.
        """
        results = []
        X = np.array([list(features.values())])
        for _ in range(n_runs):
            noise = np.random.normal(0, 0.01, X.shape)
            shap_vals = self.explainer.shap_values(X + noise)
            vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
            results.append(vals)
        arr = np.array(results)  # shape: (n_runs, n_features)
        # Rank variance: higher = less stable
        rank_variance = np.mean(np.std(arr, axis=0) / (np.abs(np.mean(arr, axis=0)) + 1e-8))
        return float(np.clip(1.0 - rank_variance, 0.0, 1.0))

    def global_summary(self, X: np.ndarray) -> dict:
        """Mean absolute SHAP values across dataset — global feature importance."""
        shap_values = self.explainer.shap_values(X)
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs = np.mean(np.abs(vals), axis=0)
        return dict(zip(self.feature_names, mean_abs.tolist()))
```

### Stability Indicator Surface
- Run `stability_score()` after every explanation
- Expose as `stability: 0.87` in API response
- Flag in UI if `< 0.6` — "⚠️ This explanation may vary"

---

## 2. FastSHAP — Real-Time SHAP via Surrogate Network ⭐ NOVEL

**Type:** Amortized Feature Attribution (Novel, 2021–2025)  
**MVP Requirement:** What-If real-time updates without per-call SHAP latency  
**Library:** Custom (based on Jethani et al. 2021 — `fastshap` PyPI package)

### Why FastSHAP?
Standard SHAP recalculates from scratch on every What-If change. For a live slider UI
with 12 features, that's 12+ calls/second to KernelSHAP — too slow.

**FastSHAP trains a neural network surrogate** that learns to output SHAP values directly.
After training (~minutes), inference is `O(1)` — same speed as a forward pass.

### How It Works
```
Train once:
  Input: learner feature vector X (12-dim)
  Output: SHAP value vector Ŝ (12-dim)
  Loss: E[(f(X) - Σ Ŝᵢ)²] + KL(Ŝ || SHAP_ground_truth)

At runtime (What-If slider):
  X_modified → FastSHAP surrogate → Ŝ in <5ms
```

### Implementation
```python
# pip install fastshap
import fastshap
import torch

class FastSHAPExplainer:
    def __init__(self, model, X_train: np.ndarray):
        self.surrogate = fastshap.Surrogate(
            nn_model=self._build_mlp(X_train.shape[1]),
            dim=X_train.shape[1]
        )
        # Train FastSHAP surrogate using imputer
        imputer = fastshap.TabularImputer(model.predict_proba, X_train)
        self.explainer_obj = fastshap.FastSHAP(self.surrogate, imputer, loss='mse')
        self.explainer_obj.train(
            X_train, X_train,
            batch_size=64, max_epochs=100, verbose=False
        )

    def _build_mlp(self, input_dim: int) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, input_dim)
        )

    def explain_fast(self, features: dict) -> dict:
        """< 5ms — use this for What-If live updates."""
        X = torch.tensor([list(features.values())], dtype=torch.float32)
        shap_vals = self.explainer_obj.shap_values(X).detach().numpy()[0]
        return dict(zip(LEARNER_FEATURES.keys(), shap_vals.tolist()))
```

**Use FastSHAP for What-If UI** (real-time). Use TreeSHAP for stored explanations (accuracy).

---

## 3. Archipelago — Feature Interaction Detection ⭐ NOVEL

**Type:** Interaction Attribution (Novel — Tsang et al. 2020)  
**MVP Requirement:** Concept-level explanation, Diagnosis layer  
**Library:** `archdetect` or custom (see below)

### Why Archipelago?
Standard SHAP treats features independently. But in education, **interactions matter**:
- `low quiz_score` alone → moderate risk
- `low quiz_score` + `high days_since_login` → catastrophic risk  

Archipelago decomposes the prediction into **main effects + pairwise interactions**,
letting you surface: "It's not just your quiz scores — it's the *combination* of low scores
and long inactivity that's putting you at high risk."

### Implementation
```python
# Archipelago interaction detection
from itertools import combinations
import numpy as np

class ArchipelagoExplainer:
    """
    Detects top pairwise feature interactions using
    SHAP interaction values (available from TreeExplainer).
    """
    def __init__(self, tree_explainer: shap.TreeExplainer):
        self.explainer = tree_explainer

    def get_interactions(self, features: dict, top_k: int = 5) -> list[dict]:
        X = np.array([list(features.values())])
        # shap_interaction_values: shape (1, n_features, n_features)
        interaction_values = self.explainer.shap_interaction_values(X)
        vals = interaction_values[0]  # (n_features, n_features)

        feature_names = list(LEARNER_FEATURES.keys())
        interactions = []
        for i, j in combinations(range(len(feature_names)), 2):
            interaction_score = float(vals[i, j])  # synergistic contribution
            interactions.append({
                "feature_a": feature_names[i],
                "feature_b": feature_names[j],
                "interaction_score": interaction_score,
                "direction": "amplifying" if interaction_score > 0 else "dampening"
            })

        # Sort by absolute interaction magnitude
        interactions.sort(key=lambda x: abs(x["interaction_score"]), reverse=True)
        return interactions[:top_k]

    def to_narrative(self, interactions: list[dict]) -> str:
        """For LLM narration input — returns structured text, NOT the explanation itself."""
        top = interactions[0]
        return (
            f"Top interaction: {top['feature_a']} × {top['feature_b']} "
            f"(score: {top['interaction_score']:.3f}, type: {top['direction']})"
        )
```

---

## 4. DiCE — Diverse Counterfactual Explanations

**Type:** Counterfactual, Prescriptive  
**MVP Requirement:** Counterfactual Engine (mandatory), Action Recommendations  
**Library:** `dice-ml`

### Implementation with Actionability Constraints
```python
import dice_ml
from dice_ml import Dice

class DiCEExplainer:
    def __init__(self, model, X_train: pd.DataFrame):
        self.d = dice_ml.Data(
            dataframe=X_train,
            continuous_features=list(LEARNER_FEATURES.keys()),
            outcome_name="dropout_risk"
        )
        self.m = dice_ml.Model(model=model, backend="sklearn")
        self.dice = Dice(self.d, self.m, method="random")

    def get_counterfactuals(
        self,
        features: dict,
        desired_class: int = 0,  # flip to low-risk
        n_cfs: int = 3
    ) -> CounterfactualResult:
        query = pd.DataFrame([features])

        cf = self.dice.generate_counterfactuals(
            query,
            total_CFs=n_cfs,
            desired_class=desired_class,
            permitted_range={
                # Only allow actionable features to change
                f: [0, 10] for f in ACTIONABLE_FEATURES
            },
            features_to_vary=ACTIONABLE_FEATURES  # immutable features locked
        )

        # Extract best (closest) counterfactual
        best_cf = cf.cf_examples_list[0].final_cfs_df.iloc[0].to_dict()
        changed = {
            k: (features[k], best_cf[k])
            for k in ACTIONABLE_FEATURES
            if abs(features.get(k, 0) - best_cf.get(k, 0)) > 0.01
        }
        return self._build_result(features, best_cf, changed)

    def _build_result(self, original, cf_features, changed) -> CounterfactualResult:
        actions = []
        for rank, (feature, (orig, target)) in enumerate(
            sorted(changed.items(), key=lambda x: abs(x[1][1] - x[1][0]), reverse=True), 1
        ):
            actions.append(PrescriptiveAction(
                feature=feature,
                current_value=orig,
                target_value=target,
                plain_language=self._feature_to_plain(feature, orig, target),
                estimated_impact=abs(target - orig) * FEATURE_IMPACT_WEIGHTS.get(feature, 0.1),
                priority_rank=rank,
                is_causal=feature in CAUSAL_FEATURES  # from DoWhy annotation
            ))
        return CounterfactualResult(
            original_features=original,
            counterfactual_features=cf_features,
            changed_features=changed,
            actions=actions
        )

    def _feature_to_plain(self, feature: str, orig: float, target: float) -> str:
        templates = {
            "login_frequency_weekly": f"Log in {target:.0f}x per week (currently {orig:.0f}x)",
            "quiz_completion_rate": f"Complete {target*100:.0f}% of quizzes (currently {orig*100:.0f}%)",
            "days_since_last_activity": f"Return to the platform within {target:.0f} days",
            "assignment_submission_rate": f"Submit {target*100:.0f}% of assignments",
            "forum_posts_count": f"Post {target:.0f} times in forums",
        }
        return templates.get(feature, f"Change {feature} from {orig:.2f} to {target:.2f}")
```

---

## 5. Contrastive Explanation Method (CEM) ⭐ NOVEL

**Type:** Pertinent Positives + Pertinent Negatives, Contrastive  
**MVP Requirement:** Diagnosis layer — "why this prediction AND why not the opposite"  
**Library:** `alibi` (`alibi.explainers.CEM`)

### Why CEM?
CEM answers two questions standard SHAP cannot:
- **PP (Pertinent Positive):** "What minimal subset of features is *sufficient* to produce this prediction?"
- **PN (Pertinent Negative):** "What minimal feature changes would *flip* this prediction?"

For educators: "This student is flagged as at-risk **because of** low quiz completion (PP).
They would **not** be flagged if they logged in 3x/week (PN)."

This maps directly to the **Diagnosis → Prescription** flow.

### Implementation
```python
from alibi.explainers import CEM
import tensorflow as tf  # CEM requires a differentiable model

class CEMExplainer:
    """
    Note: CEM requires a differentiable model.
    Wrap sklearn model in a simple Keras surrogate for gradient access.
    """
    def __init__(self, keras_surrogate, X_train: np.ndarray):
        self.explainer = CEM(
            keras_surrogate,
            mode='PN',           # Start with Pertinent Negative
            shape=X_train.shape[1:],
            kappa=0.1,           # confidence margin
            beta=0.1,            # L1 regularization
            gamma=100,           # AE reconstruction loss weight
            theta=100,           # prediction loss weight
            max_iterations=1000,
            feature_range=(X_train.min(axis=0), X_train.max(axis=0))
        )
        self.explainer.fit(X_train, no_info_type='median')
        self.feature_names = list(LEARNER_FEATURES.keys())

    def explain(self, features: dict) -> dict:
        X = np.array([list(features.values())], dtype=np.float32)

        # Pertinent Negative: minimal change to flip prediction
        pn_result = self.explainer.explain(X, mode='PN')
        # Pertinent Positive: minimal features sufficient for current prediction
        pp_result = self.explainer.explain(X, mode='PP')

        return {
            "pertinent_negative": self._to_feature_dict(pn_result.PN[0]),
            "pertinent_positive": self._to_feature_dict(pp_result.PP[0]),
            "pn_interpretation": "Features to change to reduce risk",
            "pp_interpretation": "Core features driving current risk prediction",
        }

    def _to_feature_dict(self, values: np.ndarray) -> dict:
        return dict(zip(self.feature_names, values.tolist()))
```

---

## 6. Anchors — Rule-Based High-Precision Explanations

**Type:** Rule Extraction, Local, High-Precision  
**MVP Requirement:** Instructor view — precise IF-THEN rules  
**Library:** `alibi.explainers.AnchorTabular`

### Why Anchors?
Anchors produce IF-THEN rules that are **guaranteed to hold with ≥ threshold precision**.
E.g.: *"IF quiz_completion_rate < 0.4 AND days_since_last_activity > 5 THEN dropout_risk = HIGH (precision: 94%)"*

This is the **most readable explanation type for instructors** — actionable, precise, no numbers.

### Implementation
```python
from alibi.explainers import AnchorTabular

class AnchorsExplainer:
    def __init__(self, model, X_train: np.ndarray, feature_names: list[str]):
        self.explainer = AnchorTabular(
            predictor=model.predict,
            feature_names=feature_names,
            discretizer='quartile'
        )
        self.explainer.fit(X_train)
        self.feature_names = feature_names

    def explain(self, features: dict) -> dict:
        X = np.array([list(features.values())])
        explanation = self.explainer.explain(X, threshold=0.90)

        return {
            "anchor_rule": " AND ".join(explanation.anchor),
            "precision": float(explanation.precision),
            "coverage": float(explanation.coverage),
            "human_readable": self._format_rule(explanation.anchor, explanation.precision)
        }

    def _format_rule(self, anchor: list[str], precision: float) -> str:
        rule = " AND ".join(anchor)
        return f"IF {rule} THEN at-risk prediction holds with {precision*100:.0f}% certainty"
```

---

## 7. Prototype-Based Explanations (ProtoDash) ⭐ NOVEL

**Type:** Example-Based, Instance-Level  
**MVP Requirement:** Concept-level explanation — "similar learner" context  
**Library:** `aix360` (`aix360.algorithms.protodash`)

### Why Prototypes?
Numbers and rules can be alienating. Prototypes say:
*"Your learning profile is 89% similar to 3 past students in this course.
2 of them successfully completed after increasing weekly quiz attempts."*

This is the **highest-comprehension explanation type for learners** — narrative, relatable, motivational.

### Implementation
```python
from aix360.algorithms.protodash import ProtodashExplainer as _ProtodashExplainer
from sklearn.preprocessing import StandardScaler
import numpy as np

class PrototypeExplainer:
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, learner_ids: list[str]):
        self.scaler = StandardScaler().fit(X_train)
        self.X_scaled = self.scaler.transform(X_train)
        self.y_train = y_train
        self.learner_ids = learner_ids
        self.protodash = _ProtodashExplainer()

    def explain(self, features: dict, n_prototypes: int = 3) -> dict:
        X = self.scaler.transform(np.array([list(features.values())]))

        # Find closest prototypes from same predicted class
        weights, _, _ = self.protodash.explain(X, self.X_scaled, m=n_prototypes)
        proto_indices = np.argsort(-weights)[:n_prototypes]

        prototypes = []
        for idx in proto_indices:
            similarity = 1.0 - np.linalg.norm(X[0] - self.X_scaled[idx]) / 10.0
            prototypes.append({
                "learner_id": self.learner_ids[idx],
                "outcome": "completed" if self.y_train[idx] == 0 else "dropped_out",
                "similarity_score": float(np.clip(similarity, 0, 1)),
                "distinguishing_features": self._get_diff_features(
                    list(features.values()), self.X_scaled[idx]
                )
            })

        return {
            "prototypes": prototypes,
            "narrative": self._build_narrative(prototypes)
        }

    def _get_diff_features(self, current, prototype_scaled) -> list[str]:
        diffs = np.abs(np.array(current) - prototype_scaled)
        top_diff_idx = np.argsort(-diffs)[:3]
        return [list(LEARNER_FEATURES.keys())[i] for i in top_diff_idx]

    def _build_narrative(self, prototypes: list) -> str:
        successful = [p for p in prototypes if p["outcome"] == "completed"]
        if successful:
            return (
                f"Your profile is similar to {len(successful)} learner(s) who completed this course. "
                f"Their key difference: {successful[0]['distinguishing_features'][0].replace('_', ' ')}."
            )
        return "Your profile matches learners who needed additional support. See recommended actions below."
```

---

## 8. Monte Carlo Dropout — Epistemic Uncertainty ⭐ NOVEL

**Type:** Uncertainty Quantification  
**MVP Requirement:** Trust Score, Stability Indicator  
**Library:** Custom with `torch` or `scikit-learn` calibration

### Why Uncertainty?
A prediction of 0.71 risk score means nothing without knowing if the model is *confident*.
MC Dropout runs N forward passes with dropout active → distribution over predictions.

For sklearn, approximate with **MAPIE** (conformal prediction intervals):
```python
from mapie.classification import MapieClassifier

class UncertaintyEstimator:
    def __init__(self, base_model):
        self.mapie = MapieClassifier(
            estimator=base_model,
            method="score",
            cv=5,
            random_state=42
        )

    def fit(self, X_train, y_train):
        self.mapie.fit(X_train, y_train)

    def predict_with_uncertainty(self, X: np.ndarray, alpha: float = 0.1):
        """Returns prediction + conformal prediction interval."""
        y_pred, y_pset = self.mapie.predict(X, alpha=alpha, include_last_label=True)
        confidence_width = float(y_pset[0, :, 0].sum())  # set size = uncertainty proxy
        return {
            "prediction": int(y_pred[0]),
            "confidence_interval": y_pset.tolist(),
            "uncertainty": confidence_width,  # 0 = certain, 1 = uncertain
        }
```

---

## 9. Causal Inference Layer — DoWhy + do-Calculus ⭐ BROWNIE

**Type:** Causal Reasoning, SCM  
**MVP Requirement:** Brownie — Causal DAG with do-calculus annotations  
**Library:** `dowhy`, `pgmpy`

### Learner Causal DAG
```
prior_grades ──────────────────────────────► dropout_risk
      │                                           ▲
      ▼                                           │
engagement_score ──► assessment_performance ─────┘
      ▲                     ▲
      │                     │
login_frequency      quiz_completion_rate
      │                     │
avg_session_duration  assignment_submission_rate
```

### Implementation
```python
import dowhy
from dowhy import CausalModel

class CausalAnnotator:
    """
    Builds a causal DAG over learner features.
    Annotates SHAP features as causal vs correlational.
    """
    CAUSAL_GRAPH = """
    digraph {
        prior_course_completions -> quiz_avg_score;
        prior_course_completions -> assignment_submission_rate;
        login_frequency_weekly -> video_completion_rate;
        login_frequency_weekly -> forum_posts_count;
        avg_session_duration_min -> quiz_completion_rate;
        quiz_completion_rate -> quiz_avg_score;
        assignment_submission_rate -> dropout_risk;
        quiz_avg_score -> dropout_risk;
        days_since_last_activity -> dropout_risk;
        forum_posts_count -> dropout_risk;
        missed_deadlines_count -> dropout_risk;
    }
    """

    def __init__(self, data: pd.DataFrame):
        self.model = CausalModel(
            data=data,
            treatment="login_frequency_weekly",  # example treatment
            outcome="dropout_risk",
            graph=self.CAUSAL_GRAPH
        )
        # Pre-compute causal vs correlational classification
        self.causal_features = self._identify_causal_features()

    def _identify_causal_features(self) -> list[str]:
        """Features that have a causal path to dropout_risk in the DAG."""
        return [
            "assignment_submission_rate",
            "quiz_completion_rate",
            "days_since_last_activity",
            "missed_deadlines_count",
            "login_frequency_weekly"
        ]

    def annotate_shap(self, shap_values: dict) -> dict:
        """Annotate each SHAP feature as causal or correlational."""
        return {
            feature: {
                "shap_value": value,
                "type": "causal" if feature in self.causal_features else "correlational",
                "note": "Directly influences dropout risk" if feature in self.causal_features
                        else "Associated with dropout risk but may not cause it"
            }
            for feature, value in shap_values.items()
        }

    def estimate_causal_effect(self, treatment_feature: str, treatment_value: float) -> float:
        """Use do-calculus to estimate effect of intervention."""
        identified_estimand = self.model.identify_effect(proceed_when_unidentifiable=True)
        estimate = self.model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_weighting"
        )
        return float(estimate.value)
```

---

## 10. Explanation Drift Detector

**Type:** Distribution Monitoring, Consistency Tracking  
**MVP Requirement:** Explanation Consistency & Tracking (mandatory)  
**Library:** `alibi-detect` + custom JSD

```python
from scipy.spatial.distance import jensenshannon
import numpy as np

class ExplanationDriftDetector:
    """
    Tracks SHAP attribution distributions over time.
    Flags when explanations become inconsistent.
    """
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.history: dict[str, list] = {}  # learner_id → list of shap_dicts

    def add_explanation(self, learner_id: str, shap_values: dict):
        if learner_id not in self.history:
            self.history[learner_id] = []
        self.history[learner_id].append(shap_values)
        # Keep only recent window
        self.history[learner_id] = self.history[learner_id][-self.window_size:]

    def check_drift(self, learner_id: str) -> dict:
        history = self.history.get(learner_id, [])
        if len(history) < 2:
            return {"drift_detected": False, "jsd": 0.0, "flag": None}

        # Compare most recent to previous
        prev_vals = np.array(list(history[-2].values()))
        curr_vals = np.array(list(history[-1].values()))

        # Normalize to probability distributions
        prev_prob = np.abs(prev_vals) / (np.abs(prev_vals).sum() + 1e-8)
        curr_prob = np.abs(curr_vals) / (np.abs(curr_vals).sum() + 1e-8)

        jsd = float(jensenshannon(prev_prob, curr_prob))
        drift_detected = jsd > 0.15  # threshold

        # Check rank shift in top-3 features
        prev_top3 = sorted(history[-2], key=lambda k: abs(history[-2][k]), reverse=True)[:3]
        curr_top3 = sorted(history[-1], key=lambda k: abs(history[-1][k]), reverse=True)[:3]
        rank_shift = len(set(prev_top3) - set(curr_top3))

        return {
            "drift_detected": drift_detected or rank_shift > 1,
            "jsd": jsd,
            "rank_shift": rank_shift,
            "flag": "⚠️ Explanation changed significantly" if drift_detected else None,
            "prev_top3": prev_top3,
            "curr_top3": curr_top3
        }
```

---

## 11. Trust Score Composite

**MVP Requirement:** Stability indicator per explanation (mandatory)

```python
class TrustScorer:
    """
    Composite trust score = weighted(fidelity, stability, completeness)
    """
    WEIGHTS = {"fidelity": 0.4, "stability": 0.35, "completeness": 0.25}

    def score(
        self,
        shap_values: dict,
        model,
        features: dict,
        stability: float,
        top_k: int = 5
    ) -> dict:
        fidelity = self._fidelity(shap_values, model, features)
        completeness = self._completeness(shap_values, top_k)
        composite = (
            self.WEIGHTS["fidelity"] * fidelity +
            self.WEIGHTS["stability"] * stability +
            self.WEIGHTS["completeness"] * completeness
        )
        return {
            "trust_score": round(composite, 3),
            "fidelity": round(fidelity, 3),
            "stability": round(stability, 3),
            "completeness": round(completeness, 3),
            "label": "High" if composite > 0.75 else "Medium" if composite > 0.5 else "Low"
        }

    def _fidelity(self, shap_values: dict, model, features: dict) -> float:
        """How well SHAP sum approximates actual model output."""
        base = self.base_value  # from explainer
        shap_sum = sum(shap_values.values())
        predicted = model.predict_proba([list(features.values())])[0][1]
        error = abs((base + shap_sum) - predicted)
        return float(np.clip(1.0 - error, 0.0, 1.0))

    def _completeness(self, shap_values: dict, top_k: int) -> float:
        """What fraction of total attribution is explained by top-k features."""
        vals = np.abs(list(shap_values.values()))
        top_k_sum = np.sort(vals)[-top_k:].sum()
        total = vals.sum()
        return float(top_k_sum / (total + 1e-8))
```

---

## Algorithm Selection Guide

| Use Case | Algorithm | Speed | Faithfulness |
|----------|-----------|-------|-------------|
| Feature attribution (stored) | TreeSHAP | Fast | ★★★★★ |
| Feature attribution (live/What-If) | FastSHAP | ⚡ Real-time | ★★★★☆ |
| Concept-level (grouped) | LIME over concept groups | Medium | ★★★☆☆ |
| Feature interactions | Archipelago (SHAP interaction) | Medium | ★★★★☆ |
| Counterfactuals + actions | DiCE | Medium | ★★★★★ |
| Diagnosis (PP + PN) | CEM | Slow | ★★★★☆ |
| Rule-based (instructor) | Anchors | Medium | ★★★★☆ |
| Similar learner | ProtoDash | Fast | ★★★☆☆ |
| Uncertainty | MAPIE conformal | Fast | ★★★★☆ |
| Causal annotation | DoWhy | Slow (one-time) | ★★★★★ |
| Drift detection | JSD on SHAP history | Fast | ★★★★☆ |
| Trust scoring | Composite metric | Fast | ★★★★☆ |
