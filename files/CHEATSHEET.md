# XAI System — Quick Reference Cheat Sheet

## ⚡ Most Important Rules
1. Explanations from MODEL INTERNALS only (SHAP, DiCE, CEM) — LLM only narrates
2. What-If → model.predict_proba() → NEVER heuristic
3. DiCE must receive ACTIONABLE_FEATURES list — immutable features locked
4. Store EVERY explanation in ExplanationStore after generation
5. Run TrustScorer after EVERY explanation

---

## Algorithm → MVP Requirement Map

| Requirement | Algorithm | File |
|-------------|-----------|------|
| Feature-level explanation | TreeSHAP | shap_explainer.py |
| What-If live updates | FastSHAP | fastshap_explainer.py |
| Concept-level | LIME + Prototypes | lime_explainer.py + prototype_explainer.py |
| Feature interactions | Archipelago | archipelago.py |
| Stability indicator | SHAP noise variance | shap_explainer.py |
| Instructor rule view | Anchors | anchors_explainer.py |
| Diagnosis (PP/PN) | CEM | cem_explainer.py |
| Counterfactuals | DiCE | dice_explainer.py |
| Prescriptive actions | ActionRanker | action_ranker.py |
| Consistency tracking | JSD + ExplanationStore | drift_detector.py |
| Trust score | Fidelity+Stability+Completeness | trust_scorer.py |
| Causal annotation | DoWhy | causal_annotator.py |
| Uncertainty | MAPIE | uncertainty_estimator.py |
| LLM narration | Anthropic SDK | llm_narrator.py |

---

## Critical Imports

```python
# SHAP
import shap
explainer = shap.TreeExplainer(gbm_model)
shap_values = explainer.shap_values(X)  # list[ndarray] for binary
vals = shap_values[1][0]  # class-1 values for first sample

# DiCE
import dice_ml
d = dice_ml.Data(dataframe=df, continuous_features=..., outcome_name="dropout_risk")
m = dice_ml.Model(model=sklearn_model, backend="sklearn")
dice = dice_ml.Dice(d, m, method="random")
cf = dice.generate_counterfactuals(query_df, total_CFs=3, desired_class=0,
                                    features_to_vary=ACTIONABLE_FEATURES)

# Anchors
from alibi.explainers import AnchorTabular
exp = AnchorTabular(model.predict, feature_names, discretizer='quartile')
exp.fit(X_train)
result = exp.explain(X, threshold=0.90)
rule = " AND ".join(result.anchor)

# CEM
from alibi.explainers import CEM
cem = CEM(keras_model, mode='PN', shape=(12,), ...)
cem.fit(X_train)
result = cem.explain(X)

# MAPIE
from mapie.classification import MapieClassifier
mapie = MapieClassifier(estimator=base_model, method="score", cv=5)
mapie.fit(X_train, y_train)
y_pred, y_pset = mapie.predict(X, alpha=0.1, include_last_label=True)

# DoWhy
from dowhy import CausalModel
model = CausalModel(data=df, treatment="feature", outcome="dropout_risk", graph=DAG_STRING)

# FastSHAP
# pip install fastshap
import fastshap
imputer = fastshap.TabularImputer(model.predict_proba, X_train)
fs = fastshap.FastSHAP(surrogate_nn, imputer, loss='mse')
fs.train(X_train, X_train, batch_size=64, max_epochs=100)
```

---

## Feature Reference

```python
LEARNER_FEATURES = {
    "login_frequency_weekly": float,
    "avg_session_duration_min": float,
    "forum_posts_count": int,
    "video_completion_rate": float,      # 0-1
    "quiz_avg_score": float,             # 0-100
    "quiz_completion_rate": float,       # 0-1
    "assignment_submission_rate": float, # 0-1
    "days_since_last_activity": int,
    "prior_course_completions": int,
    "current_week_in_course": int,
    "missed_deadlines_count": int,
    "help_requests_count": int,
}

ACTIONABLE_FEATURES = [
    "login_frequency_weekly", "avg_session_duration_min",
    "forum_posts_count", "video_completion_rate",
    "quiz_completion_rate", "assignment_submission_rate",
    "days_since_last_activity", "help_requests_count"
]

IMMUTABLE_FEATURES = ["prior_course_completions", "current_week_in_course"]

CAUSAL_FEATURES = [
    "assignment_submission_rate", "quiz_completion_rate",
    "days_since_last_activity", "missed_deadlines_count", "login_frequency_weekly"
]

FEATURE_ACTIONABILITY_WEIGHTS = {
    "assignment_submission_rate": 1.0,
    "quiz_completion_rate": 0.95,
    "login_frequency_weekly": 0.90,
    "days_since_last_activity": 0.85,
    "forum_posts_count": 0.80,
    "video_completion_rate": 0.75,
    "avg_session_duration_min": 0.70,
    "help_requests_count": 0.65,
}
```

---

## Trust Score Formula

```
Trust = 0.40 × Fidelity + 0.35 × Stability + 0.25 × Completeness

Fidelity = 1 - |predict_proba[1] - (sum(shap_values) + base_value)|
Stability = 1 - std(shap_rank_under_gaussian_noise, n=20) / (mean + ε)
Completeness = sum(|top5_shap|) / sum(|all_shap|)

Bands: ≥0.75 → High | 0.50-0.74 → Medium | <0.50 → Low
```

---

## Drift Detection Formula

```
JSD = jensenshannon(normalize(|prev_shap|), normalize(|curr_shap|))
Rank Shift = |set(prev_top3) - set(curr_top3)|

Drift Detected = JSD > 0.15 OR rank_shift > 1
```

---

## Action Priority Formula

```
priority_score = shap_magnitude
              × FEATURE_ACTIONABILITY_WEIGHTS[feature]
              × (1.0 if feature in CAUSAL_FEATURES else 0.7)

Sort actions descending by priority_score → rank 1 = highest impact
estimated_impact = |predict_proba(counterfactual) - predict_proba(original)|
```

---

## LLM Narration Prompt Template

```python
LEARNER_PROMPT = """
You are a supportive learning coach narrating a pre-computed AI analysis.
ONLY describe what is in the data below. Do NOT invent explanations.
Keep response under 100 words. End with exactly one actionable sentence.

Data:
- Risk level: {risk_label} (score: {risk_score})
- Top 3 risk factors: {top3_features}
- Anchor rule: {anchor_rule}
- Top recommended action: {top_action_plain_language}
- Similar learner outcome: {prototype_narrative}
"""

INSTRUCTOR_PROMPT = """
You are an educational data analyst summarizing pre-computed XAI results.
Be precise and data-driven. Under 150 words.

Data:
- Risk score: {risk_score} ({risk_label}), Model confidence: {model_confidence}
- SHAP top 3: {shap_top3_with_values}
- Anchor rule: {anchor_rule} (precision: {anchor_precision})
- Causal factors: {causal_features}
- Correlational factors: {correlational_features}
- Trust score: {trust_score} ({trust_label})
- Top intervention: {top_action}
"""
```

---

## Copilot Prompts for Each Module

### shap_explainer.py
> "Implement SHAPExplainer with TreeExplainer. Methods: explain(features_dict), stability_score(features_dict, n_runs=20), global_summary(X). SHAP sum + base_value should ≈ predict_proba output."

### dice_explainer.py
> "DiCE counterfactuals with features_to_vary=ACTIONABLE_FEATURES. Return ranked PrescriptiveAction list with plain_language and estimated_impact."

### anchors_explainer.py
> "AnchorTabular with quartile discretizer, threshold=0.90. Return anchor_rule string + precision + coverage."

### cem_explainer.py
> "CEM in PP and PN modes using Keras surrogate. Return pertinent_positives and pertinent_negatives as feature dicts."

### prototype_explainer.py
> "NearestNeighbors cosine similarity to find 3 most similar historical learners. Return similarity scores, outcomes, and motivational narrative."

### archipelago.py
> "shap_interaction_values() from TreeExplainer. Return top-5 pairwise interactions with direction (amplifying/dampening)."

### trust_scorer.py
> "Composite trust score: 0.4×fidelity + 0.35×stability + 0.25×completeness. Return {trust_score, fidelity, stability, completeness, label}."

### drift_detector.py
> "Jensen-Shannon Divergence on SHAP distributions. Drift if JSD>0.15 or top3 rank shifts >1 feature."

### llm_narrator.py
> "Anthropic SDK Claude Sonnet. Receives pre-computed data only. narrate_for_learner and narrate_for_instructor. System prompt enforces no fabrication."

---

## Test Commands

```bash
# Unit tests
pytest tests/ -v

# Integration test
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"learner_id":"test","features":{"login_frequency_weekly":1,"quiz_completion_rate":0.2,"assignment_submission_rate":0.3,"days_since_last_activity":12,"avg_session_duration_min":10,"forum_posts_count":0,"video_completion_rate":0.3,"quiz_avg_score":45,"prior_course_completions":0,"current_week_in_course":5,"missed_deadlines_count":4,"help_requests_count":0}}'

# Health check
curl localhost:8000/health

# Docker
docker-compose up --build
curl localhost:8000/health
```
