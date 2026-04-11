# Implementation Plan — XAI Learning Recommendation System
# 24-Hour Hackathon Sprint

> Copy this file into your GitHub Copilot workspace.
> Use each task block as a prompt: describe the task, paste the relevant
> section, and let Copilot generate the implementation.

---

## Pre-Sprint Setup (30 mins)

### Environment
```bash
# Create project
mkdir xai-learning-system && cd xai-learning-system
python -m venv .venv && source .venv/bin/activate

# Install all dependencies
pip install \
  scikit-learn numpy pandas scipy \
  shap fastshap dice-ml alibi alibi-detect \
  dowhy pgmpy mapie \
  aix360 \
  fastapi uvicorn[standard] sqlalchemy \
  anthropic \
  torch torchvision \
  plotly dash explainerdashboard \
  pytest httpx

# Frontend
npx create-react-app frontend --template typescript
cd frontend && npm install recharts tailwindcss @headlessui/react axios
```

### Generate Synthetic Data First
```bash
# Run this before anything else — every module depends on data
python backend/app/model/mock_data.py
# Creates: data/learners.csv, data/train.pkl, data/test.pkl
```

---

## Phase 1 — Foundation (Hours 1–3)

**Goal:** Working model + SHAP + basic API running

---

### Task 1.1 — Synthetic Data Generator
**File:** `backend/app/model/mock_data.py`

Copilot prompt:
> "Generate 1000 synthetic learner records with these features: login_frequency_weekly (1-7),
> avg_session_duration_min (5-120), forum_posts_count (0-50), video_completion_rate (0-1),
> quiz_avg_score (0-100), quiz_completion_rate (0-1), assignment_submission_rate (0-1),
> days_since_last_activity (0-30), prior_course_completions (0-10), current_week_in_course (1-12),
> missed_deadlines_count (0-10), help_requests_count (0-20).
> Label: dropout_risk=1 if assignment_submission_rate < 0.4 OR (quiz_completion_rate < 0.5
> AND days_since_last_activity > 7) OR missed_deadlines_count > 4, else 0.
> Add realistic noise. Save as CSV and train/test split pickles."

```python
# Expected output structure
{
    "n_samples": 1000,
    "dropout_rate": "~30%",   # realistic imbalance
    "files": ["data/learners.csv", "data/train.pkl", "data/test.pkl"],
    "learner_ids": ["learner_001", ..., "learner_1000"]
}
```

---

### Task 1.2 — Model Trainer
**File:** `backend/app/model/trainer.py`

Copilot prompt:
> "Train a GradientBoostingClassifier on the learner data with hyperparameter tuning via
> GridSearchCV. Also train a RandomForestClassifier as the second model for multi-model
> comparison. Calibrate probabilities with CalibratedClassifierCV. Save both models to
> models/gbm.pkl and models/rf.pkl. Print classification report and AUC-ROC score."

Key requirements:
- Use `GradientBoostingClassifier` as primary (TreeSHAP compatible)
- Use `RandomForestClassifier` as secondary (for multi-model comparison brownie)
- Calibrate with `CalibratedClassifierCV` for reliable `predict_proba`
- Save feature names list alongside model — SHAP needs it

---

### Task 1.3 — SHAP Explainer Module
**File:** `backend/app/explainers/shap_explainer.py`

Copilot prompt:
> "Implement SHAPExplainer class using shap.TreeExplainer wrapping the GBM model.
> Methods: explain(features_dict) → shap_values dict + base_value,
> stability_score(features_dict, n_runs=20) → float 0-1,
> global_summary(X_array) → mean absolute SHAP per feature.
> See XAI_ALGORITHMS.md section 1 for full implementation."

Validate with:
```python
# Test: SHAP sum should approximately equal prediction - base_value
pred = model.predict_proba([X])[0][1]
assert abs(sum(shap_vals.values()) + base_value - pred) < 0.05, "SHAP fidelity check failed"
```

---

### Task 1.4 — FastAPI Base + /predict Endpoint
**File:** `backend/app/main.py`

Copilot prompt:
> "Create FastAPI app with:
> GET /health → {status: ok, model_version, timestamp}
> POST /predict → accepts LearnerFeatures pydantic model, returns PredictionResult
> POST /explain → accepts learner_id + features, returns ExplanationResult
> POST /whatif → accepts features + modifications dict, returns live prediction + SHAP
> POST /counterfactual → accepts features, returns CounterfactualResult
> GET /history/{learner_id} → returns explanation timeline
> Use lifespan context manager to load models once at startup."

```python
# Pydantic models for type safety
class LearnerFeatures(BaseModel):
    login_frequency_weekly: float = Field(ge=0, le=14)
    avg_session_duration_min: float = Field(ge=0, le=300)
    forum_posts_count: int = Field(ge=0)
    video_completion_rate: float = Field(ge=0.0, le=1.0)
    quiz_avg_score: float = Field(ge=0.0, le=100.0)
    quiz_completion_rate: float = Field(ge=0.0, le=1.0)
    assignment_submission_rate: float = Field(ge=0.0, le=1.0)
    days_since_last_activity: int = Field(ge=0)
    prior_course_completions: int = Field(ge=0)
    current_week_in_course: int = Field(ge=1, le=52)
    missed_deadlines_count: int = Field(ge=0)
    help_requests_count: int = Field(ge=0)
```

**Checkpoint:** `curl localhost:8000/health` returns `{status: ok}` ✅

---

## Phase 2 — Core XAI Engine (Hours 3–8)

**Goal:** All 7 MVP components implemented

---

### Task 2.1 — DiCE Counterfactual Engine
**File:** `backend/app/explainers/dice_explainer.py`

Copilot prompt:
> "Implement DiCEExplainer wrapping dice-ml. Must accept actionability_constraints that
> lock immutable features (prior_course_completions, current_week_in_course).
> Generate 3 diverse counterfactuals. Return the closest one as best_cf with:
> changed_features dict, estimated_impact (delta predict_proba), and ranked PrescriptiveActions.
> Plain language templates for each feature. See XAI_ALGORITHMS.md section 4."

Test:
```python
result = dice_explainer.get_counterfactuals(high_risk_learner_features)
assert len(result.actions) >= 1
assert result.actions[0].priority_rank == 1
assert all(f not in result.changed_features for f in IMMUTABLE_FEATURES)
```

---

### Task 2.2 — Anchors Rule Explainer
**File:** `backend/app/explainers/anchors_explainer.py`

Copilot prompt:
> "Implement AnchorsExplainer using alibi.explainers.AnchorTabular with quartile discretizer.
> explain(features_dict) returns anchor_rule string, precision float, coverage float,
> and human_readable string. Fit on X_train at init. Threshold: 0.90 precision."

---

### Task 2.3 — CEM Contrastive Explainer
**File:** `backend/app/explainers/cem_explainer.py`

Copilot prompt:
> "Build a Keras MLP surrogate wrapping the sklearn GBM model (same predictions).
> Use this surrogate to instantiate alibi.explainers.CEM in both PP and PN modes.
> explain(features) returns pertinent_positives dict, pertinent_negatives dict,
> and human_readable strings for each. See XAI_ALGORITHMS.md section 5."

```python
# Keras surrogate for CEM
def build_keras_surrogate(sklearn_model, X_train, y_train):
    """Train a small neural net to mimic sklearn model predictions."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(12,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    # Use sklearn soft labels as targets
    soft_labels = sklearn_model.predict_proba(X_train)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, soft_labels, epochs=50, batch_size=32, verbose=0)
    return model
```

---

### Task 2.4 — Prototype Explainer
**File:** `backend/app/explainers/prototype_explainer.py`

Copilot prompt:
> "Implement PrototypeExplainer using cosine similarity (skip ProtoDash if aix360 install
> is complex; use sklearn NearestNeighbors instead as fallback). Given a learner's feature
> vector, find 3 most similar past learners. Return their outcome (completed/dropped_out),
> similarity score, and the top 2 distinguishing features. Build a motivational narrative
> string. See XAI_ALGORITHMS.md section 7."

Fallback implementation:
```python
from sklearn.neighbors import NearestNeighbors

class PrototypeExplainer:
    def __init__(self, X_train, y_train, learner_ids, feature_names):
        self.scaler = StandardScaler().fit(X_train)
        self.X_scaled = self.scaler.transform(X_train)
        self.nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn.fit(self.X_scaled)
        self.y_train = y_train
        self.learner_ids = learner_ids
        self.feature_names = feature_names
```

---

### Task 2.5 — Archipelago Interaction Detector
**File:** `backend/app/explainers/archipelago.py`

Copilot prompt:
> "Implement ArchipelagoExplainer using shap.TreeExplainer.shap_interaction_values().
> get_interactions(features, top_k=5) returns list of {feature_a, feature_b,
> interaction_score, direction: 'amplifying'|'dampening'}.
> to_narrative(interactions) returns structured text for LLM narration input.
> See XAI_ALGORITHMS.md section 3."

---

### Task 2.6 — Explanation Store (Consistency Tracking)
**File:** `backend/app/tracker/consistency_store.py`

Copilot prompt:
> "Create ExplanationStore using SQLAlchemy with SQLite.
> Schema: ExplanationRecord(id, learner_id, timestamp, top3_features JSON,
> shap_values JSON, risk_score, trust_score).
> Methods: save(learner_id, explanation) → id,
> get_history(learner_id) → list[ExplanationRecord] last 10,
> get_top3_timeline(learner_id) → [{timestamp, top3_features, risk_score}]"

---

### Task 2.7 — Drift Detector
**File:** `backend/app/tracker/drift_detector.py`

Copilot prompt:
> "Implement ExplanationDriftDetector using Jensen-Shannon Divergence.
> compare_explanations(prev_shap, curr_shap) → {jsd, rank_shift, drift_detected, flag}.
> check_learner_drift(learner_id, store) → retrieves last 2 from store and computes drift.
> Threshold: JSD > 0.15 OR top-3 rank change > 1 feature = drift detected.
> See XAI_ALGORITHMS.md section 10."

---

### Task 2.8 — Trust Score
**File:** `backend/app/evaluator/trust_scorer.py`

Copilot prompt:
> "Implement TrustScorer with composite score = 0.4×fidelity + 0.35×stability + 0.25×completeness.
> Fidelity: how well SHAP sum + base_value ≈ predict_proba output (1 - abs error).
> Stability: from SHAPExplainer.stability_score().
> Completeness: fraction of total attribution covered by top-5 features.
> Returns {trust_score, fidelity, stability, completeness, label: High|Medium|Low}.
> See XAI_ALGORITHMS.md section 11."

---

### Task 2.9 — Causal Annotator
**File:** `backend/app/causal/causal_annotator.py`

Copilot prompt:
> "Implement CausalAnnotator using the DAG defined in XAI_ALGORITHMS.md section 9.
> annotate_shap(shap_values) → augments each feature with {type: causal|correlational, note}.
> Use hardcoded causal feature list from DAG analysis (no live DoWhy call needed for MVP).
> For brownie: add estimate_causal_effect(treatment_feature, X_train, y_train) using DoWhy."

---

### Task 2.10 — Action Ranker
**File:** `backend/app/prescriptor/action_ranker.py`

Copilot prompt:
> "Implement ActionRanker that takes DiCE counterfactual actions and ranks them by:
> priority_score = shap_magnitude × actionability_weight × (1 if causal else 0.7).
> FEATURE_ACTIONABILITY_WEIGHTS = {login_frequency_weekly: 0.9, quiz_completion_rate: 0.95,
> assignment_submission_rate: 1.0, days_since_last_activity: 0.8, ...}.
> Returns sorted list of PrescriptiveAction with estimated_impact as delta predict_proba."

---

## Phase 3 — Dual Audience + LLM Narration (Hours 8–11)

**Goal:** Learner vs Instructor views + LLM narration layer

---

### Task 3.1 — LLM Narrator (Narration Only — Not Explanation)
**File:** `backend/app/narrator/llm_narrator.py`

Copilot prompt:
> "Implement LLMNarrator using the Anthropic SDK (claude-sonnet-4-20250514).
> narrate_for_learner(explanation: ExplanationResult) → motivational plain-language string.
> narrate_for_instructor(explanation: ExplanationResult) → technical summary string.
> CRITICAL: The LLM receives ONLY pre-computed data (SHAP values, DiCE actions, anchors).
> It NEVER generates the explanation itself. System prompt must enforce this.
> Prompt includes: top3_shap_features, anchor_rule, counterfactual_actions, risk_score,
> prototype_narrative."

```python
LEARNER_SYSTEM_PROMPT = """
You are a supportive learning coach. You receive pre-computed AI analysis results
(not raw data) and narrate them in encouraging, plain language for a student.
You must ONLY narrate what is in the data provided — never invent new explanations.
Keep responses under 100 words. End with one specific, actionable sentence.
"""

INSTRUCTOR_SYSTEM_PROMPT = """
You are an educational data analyst. You receive pre-computed XAI results and
summarize them concisely for an instructor. Include: top risk factors, confidence level,
and recommended intervention. Be precise and data-driven. Under 150 words.
"""
```

---

### Task 3.2 — Uncertainty Estimator
**File:** `backend/app/evaluator/uncertainty_estimator.py`

Copilot prompt:
> "Implement UncertaintyEstimator using MAPIE (MapieClassifier).
> fit(X_train, y_train). predict_with_uncertainty(X, alpha=0.1) returns
> {prediction, confidence_width, uncertainty_label: 'Low'|'Medium'|'High'}.
> Integrate into /predict endpoint — add uncertainty to PredictionResult."

---

## Phase 4 — Frontend What-If UI (Hours 11–16)

**Goal:** Interactive React dashboard with real model feedback

---

### Task 4.1 — What-If Panel (Most Critical UI)
**File:** `frontend/src/components/WhatIfPanel.jsx`

Copilot prompt:
> "React component with 12 sliders (one per learner feature) using Tailwind.
> On any slider change: debounce 300ms → POST /whatif with full modified features →
> re-render: risk score badge, SHAP bar chart (Recharts), DiCE counterfactual list.
> Show BEFORE vs AFTER side-by-side. Risk badge: green (LOW) / yellow (MEDIUM) / red (HIGH).
> FastSHAP values update the bar chart in real-time."

---

### Task 4.2 — Explanation Card
**File:** `frontend/src/components/ExplanationCard.jsx`

Copilot prompt:
> "Explanation card component with sections:
> 1. Prediction badge + trust score bar
> 2. Top-3 SHAP features (horizontal bar chart, positive=red/negative=green)
> 3. Anchor rule text in monospace box
> 4. Prototype match ('Similar to 2 learners who completed')
> 5. Interaction highlight ('⚠️ Low quiz scores + high inactivity amplify each other')
> 6. LLM narration text (audience-appropriate)"

---

### Task 4.3 — Audience Toggle
**File:** `frontend/src/components/AudienceToggle.jsx`

Copilot prompt:
> "Toggle button switching between 'Learner View' and 'Instructor View'.
> Learner view: motivational language, large risk badge, prototype narrative, top action.
> Instructor view: feature breakdown table, SHAP waterfall, model confidence, anchor rule,
> causal vs correlational annotations, trust score details."

---

### Task 4.4 — Counterfactual View
**File:** `frontend/src/components/CounterfactualView.jsx`

Copilot prompt:
> "Display DiCE counterfactual results as action cards.
> Each card: feature name, current → target value (animated arrow), estimated impact bar,
> plain language instruction, causal badge (🔗 Causal | 〰️ Correlated).
> Ranked by priority. Add 'Apply This Change' button that pre-fills What-If sliders."

---

### Task 4.5 — Consistency Timeline
**File:** `frontend/src/components/ConsistencyTimeline.jsx`

Copilot prompt:
> "Time-series line chart (Recharts LineChart) showing top-3 SHAP feature values over
> sessions for a learner. X-axis: session dates. Y-axis: SHAP value magnitude.
> Red vertical marker when drift_detected=true. Tooltip shows full top-3 at each point."

---

## Phase 5 — Brownie Points (Hours 16–20)

**Goal:** Causal DAG, Multi-Model, Drift alerts, Human-in-the-Loop

---

### Task 5.1 — Causal DAG Visualization
**File:** `frontend/src/components/CausalDAG.jsx`

Copilot prompt:
> "Render the learner causal DAG as an SVG using d3-force or react-flow.
> Nodes: feature names. Edges: causal relationships from CausalAnnotator.
> Color nodes: blue (causal path to dropout), grey (correlational).
> Hovering a node shows its causal effect estimate from DoWhy."

---

### Task 5.2 — Multi-Model Disagreement Panel
**File:** `frontend/src/components/ModelComparison.jsx`

Copilot prompt:
> "Run both GBM and RF models on the same learner. Show:
> - Prediction agreement badge: ✅ Agree | ⚠️ Disagree
> - Side-by-side SHAP bar charts
> - Feature rank disagreement: highlight features where GBM rank ≠ RF rank by >2 positions"

---

### Task 5.3 — Human-in-the-Loop Feedback
**File:** `backend/app/tracker/feedback_store.py`

Copilot prompt:
> "POST /feedback endpoint accepting: learner_id, explanation_id, rating (1-5),
> followed_recommendation (bool), correction ({feature, suggested_value}).
> Store in FeedbackRecord SQLite table. GET /feedback/stats returns aggregate
> recommendation_follow_rate, avg_rating per feature."

---

## Phase 6 — SDK + Docker (Hours 20–23)

---

### Task 6.1 — XAI SDK Package
**File:** `backend/sdk/xai_sdk/`

Copilot prompt:
> "Create installable Python package xai-learner-sdk with:
> XAIClient class wrapping all API calls (predict, explain, counterfactual, whatif).
> Convenience methods: explain_learner(features_dict), get_actions(features_dict).
> pyproject.toml with version 0.1.0. README with quickstart example."

```python
# Expected SDK usage (after pip install xai-learner-sdk)
from xai_sdk import XAIClient

client = XAIClient(base_url="https://your-deployment.com")
explanation = client.explain_learner({
    "login_frequency_weekly": 2.0,
    "quiz_completion_rate": 0.35,
    # ...
})
print(explanation.top_action)
# → "Complete 3 more quizzes this week → estimated 34% risk reduction"
```

---

### Task 6.2 — Docker Setup
**Files:** `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`

Copilot prompt:
> "Create multi-stage Dockerfile for FastAPI backend (python:3.11-slim).
> Dockerfile for React frontend (node:18-alpine + nginx).
> docker-compose.yml with backend (port 8000), frontend (port 3000), both with
> health checks. Backend env vars: MODEL_PATH, DB_URL, ANTHROPIC_API_KEY."

---

## Final Hour — Testing + Polish

### Task 7.1 — Integration Tests
```python
# pytest tests/test_pipeline.py
def test_full_pipeline_high_risk_learner():
    """Test complete P→E→D→P flow for a high-risk learner."""
    features = {
        "login_frequency_weekly": 1.0,
        "quiz_completion_rate": 0.2,
        "assignment_submission_rate": 0.3,
        "days_since_last_activity": 12,
        # ... all features
    }
    # 1. Prediction
    pred = client.post("/predict", json=features).json()
    assert pred["risk_label"] == "HIGH"

    # 2. Explanation
    exp = client.post("/explain", json={"features": features}).json()
    assert len(exp["shap_values"]) == 12
    assert exp["trust_score"]["trust_score"] > 0.0
    assert exp["anchor_rule"] != ""

    # 3. Counterfactual
    cf = client.post("/counterfactual", json={"features": features}).json()
    assert len(cf["actions"]) >= 1
    assert cf["actions"][0]["estimated_impact"] > 0

    # 4. Consistency stored
    history = client.get(f"/history/test_learner").json()
    assert len(history) >= 1
```

### Task 7.2 — Checklist Verification
```
□ POST /predict returns PredictionResult with uncertainty
□ POST /explain returns all 3 explanation levels (feature, concept, prediction)
□ POST /explain returns stability_score and trust_score
□ POST /whatif uses actual model.predict_proba (verified in test)
□ POST /counterfactual returns human-readable actions, no hard-coding
□ GET /history/{id} returns top-3 timeline with drift flags
□ Frontend: learner/instructor toggle works
□ Frontend: What-If sliders update SHAP chart in real-time
□ Docker: docker-compose up serves both services
□ GET /health returns 200
□ SDK installable: pip install -e backend/sdk
```

---

## Pitch Deck Structure (Final 30 mins)

1. **Problem** — Black-box learning platforms, no trust, no action
2. **Flow** — Prediction → Explanation → Diagnosis → Prescription
3. **Architecture** — Model + 4-layer XAI stack diagram
4. **Novel Contributions** — FastSHAP (real-time), Archipelago (interactions), CEM (contrastive), Prototype (relatable)
5. **Demo** — High-risk learner → full explanation → What-If → counterfactual actions
6. **Trust Score** — Fidelity + Stability + Completeness = auditable
7. **Causal Layer** — Distinguishing correlation from causation (brownie)
8. **SDK** — pip install xai-learner-sdk
