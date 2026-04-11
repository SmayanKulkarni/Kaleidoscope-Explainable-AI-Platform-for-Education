# GitHub Copilot Instructions — XAI Learning Recommendation System

## What This Project Is

An **Explainable AI (XAI) system** wrapped around a learner recommendation model. The model
predicts outcomes like "Study Topic X next" or "This learner is at risk of dropping out."
The **entire focus is the explainability layer** — every decision must be explained faithfully,
interactively, and prescriptively.

Core flow: **Prediction → Explanation → Diagnosis → Prescription**

---

## Hard Rules (Never Violate These)

1. **Explanations must be derived from model internals** — never from an LLM directly
2. **An LLM may only narrate a pre-computed explanation** (SHAP values, DiCE output, etc.)
3. **What-If changes must route through `model.predict_proba()`** — no heuristic approximations
4. **Action recommendations must be derived from counterfactual analysis** — never hard-coded
5. **Every explanation concludes with ≥1 prioritized action + estimated impact**

---

## Project Structure

```
xai-learning-system/
├── .github/
│   └── copilot-instructions.md        ← YOU ARE HERE
├── docs/
│   ├── ARCHITECTURE.md                ← System design & data flow
│   ├── XAI_ALGORITHMS.md              ← All algorithms, novel + conventional
│   ├── IMPLEMENTATION_PLAN.md         ← Sprint plan with task breakdown
│   └── API_SPEC.md                    ← All endpoint contracts
├── backend/
│   ├── app/
│   │   ├── main.py                    ← FastAPI entry point
│   │   ├── model/
│   │   │   ├── trainer.py             ← Model training (sklearn)
│   │   │   ├── predictor.py           ← predict_proba wrapper
│   │   │   └── mock_data.py           ← Synthetic learner data generator
│   │   ├── explainers/
│   │   │   ├── shap_explainer.py      ← SHAP TreeSHAP + KernelSHAP
│   │   │   ├── lime_explainer.py      ← LIME concept-level
│   │   │   ├── dice_explainer.py      ← DiCE counterfactuals
│   │   │   ├── anchors_explainer.py   ← Anchors rule-based
│   │   │   ├── cem_explainer.py       ← Contrastive Explanations (CEM)
│   │   │   ├── fastshap_explainer.py  ← FastSHAP surrogate (novel)
│   │   │   ├── prototype_explainer.py ← ProtoDash similarity
│   │   │   └── archipelago.py         ← Feature interaction detection
│   │   ├── causal/
│   │   │   ├── dag_builder.py         ← DoWhy causal DAG
│   │   │   └── causal_annotator.py    ← Tag features causal vs correlational
│   │   ├── prescriptor/
│   │   │   └── action_ranker.py       ← DiCE + SHAP → ranked actions
│   │   ├── tracker/
│   │   │   ├── consistency_store.py   ← SQLite / Redis attribution history
│   │   │   └── drift_detector.py      ← JSD + PSI drift alerts
│   │   ├── evaluator/
│   │   │   └── trust_scorer.py        ← Fidelity + Stability + Completeness
│   │   └── narrator/
│   │       └── llm_narrator.py        ← LLM narrates computed explanations ONLY
│   ├── sdk/
│   │   └── xai_sdk/                   ← Publishable SDK package
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ExplanationCard.jsx
│   │   │   ├── WhatIfPanel.jsx
│   │   │   ├── CounterfactualView.jsx
│   │   │   ├── AudienceToggle.jsx     ← Learner vs Instructor switch
│   │   │   ├── ConsistencyTimeline.jsx
│   │   │   └── TrustScoreBadge.jsx
│   │   └── views/
│   │       ├── LearnerDashboard.jsx
│   │       └── InstructorDashboard.jsx
├── docker-compose.yml
├── Dockerfile.backend
└── Dockerfile.frontend
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | `scikit-learn` — GradientBoostingClassifier or RandomForestClassifier |
| SHAP | `shap` — TreeSHAP for tree models, KernelSHAP fallback |
| Counterfactuals | `dice-ml` — DiCE with actionability constraints |
| Anchors | `alibi` — rule-based high-precision explanations |
| Causal | `dowhy` + `pgmpy` — SCM + do-calculus |
| Drift Detection | `alibi-detect` — JSD on SHAP distributions |
| Uncertainty | `mapie` or MC Dropout wrapper — epistemic uncertainty |
| API | `FastAPI` + `uvicorn` |
| Frontend | `React` + `Recharts` + `Tailwind` |
| Storage | `SQLite` (dev) / `PostgreSQL` (prod) via `SQLAlchemy` |
| Narration | `anthropic` SDK — narrate ONLY pre-computed explanations |
| Container | `Docker` + `docker-compose` |
| SDK | `setuptools` / `pyproject.toml` — installable package |

---

## Key Data Contracts

### Learner Feature Vector
```python
LEARNER_FEATURES = {
    # Engagement
    "login_frequency_weekly": float,       # avg logins per week
    "avg_session_duration_min": float,     # avg session in minutes
    "forum_posts_count": int,              # total forum participation
    "video_completion_rate": float,        # 0.0 – 1.0
    # Assessment
    "quiz_avg_score": float,               # 0.0 – 100.0
    "quiz_completion_rate": float,         # 0.0 – 1.0
    "assignment_submission_rate": float,   # 0.0 – 1.0
    "days_since_last_activity": int,
    # History
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

IMMUTABLE_FEATURES = [
    "prior_course_completions", "current_week_in_course"
]
```

### Prediction Output
```python
@dataclass
class PredictionResult:
    learner_id: str
    risk_label: str          # "LOW" | "MEDIUM" | "HIGH"
    risk_score: float        # 0.0 – 1.0 from predict_proba
    recommended_topic: str   # next study topic
    timestamp: datetime
    model_confidence: float  # calibrated confidence
    uncertainty: float       # epistemic uncertainty (MC Dropout or MAPIE)
```

### Explanation Output
```python
@dataclass
class ExplanationResult:
    prediction: PredictionResult
    # Feature-level
    shap_values: dict[str, float]         # feature → SHAP value
    shap_base_value: float
    # Concept-level
    concept_scores: dict[str, float]      # "engagement" → score
    # Anchors rule
    anchor_rule: str                       # "IF quiz_rate < 0.4 AND ..."
    anchor_precision: float
    # Contrastive (CEM)
    pertinent_positives: dict[str, float] # features driving prediction
    pertinent_negatives: dict[str, float] # features whose absence flips it
    # Stability
    stability_score: float                 # 0.0 – 1.0
    trust_score: float                     # composite fidelity+stability+completeness
    # Causal annotation
    causal_features: list[str]
    correlational_features: list[str]
    # Audience-specific text (LLM-narrated)
    learner_narrative: str
    instructor_narrative: str
```

### Counterfactual Output
```python
@dataclass
class CounterfactualResult:
    original_features: dict
    counterfactual_features: dict
    changed_features: dict[str, tuple]    # feature → (original, new)
    flip_probability: float               # new predict_proba score
    estimated_impact: float               # delta in risk score
    human_readable: str                   # "Study 2 more hours/week..."
    actions: list[PrescriptiveAction]

@dataclass
class PrescriptiveAction:
    feature: str
    current_value: float
    target_value: float
    plain_language: str                   # "Complete missed quizzes"
    estimated_impact: float               # delta risk reduction
    priority_rank: int                    # 1 = highest impact
    is_causal: bool                       # from DoWhy annotation
```

---

## Coding Conventions

- All explainer classes implement `BaseExplainer` with `.explain(features, model)` → `ExplanationResult`
- All endpoints return `{status, data, meta}` envelope
- SHAP values always returned as signed floats (negative = reduces risk, positive = increases risk)
- Use `@lru_cache` on model loading, never reload on each request
- Store explanation history in `ExplanationStore` (SQLite) with `learner_id + timestamp` index
- Feature names must match `LEARNER_FEATURES` keys exactly — validate on input
- DiCE must always receive `actionability_constraints` dict from `ACTIONABLE_FEATURES`
- All explainers must be testable independently via `pytest` with a mock model fixture

---

## What Copilot Should Help With

When writing code in this project, Copilot should:

1. **Always derive explanations from model internals** — call `model.predict_proba()`, extract
   SHAP values, pass to DiCE — never fabricate explanation text
2. **Implement `BaseExplainer` interface** for every new explainer module
3. **Respect actionability constraints** when calling DiCE — import from `config.py`
4. **Store every explanation** in `ExplanationStore.save()` after generation
5. **Compute trust score** after every explanation via `TrustScorer.score()`
6. **Use async FastAPI handlers** with `asyncio.gather()` to run SHAP + DiCE in parallel
7. **Add type hints** to everything — this project uses strict typing
8. **Follow the 4-layer response**: prediction → explanation → diagnosis (CEM) → prescription (DiCE actions)
