# Implementation Plan — XAI Learning Recommendation System
# Hackathon Sprint

> **XAI Stack:**
> TreeSHAP · DeepSHAP · Archipelago · Anchors · DiCE · Prototypes · MAPIE · Causal DAG (DoWhy) · TrustScorer · DriftDetector · Integrated Gradients (Captum)
>
> **Models:** GBM (static, TreeSHAP) + LSTM (temporal, DeepSHAP + Captum)
> **Infra:** MLflow (tracking + registry + monitoring) · Python MC Simulator · Rust MC Simulator (PyO3, Phase 5)
>
> **Note:** Frontend is owned by a separate team. This plan covers backend only.

---

## Pre-Sprint Setup

### Environment
```bash
# Core ML + XAI
pip install \
  scikit-learn numpy pandas scipy \
  shap dice-ml alibi mapie \
  torch captum \
  mlflow evidently \
  fastapi uvicorn[standard] sqlalchemy \
  groq dowhy pgmpy \
  pytest httpx

# Rust MC simulator (Phase 5, optional)
pip install maturin
cd mc_simulator && maturin develop --release
```

### Data ✅ DONE
```bash
# Static features (one row per student)
python backend/app/model/data_loader.py
# Output: data/learners.csv (32,593 rows), data/train.pkl, data/test.pkl

# Temporal features (6 snapshots per student @ weeks 2,4,6,8,10,12)
python backend/app/model/temporal_builder.py
# Output: data/temporal/snapshots.pkl (195,558 rows)
#         data/temporal/transitions.pkl (162,965 MC transition deltas)
```

### Monte Carlo Simulator ✅ DONE
**File:** `backend/app/model/temporal_builder.py` — `MonteCarloSimulator` class

> Given a student's features at week T, samples N plausible future trajectories
> from empirical transition distributions and runs `predict_proba()` to produce
> outcome distributions. Exposed via `POST /simulate`.

---

## Phase 1 — Foundation ✅ DONE

**Goal:** Trained models + SHAP + API serving predictions

### Task 1.1a — GBM Trainer ✅ DONE
**File:** `backend/app/model/trainer.py`

- `GradientBoostingClassifier` (primary, TreeSHAP-compatible)
- `RandomForestClassifier` (secondary, multi-model comparison)
- Calibrated with `CalibratedClassifierCV`
- MLflow: `mlflow.sklearn.log_model()`, logs AUC/F1/Brier/calibration metrics
- Artifacts: `models/gbm.pkl`, `models/rf.pkl`, `models/training_summary.json`

### Task 1.1b — LSTM Trainer ✅ DONE
**File:** `backend/app/model/lstm_trainer.py`

- PyTorch `DropoutLSTM` — `Input(6, 12) → LSTM(64) → LSTM(32) → Dropout(0.3) → Dense(1)`
- Trained with `BCEWithLogitsLoss`, AdamW, early stopping on validation AUC (patience=5)
- MLflow: `mlflow.pytorch.log_model()`
- Artifacts: `models/lstm.pt`, `models/lstm_config.json`

### Task 1.2 — Dual SHAP Explainer ✅ DONE
**File:** `backend/app/explainers/shap_explainer.py`

- `TreeExplainer` for GBM (extracts base estimator from `CalibratedClassifierCV`)
- `DeepExplainer` for LSTM
- Methods: `explain()`, `explain_whatif()`, `explain_temporal()`, `stability_score()`, `global_summary()`

### Task 1.3 — FastAPI Base ✅ DONE
**File:** `backend/app/main.py`

- `GET  /health`
- `POST /predict` — `{risk_score, risk_label, uncertainty}`
- `POST /explain` — full XAI suite
- `POST /whatif` — TreeSHAP on modified features
- `POST /counterfactual` — DiCE ranked actions
- `GET  /history/{learner_id}` — timeline + drift flags

---

## Phase 2 — XAI Engine ✅ DONE

**Goal:** All 9 explanation + 2 tracking modules

### Task 2.1 — DiCE Counterfactual Engine ✅ DONE
**File:** `backend/app/explainers/dice_explainer.py`

- Immutable features locked: `prior_course_completions`, `current_week_in_course`, `engagement_latent_*`
- Permitted ranges learned from data (5th–95th percentile)
- Returns `CounterfactualResult` with `PrescriptiveAction` list

### Task 2.2 — Anchors Rule Explainer ✅ DONE
**File:** `backend/app/explainers/anchors_explainer.py`

- `alibi.explainers.AnchorTabular` with quartile discretizer
- Returns `{anchor_rule, precision, coverage, human_readable}`

### Task 2.3 — Prototype Explainer ✅ DONE
**File:** `backend/app/explainers/prototype_explainer.py`

- cosine k-NN (`sklearn.neighbors.NearestNeighbors`) on `StandardScaler` space
- Top-2 distinguishing features per match, motivational narrative

### Task 2.4 — Archipelago Interaction Detector ✅ DONE
**File:** `backend/app/explainers/archipelago.py`

- `shap.TreeExplainer.shap_interaction_values()` — top-k off-diagonal pairs
- Direction: amplifying / dampening; `to_narrative()` for LLM input

### Task 2.5 — Uncertainty Estimator ✅ DONE
**File:** `backend/app/evaluator/uncertainty_estimator.py`

- MAPIE v1 `SplitConformalClassifier` — `conformalize()` API
- `predict_with_uncertainty(X, alpha=0.10)` → `{prediction_set, confidence_width, uncertainty_label}`
- Integrated into `/predict` and `/explain`

### Task 2.6 — Causal Annotator ✅ DONE
**File:** `backend/app/causal/causal_annotator.py`

- DoWhy `CausalModel` per feature — dynamic DAG from domain knowledge (no hardcoding)
- `backdoor.linear_regression` estimation + `random_common_cause` refutation
- `engagement_latent_*` features: treated as correlational, not causal (point-biserial fallback)
- `annotate_shap(shap_values)` → per-feature `{type, ate, is_causal, note}`

### Task 2.7 — Action Ranker ✅ DONE
**File:** `backend/app/prescriptor/action_ranker.py`

- IQR-derived actionability weights (data-driven)
- `priority_score = shap_magnitude × actionability × causal_weight`
- `engagement_latent_*` and immutables get zero actionability

### Task 2.8 — Trust Scorer ✅ DONE
**File:** `backend/app/evaluator/trust_scorer.py`

- `0.40×fidelity + 0.35×stability + 0.25×completeness`
- Returns `{trust_score, fidelity, stability, completeness, label}`

### Task 2.9 — Explanation Store + Drift Detector ✅ DONE
**Files:** `backend/app/tracker/consistency_store.py`, `backend/app/tracker/drift_detector.py`

- SQLAlchemy + SQLite/PostgreSQL (via `db_config.make_engine`)
- Drift: JSD on `softmax(|SHAP|)` + top-3 rank shift (threshold: JSD > 0.15)

---

## Phase 2A — MLOps Pipeline ✅ DONE

### Task M.1 — MLflow Experiment Tracking ✅ DONE
**File:** `backend/app/mlops/experiment_tracker.py`

### Task M.2 — Model Registry + Promotion ✅ DONE
**File:** `backend/app/mlops/model_registry.py`

- MLflow stages: `None → Staging → Production`
- `/predict` always loads Production model; rollback supported

### Task M.3 — Prediction Logger ✅ DONE
**File:** `backend/app/mlops/prediction_logger.py`

- Rolling SQLite/PostgreSQL table; auto-prune to max_rows=10,000
- `get_feature_matrix(n)` feeds Evidently drift monitor

### Task M.4 — Drift Monitor ✅ DONE
**File:** `backend/app/mlops/drift_monitor.py`

- Evidently `DataDriftPreset`; lazy import (avoids litestar dep chain)
- 1h TTL cache; exposed via `GET /mlops/drift-report`

### Task M.5 — Model Card Generator ✅ DONE
**File:** `backend/app/mlops/model_card.py`

- Auto-generates `MODEL_CARD.md` from `training_summary.json` + MLflow metadata

### Task M.6 — MLOps Endpoints ✅ DONE
**File:** `backend/app/main.py`

- `GET  /mlops/health` — model version, drift status, prediction count
- `GET  /mlops/drift-report` — Evidently JSON report
- `GET  /mlops/metrics` — production model metrics from training_summary.json
- `POST /mlops/retrain` — full implicit+explicit feedback retraining cycle
- `POST /mlops/reload` — hot-swap live model (atomic, fallback-safe)

---

## Phase 3 — LLM Narration ✅ DONE

### Task 3.1 — LLM Narrator ✅ DONE
**File:** `backend/app/narrator/llm_narrator.py`

- Groq SDK (set `GROQ_API_KEY` env var to enable; graceful no-op if absent)
- `narrate(explain_resp, learner_id, audience)` → `{learner, instructor}` narratives
- **CRITICAL:** LLM receives ONLY pre-computed XAI data — never generates explanations itself
- Payload to LLM: `{shap_top3, interactions, anchor_rule, actions, prototypes, risk_score, uncertainty, causal_annotations}`

```python
LEARNER_SYSTEM_PROMPT = """
You are a supportive learning coach. You receive pre-computed AI analysis results
and narrate them in encouraging, plain language for a student.
You must ONLY narrate what is in the data — never invent explanations.
Keep under 100 words. End with one specific actionable sentence.
"""

INSTRUCTOR_SYSTEM_PROMPT = """
You are an educational data analyst. Summarize pre-computed XAI results for an instructor.
Include: top risk factors, confidence level, recommended intervention.
Be precise and data-driven. Under 150 words.
"""
```

---

## Phase 4 — Backend Integration Support ✅ DONE

> Frontend is handled by a separate team. This phase covers backend contracts and handoff artifacts only.

### Task 4.1 — API Contract ✅ DONE
**File:** `files/API_SPEC.md`

All request/response schemas frozen and in sync with `main.py`.

### Task 4.2 — Frontend Handoff Fixtures ✅ DONE

Sample payloads available:
- `data/fixtures/high_risk_learner.json`
- `data/fixtures/medium_risk_learner.json`
- `data/fixtures/low_risk_learner.json`

### Task 4.3 — Auth + Implicit Feedback ✅ DONE
**Files:** `backend/app/auth/`, `backend/app/tracker/event_store.py`, `backend/app/tracker/feedback_store.py`

- JWT auth (`POST /auth/register`, `POST /auth/token`)
- `POST /events` — batch interaction event ingestion (implicit feedback)
- `POST /feedback` — explicit rating + follow flag + correction
- `GET  /feedback/stats` — aggregate follow rates, avg rating, top corrected features
- `GET  /explain/me` — authenticated student self-service endpoint

---

## Phase 5 — Advanced Features

### Task 5.1 — Rust Monte Carlo Simulator (PyO3) ✅ SCAFFOLDED
**Directory:** `mc_simulator/`

> Rust reimplementation of `MonteCarloSimulator` for 10–100× speedup.
> Exposed to Python via PyO3 + maturin. Uses Rayon for parallel trajectory sampling.

```
mc_simulator/
├── Cargo.toml           # deps: pyo3, rayon, rand, ndarray
├── pyproject.toml       # maturin build config
└── src/
    └── lib.rs           # SimConfig, simulate_trajectories(), Python bindings
```

Build:
```bash
cd mc_simulator && maturin develop --release
```

Python fallback (`MonteCarloSimulator` in `temporal_builder.py`) is active until Rust build completes.

### Task 5.2 — DL XAI: Integrated Gradients + Captum ✅ DONE
**File:** `backend/app/explainers/dl_explainer.py`

- `IntegratedGradients` and `LayerIntegratedGradients` via Captum
- `explain_temporal(sequence)` → per-timestep × per-feature attributions (6×12 matrix)
- `temporal_attention_summary(sequence)` → which weeks matter most
- Cross-validates against DeepSHAP for consistency check

### Task 5.3 — Human-in-the-Loop Feedback ✅ DONE
**File:** `backend/app/tracker/feedback_store.py`

- `POST /feedback` — rating, follow flag, feature correction
- `GET  /feedback/stats` — aggregate stats for concept drift detection

---

## Phase 6 — SDK + Docker ✅ DONE

### Task 6.1 — XAI SDK Package ✅ DONE
**Path:** `backend/sdk/`

```bash
pip install -e backend/sdk
```

```python
from xai_sdk import XAIClient

client = XAIClient(base_url="http://localhost:8000")
result = client.predict(features)
explanation = client.explain(features, learner_id="learner_042")
actions = client.counterfactual(features)
history = client.history("learner_042")
print(explanation.top_action)
```

### Task 6.2 — Docker ✅ DONE
**Files:** `Dockerfile`, `Dockerfile.backend` (symlink/alias), `docker-compose.yml`

```bash
docker-compose up
# Backend: http://localhost:8000
# MLflow:  http://localhost:5000
```

---

## Verification Checklist

```
✅ POST /predict returns risk_score + uncertainty (GBM and LSTM)
✅ POST /explain returns shap_values, interactions, anchor_rule, prototypes,
         counterfactual, causal_annotations, trust_score, uncertainty, explanation_drift, narratives
✅ POST /explain?model=lstm returns temporal DeepSHAP attributions
✅ POST /whatif uses TreeSHAP (verified < 50ms latency)
✅ POST /counterfactual returns ranked actions — no hard-coding
✅ POST /simulate returns MC forward projection with outcome distribution
✅ GET  /history/{id} returns timeline + drift flags (SQLite-backed)
✅ GET  /mlops/health returns model_version + drift_status + prediction_count
✅ GET  /mlops/drift-report returns Evidently JSON report
✅ GET  /mlops/metrics returns training AUC/F1/Brier from training_summary.json
✅ POST /mlops/retrain triggers full retraining pipeline
✅ POST /mlops/reload hot-swaps live model atomically
✅ POST /events ingests implicit interaction events
✅ POST /feedback accepts explicit learner/instructor feedback
✅ GET  /feedback/stats returns aggregate follow rates + corrected features
✅ POST /auth/register + POST /auth/token (JWT)
✅ GET  /explain/me authenticated student self-service
✅ MLflow UI accessible at :5000
✅ SDK installable: pip install -e backend/sdk
✅ Docker: docker-compose up works
✅ GET  /health returns 200 with GPU info
```
