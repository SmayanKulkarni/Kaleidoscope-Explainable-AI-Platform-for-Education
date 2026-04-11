# LLM WIKI for XAI Learning Recommendation System
> **Instructions for AI Agents**: Read this document first for project context. The full docs are in `docs/` or `files/`.

## 1. Project Overview & Rules
**What it is:** An Explainable AI (XAI) system wrapped around a dropout-risk prediction model. Focus is on *interactive and prescriptive explanations*.
**Hard Rules:**
- 🚫 NEVER invent or fake explanations via LLM directly. 
- ✅ ALWAYS derive explanations from `model.predict_proba()` and pre-computed XAI logic.
<<<<<<< HEAD
- 💬 LLM (via Anthropic SDK) is *only for narration* of data payloads.
- 🎯 Action recommendations MUST come from counterfactual analysis (DiCE), never hard-coded.

## 2. Architecture & Data Flow
**Layered Stack:**
1. **Prediction** (GBM model + MAPIE uncertainty)
2. **Explanation** (TreeSHAP for features, Archipelago for interactions, Anchors for rules, Prototypes for similarity)
3. **Diagnosis** (CEM Pertinent Positives + Negatives, DoWhy causal layer)
4. **Prescription** (DiCE + ActionRanker)
5. **Evaluation/Tracking** (TrustScorer [Fidelity/Stability/Completeness] + ExplanationStore + DriftDetector [JSD])
6. **Narration** (Claude Sonnet: dual audience Learner vs Instructor)

**Flow:** Request (`/predict` or `/explain`) → Parallel execution of XAI algorithms → Synchronous CEM & Trust Score → Save to SQLite → Generate LLM Narratives → Response back to React Frontend.
=======
- 💬 LLM (via Groq SDK) is *only for narration* of data payloads.
- 🎯 Action recommendations MUST come from counterfactual analysis (DiCE), never hard-coded.

## 2. Architecture & Data Flow
**Dual-Model + 7-Layer Stack:**
1. **Prediction** — GBM (static, 12 features) + LSTM (temporal, 6×12 sequence) + MAPIE uncertainty
2. **Explanation** — TreeSHAP/DeepSHAP (feature attribution + What-If), Archipelago (interactions), Anchors (IF-THEN rules), Prototypes/k-NN (similar learners), Captum Integrated Gradients (DL validation)
3. **Prescription** — DiCE counterfactuals + ActionRanker (causal-weighted)
4. **Evaluation** — TrustScorer (Fidelity/Stability/Completeness), DriftDetector (JSD), ExplanationStore (SQLite)
5. **Causal** — Hardcoded DAG annotation (causal vs correlational labels per feature)
6. **Narration** — Groq-hosted LLM: dual audience (Learner motivational / Instructor technical)
7. **MLOps** — MLflow (experiment tracking + model registry), Evidently (data/prediction drift), Model Card generator

**Forward Projection:** Rust Monte Carlo simulator (PyO3, Rayon-parallelized) samples N=10K trajectories from empirical transitions.

**Removed (redundant):** FastSHAP (TreeSHAP <30ms), CEM (DiCE+SHAP cover same ground).

**Flow:** Request → GBM TreeSHAP + LSTM DeepSHAP + Archipelago + Anchors + DiCE + Prototypes (parallel) → TrustScore → Log to MLflow → Save to SQLite → LLM Narration → Response.
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60

## 3. Key Data Structures
### Features (Mapped to `LEARNER_FEATURES`)
- **Actionable (Mutable):** `login_frequency_weekly`, `avg_session_duration_min`, `forum_posts_count`, `video_completion_rate`, `quiz_completion_rate`, `assignment_submission_rate`, `days_since_last_activity`, `help_requests_count`
- **Immutable:** `prior_course_completions`, `current_week_in_course`
- **Causal DAG Features:** `assignment_submission_rate`, `quiz_completion_rate`, `days_since_last_activity`, `missed_deadlines_count`, `login_frequency_weekly`

## 4. Algorithms Reference
<<<<<<< HEAD
| Need | Algorithm | Notes / Use-case |
| --- | --- | --- |
| Feature Importance | `TreeSHAP` | Base reference. Accuracy > Speed. |
| Live What-If | `FastSHAP` | Neural Net Surrogate predicting SHAP in <5ms. |
| Rule-based | `Anchors` | IF-THEN rules with ≥90% precision for Instructors. |
| Counterfactuals | `DiCE` | Generates "target states" for actionable features. |
| Diagnosis | `CEM` | Pertinent Positives (why this prediction) & Negatives (what would flip it). |
| Similar Learners | `ProtoDash` / `k-NN` | "Your profile matches 2 other students who..." |
| Interactions | `Archipelago` | SHAP interaction values highlighting synergistic risk. |

## 5. API Reference (`http://localhost:8000`)
- `POST /predict`: Standard inference. Returns `risk_score` + `uncertainty`.
- `POST /explain`: Full suite. Returns `{feature_level, concept_level, prediction_level, diagnosis, trust, narratives}`
- `POST /whatif`: Live slider updates. Requires fast `predict_proba` and `FastSHAP`.
- `POST /counterfactual`: DiCE execution to find actionable pivots. Returns `PrescriptiveAction` lists.
- `GET /history/{id}`: Timeline of previous explanations + JSD Drift flags.
=======
| Need | Algorithm | Unique Signal for LLM Narrator |
| --- | --- | --- |
| Feature Attribution (ML) | `TreeSHAP` | Per-feature signed importance with mathematical guarantees |
| Feature Attribution (DL) | `DeepSHAP` | SHAP values via DeepLIFT backprop on LSTM |
| Temporal Attribution | `Integrated Gradients` (Captum) | Per-timestep × per-feature importance on LSTM sequences |
| Feature Interactions | `Archipelago` | Pairwise synergistic risk (free via `shap_interaction_values()`) |
| Rule-based | `Anchors` | IF-THEN rules with ≥90% precision for instructor view |
| Counterfactuals | `DiCE` | Concrete actionable targets with locked immutable features |
| Similar Learners | `k-NN` (cosine) | Relatable past-student comparisons with outcomes |
| Uncertainty | `MAPIE` | Conformal prediction intervals — gates narrator confidence |
| Forward Projection | `MC Simulator` (Rust) | "If patterns continue, X% dropout by week Y" |
| Causal Annotation | Hardcoded DAG | Labels features causal vs correlational for responsible recommendations |
| Trust Quality | `TrustScorer` | Composite fidelity+stability+completeness — gates explanation display |
| Drift Monitoring | `JSD` + `Evidently` | Explanation drift (JSD) + data/prediction drift (Evidently) |

## 5. API Reference (`http://localhost:8000`)
- `POST /predict` — Returns `risk_score` + `uncertainty` (MAPIE interval) + `risk_label`. Query `?model=lstm` for temporal prediction.
- `POST /explain` — Full suite: `{shap_values, interactions, anchor_rule, prototypes, counterfactual_actions, causal_annotations, trust, narratives}`. Query `?model=lstm` adds `temporal_attributions`.
- `POST /whatif` — TreeSHAP re-run on modified features. Returns updated SHAP + risk delta.
- `POST /counterfactual` — DiCE execution. Returns ranked `PrescriptiveAction` list.
- `POST /simulate` — Rust MC forward projection. Returns `{feature_distributions, outcome_distribution}`.
- `GET /history/{id}` — Explanation timeline + drift flags.
- `GET /mlops/health` — Model version, drift status, last retrain date.
- `GET /mlops/drift-report` — Evidently data/prediction drift report.
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60

## 6. Coding Conventions
- All explainers inherit `BaseExplainer` and must return structured data.
- SHAP values: Negative = decreases risk, Positive = increases risk.
- Cache everything globally via `lifespan` in FastAPI. Do NOT reload models on request.
- Frontend uses React + Tailwind + Recharts. Backend uses FastAPI + SQLite.
<<<<<<< HEAD
=======
- Feature column order is canonical: always use `FEATURE_COLUMNS` from `data_loader.py`.
- MLflow: every training run logged. Production model served from registry.
- Rust MC simulator: import via `from mc_simulator import simulate_trajectories`.
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60
