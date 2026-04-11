# LLM WIKI for XAI Learning Recommendation System
> **Instructions for AI Agents**: Read this document first for project context. The full docs are in `docs/` or `files/`.

## 1. Project Overview & Rules
**What it is:** An Explainable AI (XAI) system wrapped around a dropout-risk prediction model. Focus is on *interactive and prescriptive explanations*.
**Hard Rules:**
- 🚫 NEVER invent or fake explanations via LLM directly. 
- ✅ ALWAYS derive explanations from `model.predict_proba()` and pre-computed XAI logic.
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

## 3. Key Data Structures
### Features (Mapped to `LEARNER_FEATURES`)
- **Actionable (Mutable):** `login_frequency_weekly`, `avg_session_duration_min`, `forum_posts_count`, `video_completion_rate`, `quiz_completion_rate`, `assignment_submission_rate`, `days_since_last_activity`, `help_requests_count`
- **Immutable:** `prior_course_completions`, `current_week_in_course`
- **Causal DAG Features:** `assignment_submission_rate`, `quiz_completion_rate`, `days_since_last_activity`, `missed_deadlines_count`, `login_frequency_weekly`

## 4. Algorithms Reference
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

## 6. Coding Conventions
- All explainers inherit `BaseExplainer` and must return structured data.
- SHAP values: Negative = decreases risk, Positive = increases risk.
- Cache everything globally via `lifespan` in FastAPI. Do NOT reload models on request.
- Frontend uses React + Tailwind + Recharts. Backend uses FastAPI + SQLite.
