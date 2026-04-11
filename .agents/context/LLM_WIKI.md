# LLM Wiki — XAI Learning Recommendation System

> **Purpose:** Authoritative reference for all agents working on this codebase. Read before every session. Contains architecture, algorithms, constraints, and non-negotiable rules.

---

## 1. Project Overview
An Explainable AI system for student dropout risk prediction on the OULAD dataset. Combines static GBM + temporal LSTM predictions with a full XAI explainer stack, a human-in-the-loop feedback loop (implicit + explicit), and a two-stage engagement retraining pipeline.

**Core goal:** Give instructors and students actionable, trustworthy, causally-grounded explanations for dropout risk — no black boxes.

---

## 2. Architecture Overview

### Model Pipeline (Two-Stage)
`
Stage 1 — Engagement Encoding
  ImplicitAggregator → 20 signals (15 implicit + 5 explicit)
  → EngagementAutoencoder → 3 latent features (engagement_latent_1/2/3)

Stage 2 — Risk Prediction
  12 OULAD features + 3 latent = 15 features → GBM / RF / LSTM
`

### XAI Flow
`
Request → GBM TreeSHAP + LSTM DeepSHAP + Archipelago + Anchors
        + DiCE + Prototypes (parallel via asyncio)
        → TrustScorer → Log (MLflow + SQLite/Postgres)
        → LLM Narration (Groq) → Response
`

### Deployment Stack
`
GitHub Actions CI/CD
  ├── ci.yml      — ruff lint + pytest (Postgres service container)
  ├── deploy.yml  — Docker → ECR → alembic migrate → EC2 systemd restart
  ├── retrain.yml — /mlops/retrain → /mlops/reload (weekly + manual)
  └── terraform.yml — infra plan (PR) / apply (main merge)

AWS (Free Tier)
  ├── EC2 t3.micro    — FastAPI container (xai-api systemd service)
  ├── RDS db.t3.micro — PostgreSQL 16 (all 4 stores)
  ├── S3              — Versioned model artifacts
  └── ECR             — Docker images (lifecycle: keep 5)

Local Dev
  └── docker-compose up  — postgres:16-alpine + api:runtime
`

---

## 3. Key Data Structures

### Feature Sets
| Category | Features | Count |
|----------|----------|-------|
| OULAD static | login_frequency_weekly, vg_session_duration_min, orum_posts_count, ideo_completion_rate, quiz_avg_score, quiz_completion_rate, ssignment_submission_rate, days_since_last_activity, prior_course_completions, current_week_in_course, missed_deadlines_count, help_requests_count | 12 |
| Engagement latent | engagement_latent_1, engagement_latent_2, engagement_latent_3 | 3 |
| **Total (model input)** | | **15** |

### Feature Rules
- **Actionable (mutable):** login_frequency_weekly, vg_session_duration_min, orum_posts_count, ideo_completion_rate, quiz_completion_rate, ssignment_submission_rate, days_since_last_activity, help_requests_count.
- **Immutable (locked in DiCE):** prior_course_completions, current_week_in_course, quiz_avg_score, engagement_latent_1, engagement_latent_2, engagement_latent_3.
- **Causal DAG:** ssignment_submission_rate, quiz_completion_rate, days_since_last_activity, missed_deadlines_count, login_frequency_weekly.
- **Latent features:** Always is_causal=False in CausalAnnotator. Never appear in DiCE suggestions. ActionRanker skips them.

### Engagement Signals (20 total → Autoencoder)
| Type | Signals (15 implicit) |
|------|----------------------|
| Navigation | session_count, pages_visited, vg_time_on_page_sec |
| What-If | whatif_interaction_count, unique_features_explored, vg_whatif_delta |
| Engagement | ction_views, ction_dismissals, explanation_revisits |
| Download | esource_downloads |
| Scroll | vg_scroll_depth_pct |
| Focus | 	otal_focus_time_sec, ocus_sessions |
| Session | 	ime_to_first_action_sec, ounce_sessions |

| Type | Signals (5 explicit) |
|------|---------------------|
| Rating | vg_explanation_rating, ating_count |
| Follow | ecommendation_follow_rate |
| Correction | correction_count |
| Recency | days_since_last_feedback |

---

## 4. Algorithms Reference

| Need | Algorithm | File | Notes |
|------|-----------|------|-------|
| Feature Attribution (GBM) | TreeSHAP | explainers/shap_explainer.py | Base reference |
| Feature Attribution (LSTM) | DeepSHAP | explainers/shap_explainer.py | DeepLIFT backprop |
| Temporal Attribution | Captum Integrated Gradients | explainers/shap_explainer.py | Per-timestep × per-feature |
| Feature Interactions | Archipelago | explainers/archipelago.py | SHAP interaction values |
| Rule-based | Anchors (alibi) | explainers/anchors_explainer.py | IF-THEN ≥90% precision |
| Counterfactuals | DiCE | explainers/dice_explainer.py | Immutable features locked |
| Similar Learners | k-NN cosine | explainers/prototype_explainer.py | Relatable past-student comparison |
| Uncertainty | MAPIE (SplitConformal) | evaluator/uncertainty_estimator.py | Conformal prediction intervals |
| Causal Annotation | DoWhy + DAG | causal/causal_annotator.py | Backdoor linear regression; latent features → correlational |
| Trust Quality | TrustScorer | evaluator/trust_scorer.py | 0.40×fidelity + 0.35×stability + 0.25×completeness |
| Explanation Drift | JSD | 	racker/drift_detector.py | Jensen-Shannon on |SHAP| softmax |
| Data/Prediction Drift | Evidently | mlops/drift_monitor.py | DataDriftPreset; 1h TTL cache |
| Engagement Compression | Denoising Autoencoder | model/engagement_model.py | 20→3 latent; cold-start = zeros |
| Retrain Orchestration | RetrainPipeline | model/retrain_pipeline.py | Validation gates: AUC≤0.02, Brier≤0.03, SHAP fidelity≤0.05 |
| Narration | Groq LLM | 
arrator/llm_narrator.py | Dual audience (learner motivational / instructor technical) |

---

## 5. API Reference (http://localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict | POST | GBM risk score + MAPIE uncertainty + risk label. ?model=lstm for temporal. |
| /explain | POST | Full XAI suite: SHAP, interactions, anchors, DiCE, prototypes, causal, trust, narratives. |
| /whatif | POST | Re-runs TreeSHAP on modified features. Returns SHAP delta + risk delta. |
| /simulate | POST | Monte Carlo forward projection. Returns outcome distribution. |
| /events | POST | Batch implicit event ingestion (EventBatchRequest, min 1, max 500). |
| /events/{learner_id} | GET | Retrieve all events for a learner. |
| /feedback | POST | Explicit feedback: rating (1-5), follow flag, feature correction. |
| /feedback/stats | GET | Aggregate stats: avg rating, follow rate, top corrected features. |
| /history/{learner_id} | GET | Explanation timeline + JSD drift flags. |
| /explain/me | POST | Authenticated student endpoint — returns own explanation summary. |
| /mlops/health | GET | Model version, drift status, last retrain timestamp. |
| /mlops/metrics | GET | Training metrics (AUC, Brier, SHAP fidelity). |
| /mlops/drift-report | GET | Evidently data/prediction drift report. |
| /mlops/retrain | POST | Trigger full retraining pipeline. Returns RetrainResult JSON. |
| /mlops/reload | POST | Hot-reload new model into live process without restart. |
| /health | GET | Simple health check (used by Docker + load balancer). |
| /auth/register | POST | Create user (student/instructor). |
| /auth/login | POST | JWT token. |
| /auth/me | GET | Current user info. |
| /auth/enroll | POST | Instructor enroll student into course. |

---

## 6. Database Schema

All stores share a single PostgreSQL instance in production (via DATABASE_URL), or use per-store SQLite files in local dev.

| Store | Tables | Config key |
|-------|--------|-----------|
| Auth | users, learner_profiles, instructor_profiles, course_enrollments | make_engine("auth") |
| Explanations | explanation_records | make_engine("explanations") |
| Feedback | eedback_records | make_engine("feedback") |
| Events | interaction_events | make_engine("events") |

Migrations managed by Alembic. Run lembic upgrade head to apply all migrations.

---

## 7. Deployment Reference

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| DATABASE_URL | Prod only | postgresql://user:pass@host:5432/xai_db |
| JWT_SECRET_KEY | Always | 32+ char random hex |
| GROQ_API_KEY | Optional | Disables narration if blank |
| AWS_S3_BUCKET | Prod only | S3 bucket for model artifacts |
| AWS_REGION | Prod only | e.g. p-south-1 |

### Key File Locations
| Artifact | Path |
|----------|------|
| GBM model | models/gbm.pkl |
| RF model | models/rf.pkl |
| LSTM model | models/lstm.pt |
| Engagement autoencoder | models/engagement/ |
| Training data | data/train.pkl, data/test.pkl |
| Temporal data | data/temporal/snapshots.pkl, data/temporal/transitions.pkl |
| Fixtures | data/fixtures/high_risk.json, medium_risk.json, low_risk.json |

---

## 8. Coding Conventions (Non-negotiable)

1. **No LLMs for prediction or explanation.** Groq is strictly for narration of pre-computed XAI payloads.
2. **Feature order is canonical.** Always use FEATURE_COLUMNS from data_loader.py. Never hardcode column order.
3. **All explainers must be stateless on request.** Load once in lifespan(), never reload per-request.
4. **TrustScorer is mandatory** after every /explain call.
5. **Immutable features must always be locked in DiCE.** Never remove from IMMUTABLE_FEATURES.
6. **Latent features are correlational.** CausalAnnotator MUST return is_causal=False for all engagement_latent_* features.
7. **MLflow required.** Every training run must be logged with metrics and artifact paths.
8. **Retrain validation gates are hard stops.** Pipeline MUST abort if AUC drop > 0.02, Brier rise > 0.03, or SHAP fidelity drops > 0.05.
9. **Cold-start handling.** Learners with no implicit events → engagement_latent_* = 0.0. Never impute with mean.
10. **Never commit .env or 	erraform.tfvars.** Both are gitignored.

---

## 9. Open Tasks / Pending
| Task | Priority | Notes |
|------|----------|-------|
| AWS 	erraform apply | High | Requires infra/terraform.tfvars filled, state bucket created |
| GitHub Secrets setup | High | 11 secrets per README table before pushing to main |
| Initial models/ → S3 upload | High | make models-upload after terraform apply |
| First CD pipeline run | High | Push to main after secrets configured |
| Rust MC Simulator | Low | Python fallback active in 	emporal_builder.py |
