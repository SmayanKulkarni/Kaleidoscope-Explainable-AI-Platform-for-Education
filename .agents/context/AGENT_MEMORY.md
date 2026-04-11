# Agent Memory & Active State Tracking

> **Purpose:** This file acts as the active context boundary (memory) for cross-IDE agents. It tracks the current focus, major architectural decisions, and open problems. Agents must READ this file upon initialization and UPDATE it when changing major contexts.

## Current Sprint Goal
- **Sprint:** AWS Deployment + ML Feedback Loop — **COMPLETE**
- All deployment infrastructure (Docker, PostgreSQL, S3, Terraform, CI/CD, Alembic) is implemented and tested.
- Implicit feedback retraining pipeline (engagement autoencoder + GBM retrain) is implemented.
- Next action: provision AWS infrastructure and push first deploy.

---

## Active Context

### Status
- **Phase 1-3 Backend:** COMPLETE — GBM, RF, LSTM, full XAI stack, MLOps.
- **Phase 4 Implicit Feedback:** COMPLETE — engagement autoencoder, retrain pipeline, hot-reload.
- **Phase 5 Deployment:** COMPLETE — Docker, PostgreSQL migration, S3, Terraform, GitHub Actions, Alembic, test suite.
- **Stress Testing:** COMPLETE in `astro` env for `tests/test_stress.py` after event alias compatibility fix (25/25 pass).
- **MLOps Hardening (Phase 0):** IN PROGRESS — `/mlops/retrain` and `/mlops/reload` now protected by admin JWT or automation token; retrain/reload serialization lock added.

### Architecture Summary
- **Models:** GBM (primary), RF (comparison), LSTM (temporal), EngagementAutoencoder (implicit → latent).
- **Features:** 12 OULAD static + 3 latent engagement (engagement_latent_1/2/3) = 15 total.
- **Explainers:** TreeSHAP, DeepSHAP, Captum IG, Archipelago, Anchors, DiCE, Prototypes, MAPIE, CausalAnnotator.
- **Stores:** ExplanationStore, FeedbackStore, EventStore, Auth — all migrated to PostgreSQL via db_config.py.
- **Deployment:** Docker multi-stage → ECR → EC2 t3.micro. PostgreSQL on RDS db.t3.micro. Models on S3 (versioned).

### Data
- data/learners.csv (32,593 rows), data/train.pkl (26,074), data/test.pkl (6,519).
- data/temporal/snapshots.pkl (195,558 rows), data/temporal/transitions.pkl (162,965 rows).

---

## Module Status

### Models
| Model | File | Status |
|-------|------|--------|
| GBM (primary) | ackend/app/model/trainer.py | ✅ Complete |
| Random Forest | ackend/app/model/trainer.py | ✅ Complete |
| LSTM (temporal) | ackend/app/model/lstm_trainer.py | ✅ Complete |
| Engagement Autoencoder | ackend/app/model/engagement_model.py | ✅ Complete |
| Implicit Aggregator | ackend/app/model/implicit_aggregator.py | ✅ Complete |
| Retrain Pipeline | ackend/app/model/retrain_pipeline.py | ✅ Complete |
| Hot-Reload | ackend/app/model/hot_reload.py | ✅ Complete |
| S3 Loader | ackend/app/model/s3_loader.py | ✅ Complete |

### XAI Stack
| Module | File | Status |
|--------|------|--------|
| TreeSHAP + What-If | ackend/app/explainers/shap_explainer.py | ✅ Complete |
| DeepSHAP + Captum IG | ackend/app/explainers/shap_explainer.py | ✅ Complete |
| Archipelago (interactions) | ackend/app/explainers/archipelago.py | ✅ Complete |
| Anchors (IF-THEN) | ackend/app/explainers/anchors_explainer.py | ✅ Complete |
| DiCE (counterfactuals) | ackend/app/explainers/dice_explainer.py | ✅ Complete |
| Prototypes (k-NN) | ackend/app/explainers/prototype_explainer.py | ✅ Complete |
| MAPIE (uncertainty) | ackend/app/evaluator/uncertainty_estimator.py | ✅ Complete |
| Causal Annotation (DAG) | ackend/app/causal/causal_annotator.py | ✅ Complete + latent features |
| TrustScorer | ackend/app/evaluator/trust_scorer.py | ✅ Complete |
| Drift Detector (JSD) | ackend/app/tracker/drift_detector.py | ✅ Complete |
| Evidently Drift Monitor | ackend/app/mlops/drift_monitor.py | ✅ Complete |
| LLM Narration (Groq) | ackend/app/narrator/llm_narrator.py | ✅ Complete |

### Stores (PostgreSQL-migrated)
| Store | File | Status |
|-------|------|--------|
| Auth DB | ackend/app/auth/database.py | ✅ PostgreSQL via db_config |
| Explanation Store | ackend/app/tracker/consistency_store.py | ✅ PostgreSQL via db_config |
| Feedback Store | ackend/app/tracker/feedback_store.py | ✅ PostgreSQL via db_config |
| Event Store | ackend/app/tracker/event_store.py | ✅ PostgreSQL via db_config |
| Central Config | ackend/app/db_config.py | ✅ Complete |

### Deployment
| Component | File/Location | Status |
|-----------|---------------|--------|
| Dockerfile (multi-stage) | Dockerfile | ✅ Complete |
| Docker Compose (local dev) | docker-compose.yml | ✅ Complete |
| .dockerignore | .dockerignore | ✅ Complete |
| Terraform VPC + SGs | infra/vpc.tf | ✅ Complete |
| Terraform ECR | infra/ecr.tf | ✅ Complete |
| Terraform S3 | infra/s3.tf | ✅ Complete |
| Terraform EC2 + IAM | infra/ec2.tf | ✅ Complete |
| Terraform RDS | infra/rds.tf | ✅ Complete |
| Terraform Outputs | infra/outputs.tf | ✅ Complete |
| EC2 User Data | infra/userdata.sh.tpl | ✅ Complete |
| CI Workflow | .github/workflows/ci.yml | ✅ Complete |
| CD Workflow | .github/workflows/deploy.yml | ✅ Complete + alembic migrate |
| Retrain Workflow | .github/workflows/retrain.yml | ✅ Complete |
| Terraform Workflow | .github/workflows/terraform.yml | ✅ Complete |
| Alembic Config | lembic.ini + lembic/env.py | ✅ Complete |
| Initial Migration | lembic/versions/0d652f39389c_initial_schema.py | ✅ Generated + applied |

### Auth
| Component | File | Status |
|-----------|------|--------|
| JWT Auth | ackend/app/auth/auth.py | ✅ Complete |
| Auth Router | ackend/app/auth/router.py | ✅ Complete + /enroll |
| ORM Models | ackend/app/auth/models.py | ✅ Complete |

### Tests (30/30 passing locally)
| File | Tests | Status |
|------|-------|--------|
| 	ests/test_db_config.py | 3 | ✅ Pass |
| 	ests/test_event_store.py | 6 | ✅ Pass |
| 	ests/test_feedback_store.py | 6 | ✅ Pass |
| 	ests/test_implicit_aggregator.py | 7 | ✅ Pass |
| 	ests/test_engagement_model.py | 8 | ✅ Pass |
| 	ests/test_api_smoke.py | 11 | ⏳ Requires full requirements.txt install |

---

## Architectural Constraints (Strict — unchanged)
- **No LLMs for prediction or explanation.** LLM (Groq) is strictly for narration of pre-computed XAI payloads.
- **Immutable features locked in DiCE:** prior_course_completions, current_week_in_course, quiz_avg_score, engagement_latent_1/2/3.
- **Latent features are correlational, not causal** — CausalAnnotator short-circuits with is_causal=False.
- **Feature order canonical:** Always use FEATURE_COLUMNS from data_loader.py.
- **TrustScorer mandatory:** Every /explain call must update trust metrics.
- **MLflow required:** Every training run logged. Production model always from registry.
- **Validation gates on retrain:** AUC drop ≤ 0.02, Brier rise ≤ 0.03, SHAP fidelity ≤ 0.05.

---

## Open Problems / Blockers

### Infrastructure (Needs Human Action)
1. **AWS provisioning not yet done** — run 	erraform apply in infra/ to create EC2, RDS, S3, ECR.
2. **GitHub Secrets not yet configured** — add 11 secrets per README.md table before pushing to main.
3. **Initial model upload to S3** — after 	erraform apply, run make models-upload.
4. **First Docker image push** — first push to main after secrets are set will trigger CD workflow.
5. **Terraform state bucket** — create xai-rec-tf-state S3 bucket + xai-rec-tf-locks DynamoDB table before 	erraform init.

### Known Technical Notes
- PyTorch CPU-only build recommended for t3.micro (GPU not available on free tier).
- LSTM Docker image ~1.2GB due to PyTorch — ECR lifecycle keeps only 5 images to manage storage costs.
- orum_posts_count, ideo_completion_rate, vg_session_duration_min, help_requests_count are VLE-approximated / synthetically augmented — documented in MODEL_CARD.md.
- Rust MC simulator still pending (Python fallback in 	emporal_builder.py is active).
- 	est_api_smoke.py requires full pip install -r requirements.txt — passes in CI, not in partial local venv.

---
*(Agents: Update Active Context and Open Problems as you progress. Entries are append-only in AGENT_LEDGER.md.)*
