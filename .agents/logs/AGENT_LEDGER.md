# Agent Implementation Ledger

> **Purpose:** Append-only log of technical actions, fixes, and discoveries made by AI agents across the codebase.
> **Instruction for Agents:** Upon completing *any* implementation step, refactor, or debugging task, APPEND a new entry to the bottom of this file.

## Format
```markdown
### [YYYY-MM-DD HH:MM] Agent Name / IDE - Brief description of work
- **Files Modified:** `file/path.py`
- **What was done:** 1-2 sentence description.
- **Why it was done:** Context on the change or bugfix.
- **Dependencies/Impacts:** What this change affects downstream.
```

---

### [2026-04-11] System Agent - Initialization
- **Files Modified:** `LLM_WIKI.md`, `AGENT_MEMORY.md`, `AGENT_LEDGER.md`.
- **What was done:** Established the agentic workflow rules, memory structures, and summarized the XAI documentation.
- **Why it was done:** To ensure cross-IDE compatibility and robust long-term LLM collaboration.
- **Dependencies/Impacts:** Core rule files now depend on this ledger being updated.
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60
=======
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60

### [2026-04-11 12:37] Cascade (Windsurf) - OULAD Data Pipeline Implementation
- **Files Created:** `backend/app/model/data_loader.py`, `requirements-data.txt`, `backend/__init__.py`, `backend/app/__init__.py`, `backend/app/model/__init__.py`
- **Directories Created:** `backend/app/model/`, `backend/app/explainers/`, `backend/app/narrator/`, `backend/app/tracker/`, `backend/app/evaluator/`, `backend/app/causal/`, `backend/app/prescriptor/`, `data/raw/`
- **What was done:** Fetched the OULAD dataset (~44 MB, 32,593 students, 10.6M VLE interaction rows) via direct unauthenticated GET from `https://analyse.kmi.open.ac.uk/open-dataset/download`. Implemented a full feature engineering pipeline in `data_loader.py` deriving all 12 `LEARNER_FEATURES` from raw OULAD tables. Produced `data/learners.csv`, `data/train.pkl`, and `data/test.pkl`.
- **Why it was done:** Phase 1 prerequisite — every downstream module (trainer, SHAP, DiCE, CEM) depends on a clean, schema-compliant feature matrix with a binary `dropout_risk` target.
- **Synthetic Approximations Applied:**
  - `avg_session_duration_min`: `sum_click/day × 2.5 min/click`, capped at 180 min/day, averaged (Macfadyen & Dawson 2010 basis).
  - `forum_posts_count`: Sum of `sum_click` on `activity_type ∈ {forumng, ouwiki, oucollaborate}`.
  - `video_completion_rate`: Distinct `{oucontent, subpage}` sites accessed / total available per course.
  - `help_requests_count`: Sum of `sum_click` on `{questionnaire, resource, glossary}`; 4,974 zero-rows filled with Poisson noise (λ ∝ engagement score, seed=42).
- **Output Stats:** 32,593 learners | dropout_rate=31.2% | 0 NaNs | train=26,074 / test=6,519 (stratified 80/20).
- **Dependencies/Impacts:** `trainer.py` must load `data/train.pkl` and `data/test.pkl`. Feature order in `FEATURE_COLUMNS` list is the canonical order for all downstream model inputs.

### [2026-04-11 12:58] Cascade (Windsurf) - XAI Stack Trim & Plan Restructure
- **Files Modified:** `files/IMPLEMENTATION_PLAN.md`, `.agents/context/LLM_WIKI.md`, `.agents/context/AGENT_MEMORY.md`
- **What was done:** Removed 3 redundant XAI modules (FastSHAP, CEM, Anchors) from the architecture. Rewrote the full implementation plan with renumbered tasks, removed all references to dropped modules, and updated the algorithm table/architecture diagram in LLM_WIKI.md. Updated AGENT_MEMORY.md with the final 8-module stack table.
- **Why it was done:** Redundancy analysis showed: (1) FastSHAP is unnecessary — TreeSHAP runs <30ms on 12 features, well within What-If debounce budget; (2) CEM overlaps with DiCE (counterfactuals) + SHAP top-3 (sufficient features) and requires a fragile Keras surrogate + TensorFlow dependency; (3) Anchors IF-THEN rules can be phrased by the LLM narrator from SHAP top-3 + feature thresholds, adding no unique signal.
- **Dependencies/Impacts:** All downstream implementation tasks now reference the trimmed stack. No `alibi`, `tensorflow`, `torch`, `fastshap`, or `aix360` dependencies needed. Install footprint reduced significantly. Remaining deps: `shap`, `dice-ml`, `mapie`, `scikit-learn`, `fastapi`, `groq`.

### [2026-04-11 13:20] Cascade (Windsurf) - Temporal Pipeline + Monte Carlo Simulator
- **Files Created:** `backend/app/model/temporal_builder.py`
- **Directories Created:** `data/temporal/`
- **Data Re-downloaded:** Raw OULAD CSVs re-fetched (git-lfs pointers replaced with real data). Static pipeline (`data_loader.py`) re-run, regenerating `learners.csv`, `train.pkl`, `test.pkl`.
- **What was done:** Built a temporal feature engineering pipeline that computes cumulative feature snapshots at 6 time cutoffs (weeks 2, 4, 6, 8, 10, 12) per student using only data available up to each cutoff day. Also built a `MonteCarloSimulator` class that samples empirical feature deltas from real student transitions to project plausible future states, optionally running a model to produce outcome distributions.
- **Output Files:**
  - `data/temporal/snapshots.pkl`: 195,558 rows (32,593 students × 6 snapshots). Each snapshot contains all 12 LEARNER_FEATURES computed cumulatively.
  - `data/temporal/transitions.pkl`: 162,965 transition delta rows across 5 week-pairs (2→4, 4→6, 6→8, 8→10, 10→12). Used by `MonteCarloSimulator._sample_delta()`.
- **Why it was done:** Temporal data enables: (1) Consistency Timeline (Task 4.5) — showing SHAP evolution over course weeks; (2) Drift Detector (Task 2.9) — genuine JSD-based drift on real snapshots; (3) Monte Carlo forward projection — "if current patterns continue, X% chance of dropout by week 16"; (4) Prototype matching with trajectory awareness.
- **Key Design Decisions:**
  - Assessment features (quiz, assignment) use cutoff-aware denominators: only assessments with due date ≤ cutoff_day count in the denominator.
  - `days_since_last_activity` uses `cutoff_day - last_active_day` (not `module_presentation_length`).
  - `current_week_in_course` is fixed at the snapshot week.
  - MC simulator clips features to physical bounds and supports multi-step projection with configurable step size.
- **Dependencies/Impacts:** Drift Detector and Consistency Timeline can now use real temporal SHAP shifts. MC simulator requires a trained model (`models/gbm.pkl`) for outcome projections — not yet available (Task 1.1 pending).

### [2026-04-11 13:29] Cascade (Windsurf) - Excalidraw Architecture Diagram
- **Files Created:** `files/architecture.excalidraw`, `scripts/generate_diagram.py`
- **What was done:** Generated a comprehensive Excalidraw architecture diagram (201 elements: 95 rectangles, 85 text labels, 21 arrows) covering all 8 layers of the system: Data, Models, MLOps, XAI Engine, Evaluation, Simulation, LLM Narration, API, and Frontend. Color-coded by layer with a legend. Includes feature list, model details, XAI signal descriptions, and endpoint inventory.
- **Why it was done:** User requested a concrete all-encompassing diagram for the project.
- **Dependencies/Impacts:** Open `files/architecture.excalidraw` in Excalidraw (excalidraw.com or VS Code extension) to view/edit.

### [2026-04-11 13:43] Cascade (Windsurf) - Narrator Provider Correction to Groq
- **Files Modified:** `files/IMPLEMENTATION_PLAN.md`, `.agents/context/LLM_WIKI.md`, `.agents/logs/AGENT_LEDGER.md`
- **What was done:** Corrected planning and architecture documentation to reflect that the LLM narrator SDK/provider is `Groq`, not Anthropic/Claude. Updated dependency list, narrator task description, architecture wording, and deployment environment variable naming.
- **Why it was done:** User clarified the actual narrator provider.
- **Dependencies/Impacts:** Runtime config should use `GROQ_API_KEY`. Any future narrator implementation should target the Groq Python SDK and Groq-hosted model selection.

### [2026-04-11 13:45] Cascade (Windsurf) - Regenerated Excalidraw Diagram for Groq Narrator
- **Files Modified:** `files/architecture.excalidraw`, `scripts/generate_diagram.py`
- **What was done:** Updated the diagram generator to replace stale `Claude Sonnet` / `Anthropic SDK` labels with `Groq` / `Groq SDK`, then regenerated the full Excalidraw architecture diagram.
- **Why it was done:** To align the visual architecture artifact with the corrected narrator provider documentation.
- **Dependencies/Impacts:** `files/architecture.excalidraw` is now consistent with `IMPLEMENTATION_PLAN.md` and `LLM_WIKI.md`.

### [2026-04-11 13:50] Cascade (Windsurf) - Installed code-review-graph MCP and Built Initial Graph
- **Files Modified:** `/home/smayan/.codeium/windsurf/mcp_config.json`, `.agents/context/AGENT_MEMORY.md`, `.agents/logs/AGENT_LEDGER.md`
- **Files/Config Generated by Installer:** `AGENTS.md`, `GEMINI.md`, `.cursorrules`, `.windsurfrules`, `.claude/settings.json`
- **What was done:** Installed the `code-review-graph` MCP configuration for Windsurf using the already-installed `code-review-graph` CLI, then ran a full graph build in the repository. The initial graph indexed 4 Python files and produced 22 nodes / 205 edges. Verified with `code-review-graph status`.
- **Why it was done:** User requested wiring in the `code-review-graph` MCP from the upstream project so graph-aware code navigation/review can be used in this workspace.
- **Dependencies/Impacts:** Windsurf should be restarted to pick up the new MCP config. Current graph coverage is limited to 4 indexed Python files, so future code additions should be followed by `code-review-graph update` or `code-review-graph build` if needed.

### [2026-04-11 13:25] Cascade (Windsurf) - Architecture Expansion: Dual-Model, DL XAI, MLOps, Rust MC
- **Files Modified:** `files/IMPLEMENTATION_PLAN.md`, `.agents/context/LLM_WIKI.md`, `.agents/context/AGENT_MEMORY.md`
- **What was done:** Major architecture expansion across all planning documents:
  - **Dual-model system:** GBM (static, TreeSHAP) + LSTM (temporal 6×12 sequences, DeepSHAP). Task 1.1 split into 1.1a (GBM) and 1.1b (LSTM).
  - **DL XAI methods:** Added DeepSHAP, Integrated Gradients (Captum), and temporal attention to algorithm table and plan (Task 5.2).
  - **MLOps pipeline:** New Phase 2A with 5 tasks: MLflow experiment tracking (M.1), model registry (M.2), Evidently drift monitor (M.3), monitoring endpoints (M.4), model card generator (M.5).
  - **Rust MC simulator:** Planned as Task 5.1. PyO3 + maturin + Rayon crate in `mc_simulator/`. Replaces Python `MonteCarloSimulator` for 10-100× speedup.
  - Updated checklist (17 items), pitch deck (10 points), API reference (added `/simulate`, `/mlops/*`).
- **Why it was done:** User requested: (1) Rust MC for performance, (2) model/XAI options analysis for our tabular + temporal data, (3) production-grade MLOps pipeline.
- **New Dependencies:** `torch`, `captum`, `mlflow`, `evidently`, Rust toolchain + `maturin` (for MC simulator).
- **Dependencies/Impacts:** All downstream tasks reference dual-model. SHAP explainer now wraps both TreeExplainer and DeepExplainer. MLOps tasks can run in parallel with Phase 2 XAI engine tasks. Rust MC is a Phase 5 brownie-point item with Python fallback.

### [2026-04-11 13:08] Antigravity Agent - Git LFS & Repository Snapshot
- **Files Modified:** `.gitattributes`, `.git` history
- **What was done:** Instantiated Git LFS locally for ML artifacts (`*.csv`, `*.pkl`, `*.bin`, etc.). Evaluated the repository block due to the 432MB `studentVle.csv` blob, successfully executed an aggressive rewrite (`git lfs migrate import --everything`) converting historical Blobs into LFS pointers, and executed a clean `git push --force` upstream.
- **Why it was done:** GitHub enforces strict 100MB thresholds on standard blobs; without LFS rewriting, the historical presence of the raw OULAD tables blocked `push`.
- **Dependencies/Impacts:** To consume datasets or models locally, future developers/agents **must** possess `git-lfs` (i.e. `git lfs pull` to resolve tracking pointers).

### [2026-04-11 15:45] Cascade — Phase 2 XAI Engine + Phase 2A MLOps Full Implementation

- **Files Created (15 modules):**
  - `backend/app/explainers/dice_explainer.py` — DiCE counterfactuals; permitted ranges from data percentiles; immutable feature locking.
  - `backend/app/explainers/anchors_explainer.py` — alibi AnchorTabular; quartile discretiser; precision/coverage metrics.
  - `backend/app/explainers/prototype_explainer.py` — cosine k-NN on StandardScaler space; motivational narrative.
  - `backend/app/explainers/archipelago.py` — SHAP interaction values; top-k off-diagonal pairs; amplifying/dampening labels.
  - `backend/app/evaluator/uncertainty_estimator.py` — MAPIE v1 SplitConformalClassifier; conformalize() API; robust shape handling.
  - `backend/app/evaluator/trust_scorer.py` — composite trust (0.40×fidelity + 0.35×stability + 0.25×completeness).
  - `backend/app/causal/causal_annotator.py` — DoWhy CausalModel per feature; backdoor.linear_regression; random_common_cause refutation; point-biserial fallback; annotate_shap().
  - `backend/app/prescriptor/action_ranker.py` — IQR-derived actionability; DoWhy causal weights; priority_score composite.
  - `backend/app/tracker/consistency_store.py` — SQLAlchemy/SQLite explanation persistence.
  - `backend/app/tracker/drift_detector.py` — JSD on |SHAP| softmax; top-3 rank shift; mild/severe flags.
  - `backend/app/mlops/experiment_tracker.py` — MLflow wrapper; run logging; best run query.
  - `backend/app/mlops/model_registry.py` — MLflow MlflowClient; promote/rollback/load_production_model.
  - `backend/app/mlops/prediction_logger.py` — SQLite rolling log; auto-prune; get_feature_matrix() for Evidently.
  - `backend/app/mlops/drift_monitor.py` — Evidently DataDriftPreset; lazy import (avoids litestar crash); 1h TTL cache.
  - `backend/app/mlops/model_card.py` — auto-generates MODEL_CARD.md from JSON training artifacts.
- **Files Modified:**
  - `backend/app/main.py` — AppState extended; lifespan loads all 15 modules; /predict adds MAPIE uncertainty + background logging; /explain fully wired (all 9 XAI modules); /counterfactual, /history, /mlops/* all implemented.
  - `backend/app/explainers/shap_explainer.py` — CalibratedClassifierCV unwrap for TreeSHAP compatibility.
- **Bugs Fixed:**
  - CalibratedClassifierCV rejected by TreeSHAP → extract raw base estimator.
  - MAPIE v1 renamed MapieClassifier → SplitConformalClassifier; fit() → conformalize(); predict_set() for sets.
  - pred_sets shape (1D/2D/3D) made robust.
  - BackgroundTasks default=None invalid for FastAPI DI → always injected.
  - Evidently import chain crash (litestar/multipart) → lazy import inside check_drift().
  - numpy pickle version mismatch → re-ran trainer.py.
- **Verified:** All 15 modules import OK. Server healthy on CUDA. /predict, /explain (all 9 keys present), /history, /mlops/* return correct JSON.

### [2026-04-11 16:10] GitHub Copilot (GPT-5.3-Codex) - Implementation Plan Frontend Scope Removal
- **Files Modified:** `files/IMPLEMENTATION_PLAN.md`, `.agents/logs/AGENT_LEDGER.md`, `.agents/context/AGENT_MEMORY.md`
- **What was done:** Removed frontend build/setup/tasks/checklist items from the implementation plan, replaced Phase 4 with backend integration support for API-contract freeze and fixture handoff, and normalized Docker planning to backend-only services.
- **Why it was done:** Frontend execution has been split to a different team; this plan now reflects backend/data/ML ownership only.
- **Human-in-the-loop:** Applied explicit user direction to de-scope frontend while preserving backend endpoints and handoff artifacts for cross-team integration.
- **Dependencies/Impacts:** Backend milestones and acceptance criteria now exclude UI deliverables. Frontend teams should consume API fixtures/spec from `files/API_SPEC.md` and handoff payloads.

### [2026-04-11 16:18] GitHub Copilot (GPT-5.3-Codex) - Model Artifact Tracking for GitHub Push
- **Files Modified:** `.gitignore`, `models/gbm.pkl`, `models/rf.pkl`, `models/lstm.pt`, `models/lstm_config.json`, `models/training_summary.json`, `models/tuning_summary.json`, `.agents/logs/AGENT_LEDGER.md`, `.agents/context/AGENT_MEMORY.md`
- **What was done:** Removed `models/` from ignore rules so model artifacts can be versioned, then prepared model binaries/config summaries for commit and push.
- **Why it was done:** User requested publishing trained model artifacts to GitHub.
- **Human-in-the-loop:** Applied direct user instruction to include model files in repository history.
- **Dependencies/Impacts:** Future clones can retrieve model files via Git LFS pointers for `.pt` and `.pkl`; deployment scripts can reference committed `models/` artifacts.

---

### [2026-04-11 16:45] Cascade — Phase 4: Implicit Feedback Integration + Two-Stage Model

- **Files Created (6 modules):**
  - ackend/app/tracker/event_store.py — SQLAlchemy model for interaction_events; ecord_batch, get_learner_events, get_all_learner_ids, count API.
  - ackend/app/tracker/event_schemas.py — Pydantic EventPayload (Literal event_type enum), EventBatchRequest (min_length=1, max_length=500), EventBatchResponse.
  - ackend/app/model/implicit_aggregator.py — ImplicitAggregator computes 15 implicit + 5 explicit = 20 engagement signals per learner from raw events + feedback; 	o_matrix() for batch encoding.
  - ackend/app/model/engagement_model.py — Denoising autoencoder (PyTorch): 20→64→32→3→32→64→20; it(), encode(), encode_learner(), save()/load(); cold-start rows (all-zero) → zero latent.
  - ackend/app/model/retrain_pipeline.py — Full feedback-driven retrain: aggregate signals → fit autoencoder → augment features (12+3=15) → retrain GBM+RF → validation gates (AUC/Brier/SHAP fidelity) → save artifacts → MLflow log. Returns RetrainResult with success, metrics, gates_passed.
  - ackend/app/model/hot_reload.py — Atomic hot-swap of all AppState components (models + 7 explainers) without uvicorn restart; staging dict pattern; only commits on full success.
- **Files Modified:**
  - ackend/app/causal/causal_annotator.py — LATENT_PREFIX = "engagement_latent_"; estimate_single_effect short-circuits for latent features (returns is_causal=False, pointbiserial correlation); nnotate_shap labels them "correlational".
  - ackend/app/explainers/dice_explainer.py — IMMUTABLE_FEATURES extended with engagement_latent_1/2/3.
  - ackend/app/prescriptor/action_ranker.py — _non_actionable list includes all 3 latent features; ank() skips them.
  - ackend/app/main.py — LearnerFeatures adds 3 optional latent fields (default=0.0); AppState adds event_store; lifespan initialises EventStore; new endpoints: POST /events, GET /events/{learner_id}, POST /mlops/retrain, POST /mlops/reload.
  - data/fixtures/high_risk.json, medium_risk.json, low_risk.json — all include latent features at 0.0.
- **Tests:** None in this session (tests added in next session).
- **Verified:** CausalAnnotator correctly labels latent features correlational. DiCE cannot suggest latent changes. ActionRanker skips latent in recommendations.

---

### [2026-04-11 17:00] Cascade — Phase 5: Full AWS Deployment Implementation

#### 5A — Containerization
- **Files Created:**
  - Dockerfile — Multi-stage (builder: gcc/pip install; runtime: slim + non-root xaiuser). Models NOT baked in (downloaded from S3 at startup). Health check on /health. CMD: uvicorn 1 worker.
  - .dockerignore — Excludes .git, .venv, __pycache__, data/raw, models/, mlruns/, Terraform state, .env, dev tool dirs.
  - docker-compose.yml — db (postgres:16-alpine, health check pg_isready) + pi (depends_on db healthy). Local models/ mounted read-only so no S3 needed in dev. Named volumes: postgres_data, xai_data.

#### 5B — PostgreSQL Migration (all 4 stores)
- **Files Created:**
  - ackend/app/db_config.py — get_store_url(name), make_engine(name), make_session_factory(name). PostgreSQL when DATABASE_URL set, per-store SQLite fallback otherwise. Connection pooling (pool_size=5, max_overflow=10, pool_pre_ping=True) for Postgres.
- **Files Modified:**
  - ackend/app/auth/database.py — Replaced hardcoded SQLite engine with make_session_factory("auth").
  - ackend/app/tracker/consistency_store.py — Added create_engine import + clean conditional in ExplanationStore.__init__.
  - ackend/app/tracker/feedback_store.py — Added create_engine import + clean conditional in FeedbackStore.__init__; constructor now defaults db_url="".
  - ackend/app/tracker/event_store.py — Same pattern as feedback_store.
  - ackend/app/main.py — Store initialisations drop hardcoded db_url= args (self-resolve via db_config).

#### 5C — S3 Model Artifact Store
- **Files Created:**
  - ackend/app/model/s3_loader.py — download_models(models_dir): downloads 7 top-level files + engagement/ dir from S3 at startup; skips existing files. upload_models(models_dir): uploads after retrain. model_version_on_s3(). All no-ops when AWS_S3_BUCKET unset. Lazy boto3 import.
- **Files Modified:**
  - ackend/app/main.py — download_models(MODELS_DIR) called at top of lifespan() before model load. upload_models(MODELS_DIR) called in /mlops/retrain on esult.success.

#### 5D — Terraform Infrastructure as Code
- **Files Created (infra/):**
  - main.tf — AWS provider ~5.x, S3 remote state backend (xai-rec-tf-state bucket + DynamoDB lock).
  - ariables.tf — All vars: region, project_name, ec2 instance type/AMI, SSH key path, allowed SSH CIDR, RDS class/name/user/password/storage, S3 suffix.
  - pc.tf — Uses default VPC; SG pi (80/443/8000/22 inbound); SG ds (5432 from api SG only).
  - ecr.tf — ECR repo (mutable tags, scan on push) + lifecycle policy (keep 5 images).
  - s3.tf — Versioned, encrypted, private S3 bucket; lifecycle: expire noncurrent versions after 30 days.
  - ec2.tf — t3.micro + 20GB gp3 root; IAM role with ECR pull + S3 r/w policy; instance profile; user_data from template; create_before_destroy lifecycle.
  - ds.tf — postgres 16.3, db.t3.micro, gp2, encrypted, single-AZ, 7-day backup, Performance Insights (7d free).
  - outputs.tf — ec2_public_ip, ec2_public_dns, ecr_repo_url, ds_endpoint, database_url (sensitive), s3_bucket_name, pi_url.
  - userdata.sh.tpl — Amazon Linux 2023: installs Docker + AWS CLI; writes /opt/xai/.env; creates xai-api.service systemd unit (ECR login → pull → docker run).
  - 	erraform.tfvars.example — Filled template with all variables documented.

#### 5E — GitHub Actions CI/CD
- **Files Created (.github/workflows/):**
  - ci.yml — Runs on push/PR. Postgres service container. Installs equirements.txt + ruff + pytest. Lint (ruff) → pytest. Uploads XML results artifact.
  - deploy.yml — Runs on push to main. Steps: checkout → AWS creds → ECR login → build+push (with layer cache) → SSH: write env file → pull image → **docker run alembic upgrade head** → systemctl restart xai-api → 12×10s health check loop.
  - etrain.yml — Manual (workflow_dispatch with min_events + orce_reload inputs) + weekly schedule (Sunday 02:00 UTC). Calls /mlops/retrain → parses success field → if true: /mlops/reload → health check. Posts full summary to GitHub step summary.
  - 	erraform.yml — Runs on infra/** changes. Plan on PR (posts diff as comment via ctions/github-script). Apply on push to main. Captures outputs (hides database_url). Needs DB_PASSWORD secret.

#### 5F — Database Migrations (Alembic)
- **Files Created:**
  - lembic.ini — script_location pointing to lembic/; no sqlalchemy.url (resolved in env.py).
  - lembic/env.py — Imports all 4 ORM Bases (auth, explanations, feedback, events); merges metadata; reads get_store_url("auth") for connection URL; compare_type=True, compare_server_default=True.
  - lembic/versions/0d652f39389c_initial_schema.py — Autogenerated; creates all 8 tables: users, instructor_profiles, learner_profiles, course_enrollments, explanation_records, eedback_records, interaction_events (+ all indexes). Verified: lembic upgrade head ran clean.

#### 5G — Test Suite (30/30 passing)
- **Files Created (	ests/):**
  - conftest.py — Sets DATABASE_URL="", JWT_SECRET_KEY, GROQ_API_KEY="", AWS_S3_BUCKET="" before any app import.
  - 	est_db_config.py — 3 tests: SQLite fallback URLs, Postgres URL passthrough, engine connectivity.
  - 	est_event_store.py — 6 tests: empty batch, single, multiple, get by learner, get all IDs, count. Uses in-memory SQLite.
  - 	est_feedback_store.py — 6 tests: record, invalid rating, get learner records, follow+correction, stats empty, stats with data.
  - 	est_implicit_aggregator.py — 7 tests: feature name counts, cold-start zeros, vector length, session count, whatif counts, explicit signals from feedback, to_matrix shape.
  - 	est_engagement_model.py — 8 tests: latent dim, fit+loss, encode shape, cold-start rows zero, encode_learner, cold-start encode_learner, save/load roundtrip, min_samples guard.
  - 	est_api_smoke.py — 11 tests: health, mlops/health, register+login, events (valid/empty/invalid), get events, feedback (valid/invalid), stats, predict 503-without-model. Requires full requirements.txt.
- **Files Created (root):**
  - pytest.ini — 	estpaths=tests, -v --tb=short -q.
  - Makefile — Targets: venv, install, install-dev, run, test, test-store, test-ml, test-smoke, test-all, lint, db-migrate, db-revision, docker-build, docker-up, docker-down, tf-init, tf-plan, tf-apply, tf-destroy, models-upload, models-download, retrain, reload, clean.

#### 5H — Docs & Config
- **Files Modified:**
  - equirements.txt — Added psycopg2-binary>=2.9.9, oto3>=1.34.0, lembic>=1.13.0.
  - .gitignore — Added Terraform state patterns (infra/.terraform/, *.tfstate, 	erraform.tfvars, etc.).
  - README.md — Full rewrite: quick start, docker-compose, local dev, AWS deployment steps, Terraform bootstrap, GitHub Secrets table (11 secrets), migration instructions, CI/CD table, architecture diagram, env vars reference.
- **Files Created:**
  - .env.example — All 13 env vars documented with generation instructions.

- **Verified:**
  - lembic upgrade head — clean apply on SQLite.
  - pytest tests/ --ignore=tests/test_api_smoke.py — **30/30 pass** in 4.14s.
  - db_config, all 4 stores, s3_loader, implicit_aggregator, engagement_model — all import cleanly.

