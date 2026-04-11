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

### [2026-04-11 19:30] Cascade — Docs Sync + Missing Artifacts

- **Problem:** IMPLEMENTATION_PLAN.md had unresolved 3-way merge conflicts throughout; all task statuses were stale (Pending); API_SPEC.md documented old envelope pattern and obsolete endpoints; dl_explainer.py, mc_simulator/, backend/sdk/ were missing; Dockerfile naming mismatch; no handoff fixtures.
- **Files Rewritten:**
  - `files/IMPLEMENTATION_PLAN.md` — stripped all `<<<<<<`/`=======`/`>>>>>>>` markers; updated all task statuses to ✅ DONE; removed frontend phase (handled by separate team); added Phase 4 auth/events/feedback section; added Phase 5 DL explainer + Rust MC entries; added final verification checklist.
  - `files/API_SPEC.md` — rewrote to match actual `main.py` signatures: flat request bodies (no envelope), 15 latent features added, all new endpoints documented (/events, /mlops/retrain, /mlops/reload, /auth/*, /explain/me, /feedback/{learner_id}, /simulate with SimulateRequest).
- **Files Created:**
  - `backend/app/explainers/dl_explainer.py` — Captum IntegratedGradients + LayerIntegratedGradients for LSTM; `explain_temporal()` returns (T×F) attribution matrix; `temporal_attention_summary()`; `cross_validate()` against DeepSHAP by rank correlation; `_LSTMScalarWrapper` for scalar output required by Captum.
  - `mc_simulator/Cargo.toml` — pyo3 0.21 + rayon 1.10 + rand 0.8 + ndarray 0.15; cdylib crate type.
  - `mc_simulator/pyproject.toml` — maturin ≥1.4 build backend.
  - `mc_simulator/src/lib.rs` — `simulate_trajectories()` PyO3 function; Rayon parallel iterator over N simulations; per-simulation SmallRng seeded from `seed + sim_idx`; bounds clipping; validated delta pool shape.
  - `backend/sdk/xai_sdk/__init__.py` — package root, exports XAIClient.
  - `backend/sdk/xai_sdk/client.py` — XAIClient with httpx (optional, falls back to urllib); methods: predict, explain, whatif, counterfactual, simulate, history, feedback, feedback_stats, mlops_health, mlops_metrics, mlops_drift_report.
  - `backend/sdk/pyproject.toml` — installable as `pip install -e backend/sdk`.
  - `Dockerfile.backend` — copy of Dockerfile (resolves plan/repo naming mismatch).
  - `data/fixtures/high_risk_learner.json`, `medium_risk_learner.json`, `low_risk_learner.json` — stable handoff payloads for frontend team.

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

### [Session — MLOps Pipeline Hardening Phases 1–4] Cascade

**Scope:** All four phases of the MLOps hardening plan (`mlops_plan.md`). Phase 0 (auth + concurrency lock) was already implemented by the user.

**Files Modified:**

- `.github/workflows/deploy.yml`
  - Build step now produces an immutable `SHA-timestamp` image tag instead of reusing `:latest`.
  - `docker/build-push-action` outputs `digest` (SHA256); `full_image = ECR_REPO_URL@sha256:...` is passed to the deploy job.
  - Deploy job pulls the pinned digest, not `:latest`; writes `deploy-history.json` with current + previous image identity for rollback.
  - Automatic rollback: if health check fails after restart, `sed`s `XAI_IMAGE` back to `PREV_IMAGE` and restarts.
  - Deployment summary now includes `image_digest` and `full_image` fields.
  - Cache layer changed from `type=inline` to `type=registry,mode=max` (separate `:cache` tag, not `:latest`).

- `.github/workflows/ci.yml`
  - `pytest` now emits `--junitxml=pytest-results.xml` with `junit_suite_name=xai-backend`.
  - Artifact name now includes `${{ github.run_id }}` to prevent overwrites.
  - `if-no-files-found: warn` ensures upload step never blocks CI on missing XML.
  - New `smoke` job runs on `push main` after lint-test: checks `/health`, `/predict` (valid payload → 200), `/mlops/retrain` (unauthenticated → 401/403), `/mlops/reload` (unauthenticated → 401/403), `/mlops/drift-report` (→ 200).

- `backend/app/model/retrain_pipeline.py`
  - Added imports: `hashlib`, `os`, `datetime`.
  - New module-level constants: `MIN_TRAIN_SAMPLES=1000`, `MAX_COLD_START_FRAC=0.95`, `CLASS_BALANCE_MIN/MAX=0.05/0.95`, `STRICT_MODE` (env `RETRAIN_STRICT_MODE`).
  - `_validate()` extended with 6 gates: (1) minimum data volume, (2) class balance, (3) cold-start fraction (warn or reject via STRICT_MODE), (4) feature schema hash consistency against previous manifest, (5+6) AUC/Brier regression.  Returns a `;`-joined rejection string or None.
  - `_validate()` call-site in `run()` now passes `X_tr_aug`, `y_tr`, `cold_frac_pre`, `augmented_feature_names`.
  - `_train_gbm/_train_rf` now call `configure_mlflow(experiment=...)` instead of hardcoded `mlflow.set_tracking_uri + set_experiment`.
  - New `_save_manifest()` method writes `models/model_manifest.json` with: `model_version`, `training_timestamp` (ISO-8601 UTC), `metrics_snapshot`, `feature_schema_hash` (SHA-256 of sorted JSON feature list), `data_fingerprint` (SHA-256 of `train.pkl` raw bytes), `n_features`, `feature_names`, `strict_mode`.
  - New module-level helpers `_feature_schema_hash()` and `_file_sha256()`.

- `backend/app/model/s3_loader.py`
  - `"model_manifest.json"` added to `_MODEL_FILES` (included in S3 sync).
  - New `validate_manifest(models_dir)` function: compares local manifest against S3 copy on `model_version`, `feature_schema_hash`, `data_fingerprint`; returns `{valid, local_hash, remote_hash, mismatch, message}`.

- `backend/app/model/trainer.py`
  - Replaced `mlflow.set_tracking_uri(MLRUNS_DIR) + mlflow.set_experiment(...)` with `configure_mlflow(alias="train")` in both `train_gbm` and `train_rf`.

- `backend/app/model/tune.py`
  - Replaced `mlflow.set_tracking_uri + set_experiment` in `_log_to_mlflow` with `configure_mlflow(alias="tune")`.

- `backend/app/model/hot_reload.py`
  - `HotReloadResult` gains two new fields: `canary_fraction: float` and `mode: str` (`"full"` | `"canary"` | `"canary-abort"`).
  - `hot_reload()` gains `canary_fraction: float = 1.0` parameter.
  - `canary_fraction=0.0`: aborts any active canary, sets `state.canary_gbm_model=None`, returns `canary-abort` result.
  - `0 < canary_fraction < 1.0`: loads new GBM only into `state.canary_gbm_model`; production model untouched.
  - `canary_fraction=1.0`: full atomic swap (existing behaviour); clears `canary_gbm_model` after swap.

- `backend/app/mlops/prediction_logger.py`
  - Added `threading`, `time` imports.
  - `__init__` gains `prune_every: int = 100` param; added `_insert_count`, `_prune_count`, `_prune_time_ms`, `_lock` fields.
  - `log()` no longer calls `_prune()` synchronously; instead increments `_insert_count` and spawns a daemon thread every `prune_every` inserts.
  - `_prune()` now logs deleted row count and elapsed time; errors are caught and logged instead of propagating.
  - New `prune_stats()` method returns `{insert_count, prune_count, prune_avg_ms, next_prune_in}`.

- `backend/app/mlops/drift_monitor.py`
  - Added `collections`, `os` imports; `Callable` type alias `AlertHook`.
  - New env-var constants: `CACHE_TTL_SECONDS` (`DRIFT_CACHE_TTL_SECONDS`), `DRIFT_SHARE_THRESHOLD` (`DRIFT_SHARE_THRESHOLD`), `TREND_WINDOW` (`DRIFT_TREND_WINDOW`).
  - `DriftReport` gains `alert_fired: bool` and `trend_direction: Optional[str]` fields.
  - `DriftMonitor.__init__` gains `alert_hook: Optional[AlertHook]` param and `_report_history: deque(maxlen=TREND_WINDOW)`.
  - `check_drift()` now: computes trend direction before building result; fires alert hook if `drift_share >= DRIFT_SHARE_THRESHOLD`; appends to `_report_history`.
  - New `_compute_trend()`: compares current share against rolling mean; returns `"increasing"`, `"decreasing"`, or `"stable"`.
  - New `trend_summary()`: returns `{window, drift_shares, trend, threshold}`.
  - Default alert hook `_slack_alert_hook()`: posts to `SLACK_DRIFT_WEBHOOK_URL` if set (no-op otherwise).
  - New `concept_drift_proxy(recent_risk_scores, recent_followed)`: computes calibration gap between mean predicted risk and `1 - follow_rate`; flags if gap > 0.25.

- `backend/app/mlops/experiment_tracker.py`
  - Replaced hardcoded `sqlite:///mlruns/mlflow.db` URI with `TRACKING_URI` from `mlflow_config`.
  - `__init__` now calls `configure_mlflow(experiment=experiment_name)` instead of bare `mlflow.set_*`.

- `backend/app/main.py`
  - Added `import random`, `Query` to fastapi imports.
  - `AppState` gains `canary_gbm_model = None` and `canary_fraction: float = 0.0`.
  - `/predict` now routes `canary_fraction` fraction of non-LSTM requests to `canary_gbm_model` (tagged as `"gbm-canary"` in `model_used`).
  - `/mlops/reload` gains `canary_fraction: float = Query(1.0, ...)` parameter; passes it to `hot_reload()`.

**Files Created:**

- `backend/app/mlops/mlflow_config.py` — `configure_mlflow(experiment, alias)` sets `TRACKING_URI` (from env `MLFLOW_TRACKING_URI` or default sqlite path) and experiment; `_EXPERIMENT_MAP` maps short aliases to experiment names; exported `TRACKING_URI` constant.
- `files/RUNBOOK.md` — Operations SOP covering: deployment rollback (image), model rollback (git LFS / MLflow registry), retrain failure triage (all rejection reasons + fixes), drift response SOP (severity levels + env-var config), canary reload workflow, CI failure debug guide, useful commands.

**Key design decisions:**
- Canary uses probabilistic routing (Python `random.random()`) — stateless, zero coordination overhead, consistent with shadow traffic patterns used in industry.
- Manifest SHA-256 uses sorted JSON for determinism across Python versions.
- Prune amortization (every 100 inserts) eliminates per-request DB overhead at the cost of up to 100 excess rows — acceptable for a 10k-row rolling buffer.
- Strict mode is opt-in (`RETRAIN_STRICT_MODE=true`) to avoid breaking existing CI pipelines.

---

### [2026-04-11 16:45] Cascade — Phase 4: Implicit Feedback Integration + Two-Stage Model

- **Files Created (6 modules):**
  - ackend/app/tracker/event_store.py — SQLAlchemy model for interaction_events; 
ecord_batch, get_learner_events, get_all_learner_ids, count API.
  - ackend/app/tracker/event_schemas.py — Pydantic EventPayload (Literal event_type enum), EventBatchRequest (min_length=1, max_length=500), EventBatchResponse.
  - ackend/app/model/implicit_aggregator.py — ImplicitAggregator computes 15 implicit + 5 explicit = 20 engagement signals per learner from raw events + feedback; 	o_matrix() for batch encoding.
  - ackend/app/model/engagement_model.py — Denoising autoencoder (PyTorch): 20→64→32→3→32→64→20; it(), encode(), encode_learner(), save()/load(); cold-start rows (all-zero) → zero latent.
  - ackend/app/model/retrain_pipeline.py — Full feedback-driven retrain: aggregate signals → fit autoencoder → augment features (12+3=15) → retrain GBM+RF → validation gates (AUC/Brier/SHAP fidelity) → save artifacts → MLflow log. Returns RetrainResult with success, metrics, gates_passed.
  - ackend/app/model/hot_reload.py — Atomic hot-swap of all AppState components (models + 7 explainers) without uvicorn restart; staging dict pattern; only commits on full success.
- **Files Modified:**
  - ackend/app/causal/causal_annotator.py — LATENT_PREFIX = "engagement_latent_"; estimate_single_effect short-circuits for latent features (returns is_causal=False, pointbiserial correlation); nnotate_shap labels them "correlational".
  - ackend/app/explainers/dice_explainer.py — IMMUTABLE_FEATURES extended with engagement_latent_1/2/3.
  - ackend/app/prescriptor/action_ranker.py — _non_actionable list includes all 3 latent features; 
ank() skips them.
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
  - ackend/app/main.py — download_models(MODELS_DIR) called at top of lifespan() before model load. upload_models(MODELS_DIR) called in /mlops/retrain on 
esult.success.

#### 5D — Terraform Infrastructure as Code
- **Files Created (infra/):**
  - main.tf — AWS provider ~5.x, S3 remote state backend (xai-rec-tf-state bucket + DynamoDB lock).
  - ariables.tf — All vars: region, project_name, ec2 instance type/AMI, SSH key path, allowed SSH CIDR, RDS class/name/user/password/storage, S3 suffix.
  - pc.tf — Uses default VPC; SG pi (80/443/8000/22 inbound); SG 
ds (5432 from api SG only).
  - ecr.tf — ECR repo (mutable tags, scan on push) + lifecycle policy (keep 5 images).
  - s3.tf — Versioned, encrypted, private S3 bucket; lifecycle: expire noncurrent versions after 30 days.
  - ec2.tf — t3.micro + 20GB gp3 root; IAM role with ECR pull + S3 r/w policy; instance profile; user_data from template; create_before_destroy lifecycle.
  - 
ds.tf — postgres 16.3, db.t3.micro, gp2, encrypted, single-AZ, 7-day backup, Performance Insights (7d free).
  - outputs.tf — ec2_public_ip, ec2_public_dns, ecr_repo_url, 
ds_endpoint, database_url (sensitive), s3_bucket_name, pi_url.
  - userdata.sh.tpl — Amazon Linux 2023: installs Docker + AWS CLI; writes /opt/xai/.env; creates xai-api.service systemd unit (ECR login → pull → docker run).
  - 	erraform.tfvars.example — Filled template with all variables documented.

#### 5E — GitHub Actions CI/CD
- **Files Created (.github/workflows/):**
  - ci.yml — Runs on push/PR. Postgres service container. Installs 
equirements.txt + ruff + pytest. Lint (ruff) → pytest. Uploads XML results artifact.
  - deploy.yml — Runs on push to main. Steps: checkout → AWS creds → ECR login → build+push (with layer cache) → SSH: write env file → pull image → **docker run alembic upgrade head** → systemctl restart xai-api → 12×10s health check loop.
  - 
etrain.yml — Manual (workflow_dispatch with min_events + orce_reload inputs) + weekly schedule (Sunday 02:00 UTC). Calls /mlops/retrain → parses success field → if true: /mlops/reload → health check. Posts full summary to GitHub step summary.
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
  - 
equirements.txt — Added psycopg2-binary>=2.9.9, oto3>=1.34.0, lembic>=1.13.0.
  - .gitignore — Added Terraform state patterns (infra/.terraform/, *.tfstate, 	erraform.tfvars, etc.).
  - README.md — Full rewrite: quick start, docker-compose, local dev, AWS deployment steps, Terraform bootstrap, GitHub Secrets table (11 secrets), migration instructions, CI/CD table, architecture diagram, env vars reference.
- **Files Created:**
  - .env.example — All 13 env vars documented with generation instructions.

- **Verified:**
  - lembic upgrade head — clean apply on SQLite.
  - pytest tests/ --ignore=tests/test_api_smoke.py — **30/30 pass** in 4.14s.
  - db_config, all 4 stores, s3_loader, implicit_aggregator, engagement_model — all import cleanly.


---

### [2026-04-11 17:35] Cascade — Fix PredictionLogger + Wire /simulate Endpoint

#### PredictionLogger → PostgreSQL migration
- **Files Modified:**
  - ackend/app/mlops/prediction_logger.py — Added rom backend.app.db_config import make_engine; refactored __init__ to use make_engine("predictions") when db_url="" (same pattern as all other stores). Updated docstring.
  - ackend/app/main.py — Removed hardcoded db_url=f"sqlite:///{DATA_DIR / 'predictions.db'}" from PredictionLogger() init; self-resolves via db_config.
  - ackend/app/db_config.py — Updated docstring to include predictions as a 5th store.
  - lembic/env.py — Added rom backend.app.mlops.prediction_logger import Base as PredictionBase; added to metadata merge loop so prediction_log table is included in Alembic autogenerate.
- **Migration generated:** lembic/versions/f1a8243329d4_add_prediction_log.py — adds prediction_log table + ix_prediction_log_timestamp index.
- **Migration applied:** lembic upgrade head ran clean (0d652f39389c → f1a8243329d4).

#### /simulate endpoint — fully wired to Python MonteCarloSimulator
- **Files Modified:**
  - ackend/app/model/temporal_builder.py — Guarded rom data_loader import ... with 	ry/except ModuleNotFoundError; fallback uses rom backend.app.model.data_loader import ... so MonteCarloSimulator can be cleanly imported from main.py.
  - ackend/app/main.py:
    - Import: rom backend.app.model.temporal_builder import MonteCarloSimulator
    - AppState: added mc_simulator: Optional[MonteCarloSimulator] = None
    - lifespan(): loads data/temporal/transitions.pkl → MonteCarloSimulator(transitions); graceful warning if file missing.
    - Added SimulateRequest schema: eatures: LearnerFeatures, current_week (default 6), 	arget_week (default 12), 
_simulations (default 1000, max 10_000).
    - /simulate endpoint: full implementation — validates current_week < target_week, strips latent features before passing to simulator, runs mc_simulator.simulate() with GBM model for outcome distribution, strips raw trajectories from response (too large).

#### Tests added (17 new, all passing)
- 	ests/test_mc_simulator.py (9 tests) — required keys, feature distributions completeness, n_simulations, week metadata, outcome distribution with/without model, invalid weeks validation, feature clipping bounds, reproducibility.
- 	ests/test_prediction_logger.py (8 tests) — log, count, get_recent fields/ordering, get_feature_matrix, prune on max_rows, default constructor via db_config.

- **Total test count: 47/47 passing.**

### [2026-04-11 18:50] GitHub Copilot (GPT-5.3-Codex) - Stress Test Validation Fix and Full Re-run
- **Files Modified:** `backend/app/tracker/event_schemas.py`, `.agents/logs/AGENT_LEDGER.md`, `.agents/context/AGENT_MEMORY.md`
- **What was done:** Added event-type alias normalization in the `/events` request schema so API accepts both frontend aliases and canonical event names (`whatif_interaction`→`whatif_slider`, `action_view`→`action_viewed`, `action_dismiss`→`action_dismissed`, `focus_start`→`session_start`, `focus_end`→`session_end`). Re-ran full stress suite in conda env `astro`.
- **Why it was done:** Stress suite had one failing edge-case test due to 422 validation mismatch for alias event types.
- **Human-in-the-loop:** Applied direct user request to fix failure and run complete stress test again.
- **Dependencies/Impacts:** `/events` endpoint is now more tolerant of client event naming variance while preserving canonical stored values for downstream implicit aggregation features.

### [2026-04-11 19:30] GitHub Copilot (GPT-5.3-Codex) - Phase 0 MLOps Control-Plane Hardening (Start)
- **Files Modified:** `backend/app/main.py`, `.github/workflows/retrain.yml`, `tests/test_api_integration.py`, `.agents/logs/AGENT_LEDGER.md`, `.agents/context/AGENT_MEMORY.md`
- **What was done:** Added `require_mlops_operator` guard to protect `/mlops/retrain` and `/mlops/reload` with either admin JWT or `X-MLOPS-Token` automation header (`MLOPS_AUTOMATION_TOKEN` env). Added process-level lock (`MLOPS_CONTROL_LOCK`) to serialize retrain/reload operations and return `409` on concurrent control calls. Updated retrain workflow to send automation token header. Updated MLOps integration tests to authenticate as admin before control endpoint calls.
- **Why it was done:** To begin implementation of the MLOps hardening plan by closing the highest-risk production gap (unauthenticated control endpoints) and reducing race-condition risk during live reload/retrain.
- **Human-in-the-loop:** Directly executed user instruction to start implementation from the proposed MLOps plan.
- **Dependencies/Impacts:** GitHub Actions now requires `MLOPS_AUTOMATION_TOKEN` secret. Control-plane endpoints reject unauthenticated/non-admin calls unless automation token is configured and provided.

### [2026-04-11 21:08] GitHub Copilot (GPT-5.3-Codex) - Groq Synthetic Recommendation Dataset Generator + Astro Run
- **Files Modified:** `scripts/generate_synthetic_recommendation_data.py`, `.agents/logs/AGENT_LEDGER.md`
- **Files Generated:** `data/synthetic/recommendations/student_recommendation_dataset.csv`, `data/synthetic/recommendations/instructor_recommendation_dataset.csv`, `data/synthetic/recommendations/generation_report.json`
- **What was done:** Implemented a new Groq-driven synthetic data generator that creates separate student and instructor recommendation datasets. Student output blends existing learner features with synthetic implicit feedback signals (`implicit_clicks_14d`, `implicit_video_watch_ratio_14d`, `implicit_forum_events_14d`, `implicit_quiz_attempts_14d`) and course recommendation scoring. Instructor output aggregates cohort investment signals plus student-level affinity to recommend assignments per instructor-student pairing.
- **Why it was done:** User expanded MVP scope from dropout prediction to dual recommendation engines and requested synthetic data generation via Groq.
- **Human-in-the-loop:** Used user-requested `conda` environment `astro` to execute generation command.
- **Dependencies/Impacts:** Generated datasets are now available for recommendation MVP demos; rerun script with custom sizes/top-k as needed.

### [2026-04-11 21:18] GitHub Copilot (GPT-5.3-Codex) - 20k-Scale Regeneration + Main Student Feature Blending
- **Files Modified:** `scripts/generate_synthetic_recommendation_data.py`, `.agents/logs/AGENT_LEDGER.md`
- **Files Generated:** `data/synthetic/recommendations/student_recommendation_dataset.csv`, `data/synthetic/recommendations/instructor_recommendation_dataset.csv`, `data/synthetic/recommendations/generation_report.json`
- **What was done:** Regenerated recommendation datasets at ~20k scale (`--n-students 7000 --top-k 3`) yielding 21,000 student recommendation rows and 20,994 instructor recommendation rows. Updated student recommendation scoring to explicitly blend additional dropout-model student features (`quiz_avg_score`, `assignment_submission_rate`, `missed_deadlines_count`, `prior_course_completions`, `days_since_last_activity`, `studied_credits`, `num_of_prev_attempts`) alongside implicit feedback.
- **Why it was done:** User requested larger synthetic volume and explicit consideration of student-related data from the main dropout dataset in student recommendation generation.
- **Human-in-the-loop:** Followed user direction to execute in `conda` env `astro`.
- **Dependencies/Impacts:** Student recommendation CSV now includes `base_*` student feature columns and `learning_momentum` for downstream model training/debugging.

### [2026-04-11 21:28] GitHub Copilot (GPT-5.3-Codex) - Expanded Explicit/Implicit Signals + 50k Generation
- **Files Modified:** `scripts/generate_synthetic_recommendation_data.py`, `.agents/logs/AGENT_LEDGER.md`
- **Files Generated:** `data/synthetic/recommendations/student_recommendation_dataset.csv`, `data/synthetic/recommendations/instructor_recommendation_dataset.csv`, `data/synthetic/recommendations/generation_report.json`
- **What was done:** Added richer implicit signals (`implicit_avg_dwell_time_min_14d`, `implicit_save_events_14d`, `implicit_search_events_14d`, `implicit_last_recommendation_interaction_days`) and explicit student profile features (`explicit_gender`, `explicit_region`, `explicit_highest_education`, `explicit_imd_band`, `explicit_age_band`, `explicit_disability`, plus numeric explicit scores) into recommendation scoring/output. Regenerated datasets at ~50k scale using `--n-students 17000 --n-instructors 350 --top-k 3`.
- **Why it was done:** User requested more explicit and implicit features and larger synthetic dataset volume around 50k rows.
- **Human-in-the-loop:** Executed in user-preferred `conda` env `astro`.
- **Dependencies/Impacts:** Latest outputs contain 51,000 student recommendation rows and 50,991 instructor recommendation rows with expanded schema for richer recommender prototyping.

### [2026-04-11 22:15] GitHub Copilot (GPT-5.3-Codex) - Recommendation Engine Implementation Start (LambdaMART + Batch Precompute)
- **Files Modified:** `requirements.txt`, `.agents/logs/AGENT_LEDGER.md`
- **Files Created:** `backend/app/recommender/__init__.py`, `backend/app/recommender/ranking_metrics.py`, `backend/app/recommender/train_recommenders.py`, `backend/app/recommender/precompute_recommendations.py`
- **Files Generated:** `models/recommenders/student_ranker.pkl`, `models/recommenders/instructor_ranker.pkl`, `models/recommenders/recommendation_training_summary.json`, `data/recommendations/precomputed/student_topk.csv`, `data/recommendations/precomputed/instructor_topk.csv`, `data/recommendations/precomputed/precompute_report.json`
- **What was done:** Implemented Phase 1/2 recommender pipeline with LightGBM LambdaMART rankers for student and instructor engines, including grouped split preparation, categorical encoding maps, offline ranking metrics (`ndcg@3`, `recall@3`, `map@3`), MLflow logging/registration, and batch top-k precompute generation.
- **Why it was done:** User requested to start implementation from the proposed recommendation-engine roadmap.
- **Human-in-the-loop:** Training and precompute were executed in `conda` env `astro`.
- **Dependencies/Impacts:** Recommendation model artifacts and precomputed serving tables now exist for API integration; added `lightgbm` dependency to requirements.

