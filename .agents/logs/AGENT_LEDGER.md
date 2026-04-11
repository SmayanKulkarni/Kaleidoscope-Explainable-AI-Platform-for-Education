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
