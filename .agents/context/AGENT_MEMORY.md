# Agent Memory & Active State Tracking

> **Purpose:** This file acts as the active context boundary (memory) for cross-IDE agents. It tracks the current focus, major architectural decisions, and open problems. Agents must READ this file upon initialization and UPDATE it when changing major contexts.

## Current Sprint Goal
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
- **Phase 1-3:** Build MVP Backend for XAI Recommendation Engine. (Refer to `LLM_WIKI.md` and `docs/IMPLEMENTATION_PLAN.md`).

## Active Context
- **Status:** Initialized project.
- **Next Steps Required:**
  1. Generate synthetic data using `backend/app/model/mock_data.py`.
  2. Implement Model Trainer (`backend/app/model/trainer.py`).
  3. Wire up FastAPI base and predictions (`backend/app/main.py`).

## Architectural Constraints (Strict)
- **Do not use LLMs for prediction or explanation generation.** LLM is strictly for narration.
- **Maintain Actionability:** Immutable features must remain locked in DiCE.
- **Trust Scores:** Every explanation generation MUST be followed by the `TrustScorer` metric updates.

## Open Problems / Blockers
- None at this time. Wait for data generation.
=======
=======
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60
=======
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60
- **Phase 1-3:** Build MVP Backend for XAI Recommendation Engine. (Refer to `LLM_WIKI.md` and `files/IMPLEMENTATION_PLAN.md`).

## Active Context
- **Status:** Data pipeline + temporal pipeline complete. Implementation plan finalized with trimmed XAI stack.
- **Scope Update:** Frontend implementation is now owned by a separate team; this code stream is backend/data/ML + API-contract handoff only.
- **Repo Update:** `models/` is now tracked in Git (no longer ignored) so trained artifacts can be pushed to GitHub.
- **Data (static):** `data/learners.csv` (32,593 rows), `data/train.pkl` (26,074), `data/test.pkl` (6,519). Dropout rate 31.2%. Zero NaNs.
- **Data (temporal):** `data/temporal/snapshots.pkl` (195,558 rows = 32,593 students × 6 snapshots @ weeks 2,4,6,8,10,12). `data/temporal/transitions.pkl` (162,965 transition rows across 5 week-pairs for Monte Carlo).
- **Feature source:** `backend/app/model/data_loader.py` (static), `backend/app/model/temporal_builder.py` (temporal + MC simulator).
- **Note:** Raw OULAD CSVs re-downloaded (git-lfs pointers replaced with actual data). After any `git pull`, run `git lfs pull` or re-run `data_loader.py` + `temporal_builder.py`.
- **Code graph:** `code-review-graph` installed for Windsurf. Initial graph built successfully for this repo: 22 nodes, 205 edges, 4 Python files indexed. MCP config written to `/home/smayan/.codeium/windsurf/mcp_config.json`.

### Models (Dual Architecture)
| Model | Type | Input | XAI | File |
|-------|------|-------|-----|------|
| GBM (primary) | Static | 12 features | TreeSHAP | `backend/app/model/trainer.py` |
| Random Forest | Static (comparison) | 12 features | TreeSHAP | `backend/app/model/trainer.py` |
| LSTM | Temporal | 6×12 sequence | DeepSHAP + Captum IG | `backend/app/model/lstm_trainer.py` |

### XAI Stack (8 explanation + 3 tracking/infra, zero redundancy)
| Module | File | Status |
|--------|------|--------|
| TreeSHAP (GBM + What-If) | `backend/app/explainers/shap_explainer.py` | Pending |
| DeepSHAP (LSTM) | `backend/app/explainers/shap_explainer.py` | Pending |
| Integrated Gradients (LSTM) | `backend/app/explainers/dl_explainer.py` | Pending |
| Archipelago (interactions) | `backend/app/explainers/archipelago.py` | Pending |
| Anchors (IF-THEN rules) | `backend/app/explainers/anchors_explainer.py` | Pending |
| DiCE (counterfactuals) | `backend/app/explainers/dice_explainer.py` | Pending |
| Prototypes (k-NN) | `backend/app/explainers/prototype_explainer.py` | Pending |
| MAPIE (uncertainty) | `backend/app/evaluator/uncertainty_estimator.py` | Pending |
| Causal Annotation (DAG) | `backend/app/causal/causal_annotator.py` | Pending |
| TrustScorer | `backend/app/evaluator/trust_scorer.py` | Pending |
| Drift Detector (JSD + Evidently) | `backend/app/tracker/drift_detector.py` + `backend/app/mlops/drift_monitor.py` | Pending |

### MLOps Stack
| Component | Tool | File |
|-----------|------|------|
| Experiment Tracking | MLflow | `backend/app/mlops/experiment_tracker.py` |
| Model Registry | MLflow | `backend/app/mlops/model_registry.py` |
| Data/Prediction Drift | Evidently AI | `backend/app/mlops/drift_monitor.py` |
| Prediction Logging | SQLite | `backend/app/mlops/prediction_logger.py` |
| Model Card | Auto-generated | `backend/app/mlops/model_card.py` |

### Rust MC Simulator
- **Dir:** `mc_simulator/` (PyO3 + maturin + Rayon)
- **Status:** Pending (Python prototype done in `temporal_builder.py`)

**Removed (redundant):** FastSHAP, CEM.

- **Next Steps:**
  1. **Task 1.1a** — GBM Trainer + MLflow logging (`backend/app/model/trainer.py`)
  2. **Task 1.1b** — LSTM Trainer (`backend/app/model/lstm_trainer.py`)
  3. **Task 1.2** — Dual SHAP Explainer (TreeSHAP + DeepSHAP)
  4. **Task 1.3** — FastAPI base + `/predict` endpoint
  5. **Task 4.x** — API contract freeze + frontend handoff fixtures (`files/API_SPEC.md`, `data/`)
  6. **Task M.1** — MLflow experiment tracking setup

## Architectural Constraints (Strict)
- **Do not use LLMs for prediction or explanation generation.** LLM is strictly for narration of pre-computed XAI payloads.
- **Maintain Actionability:** Immutable features must remain locked in DiCE.
- **Trust Scores:** Every explanation generation MUST be followed by `TrustScorer` metric updates.
- **Feature order is canonical:** Always use `FEATURE_COLUMNS` list from `data_loader.py`.
- **No FastSHAP:** TreeSHAP handles both stored explanations and What-If (<30ms on 12 features).
- **No CEM:** DiCE counterfactuals + SHAP top-3 cover the same diagnostic ground.
- **MLflow required:** Every training run must be logged. Production model always from registry.
- **Rust MC preferred:** Once built, `mc_simulator` replaces Python `MonteCarloSimulator` for `/simulate` endpoint.

## Open Problems / Blockers
- `forum_posts_count`, `video_completion_rate`, `avg_session_duration_min`, `help_requests_count` are VLE-approximated / synthetically augmented. Document in model card (auto-generated via Task M.5).
- LSTM requires PyTorch — adds ~2GB to Docker image. Consider CPU-only build.
- Rust MC requires Rust toolchain for build. Provide pre-built wheel or fallback to Python.
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60
=======
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60
=======
>>>>>>> 741a24ed89c99b98a036f0d03c34830ee3530d60

---
*(Agents: Update the `Active Context` and `Open Problems` sections as you progress. Ensure changes are atomic and narrative.)*
