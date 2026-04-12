# Context Sync Log

## 2026-04-11 21:08 - GitHub Copilot (GPT-5.3-Codex)
- Sprint context update: MVP now includes recommendation data generation artifacts in addition to dropout workflow.
- New script: scripts/generate_synthetic_recommendation_data.py.
- New outputs: data/synthetic/recommendations/student_recommendation_dataset.csv, data/synthetic/recommendations/instructor_recommendation_dataset.csv, data/synthetic/recommendations/generation_report.json.

## 2026-04-11 21:18 - GitHub Copilot (GPT-5.3-Codex)
- Context update: recommendation datasets regenerated at ~20k scale for MVP demo readiness.
- Student recommendation generation now blends core dropout-model student features with synthetic implicit feedback.
- Student output schema expanded with `learning_momentum` and `base_*` feature columns for traceability and downstream model experimentation.

## 2026-04-11 21:28 - GitHub Copilot (GPT-5.3-Codex)
- Context update: recommendation datasets regenerated at ~50k scale.
- Student schema now includes richer implicit features (dwell time, save/search events, recommendation recency).
- Student schema now includes explicit profile fields and explicit numeric profile scores used in recommendation scoring.

## 2026-04-11 22:15 - GitHub Copilot (GPT-5.3-Codex)
- Context update: recommendation implementation has started with working LambdaMART training and batch precompute.
- New module path: `backend/app/recommender/` with training, ranking-metric, and precompute scripts.
- New artifacts: ranker pickles under `models/recommenders/` and precomputed serving files under `data/recommendations/precomputed/`.

## 2026-04-12 01:34 - GitHub Copilot (GPT-5.3-Codex)
- Context update: demo authentication failures were caused by a runtime dependency mismatch (`passlib` with incompatible `bcrypt` version), not stale frontend prefill values.
- Auth runtime was stabilized by pinning `bcrypt==4.0.1` in backend dependencies and reinstalling in `.venv`.
- Demo accounts were reseeded and verified valid against `/auth/login`, restoring out-of-the-box frontend demo login flow.

## 2026-04-12 09:17 - GitHub Copilot (GPT-5.4-mini)
- Context update: the student overview page is now intended to consume real explainability data rather than the old canned sample payload.
- The xai service now owns feature-history retrieval, explain-response normalization, and student dashboard data assembly.
- Admin enrollment now treats instructor profile id as the canonical write-path identifier, and backend responses include instructor display fields for the UI.

## 2026-04-12 09:21 - GitHub Copilot (GPT-5.4-mini)
- Context update: ActionPlan now queries the same learner explainability path and supports instructor-selected roster learners.
- Frontend test tooling is now present in the workspace (`vitest`, `happy-dom`), so the contract tests can be run locally.
- Focused frontend validation is green after the service and page rewiring.
