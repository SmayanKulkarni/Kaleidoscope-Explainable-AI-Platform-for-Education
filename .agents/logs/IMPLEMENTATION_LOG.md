# Implementation Log

## 2026-04-11 21:08 - GitHub Copilot (GPT-5.3-Codex)
- Added Groq-powered generator at scripts/generate_synthetic_recommendation_data.py.
- Generates separate student/instructor recommendation datasets.
- Student dataset includes synthetic implicit feedback features and blended recommendation scoring.
- Executed with conda env astro and produced CSV outputs in data/synthetic/recommendations/.

## 2026-04-11 21:18 - GitHub Copilot (GPT-5.3-Codex)
- Regenerated larger datasets using astro env (`--n-students 7000 --n-instructors 220 --top-k 3`).
- Output scale: 21,000 student recommendation rows and 20,994 instructor recommendation rows.
- Enhanced student scoring to explicitly blend additional main dropout-dataset features.
- Added output `base_*` columns and `learning_momentum` to student recommendation CSV.

## 2026-04-11 21:28 - GitHub Copilot (GPT-5.3-Codex)
- Expanded implicit feedback generation with dwell-time/search/save/recency signals.
- Added explicit student profile features and numeric profile scores to student recommendation output.
- Regenerated large dataset with astro env (`--n-students 17000 --n-instructors 350 --top-k 3`).
- Final scale: 51,000 student rows and 50,991 instructor rows.

## 2026-04-11 22:15 - GitHub Copilot (GPT-5.3-Codex)
- Started recommendation engine implementation with new `backend/app/recommender/` package.
- Added LightGBM LambdaMART training pipeline for student + instructor rankers with ranking metrics.
- Added batch precompute script for role-serving top-k outputs.
- Trained models and generated artifacts/precomputed tables in `astro` env.

## 2026-04-12 01:34 - GitHub Copilot (GPT-5.3-Codex)
- Investigated stale demo login failures and traced root cause to auth register returning 500 from `passlib` + `bcrypt` runtime incompatibility.
- Installed `bcrypt==4.0.1`, reseeded demo accounts via `scripts/seed_demo_accounts.py`, and verified `/auth/login` returns 200 for `demo_student`, `demo_instructor`, and `demo_admin`.
- Pinned `bcrypt==4.0.1` in `requirements.txt` to prevent recurrence in fresh environments.

## 2026-04-12 09:17 - GitHub Copilot (GPT-5.4-mini)
- Replaced the static student xai service stub with API-backed data fetching, feature history caching, and explain-response transformation.
- Added `FEATURE_METADATA`, `fetchLearnerFeatures`, `getStoredFeatures`, and `simulateWhatIf` exports to match the frontend tests and dashboard callers.
- Fixed admin enrollment contract to use `instructor_profile_id` in the UI and validated the instructor profile on the backend.
- Extended admin enrollment list responses with instructor name fields so the table can render a human-readable instructor label.

## 2026-04-12 09:21 - GitHub Copilot (GPT-5.4-mini)
- Wired the ActionPlan page to the real learner explanation query path instead of static copy.
- Added instructor fallback selection so the page can load a rostered learner when an instructor opens it.
- Installed `vitest` and `happy-dom` in the frontend workspace and verified the focused frontend contract tests pass.
