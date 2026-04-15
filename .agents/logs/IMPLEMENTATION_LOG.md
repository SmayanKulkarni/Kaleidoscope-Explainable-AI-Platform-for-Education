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

## 2026-04-12 09:44 - GitHub Copilot (GPT-5.3-Codex)
- Added reusable recommender trainer API (`train_recommenders`) so MLOps can trigger RS retraining programmatically.
- Added `backend/app/recommender/retrain_pipeline.py` to retrain student/instructor rankers using learned implicit engagement features merged by `learner_id`.
- Added new unified backend control-plane endpoints: `POST /mlops/retrain-all` and `POST /mlops/reload-all`.
- Added runtime recommender hot-reload helper to refresh both ranker explainers and reference pools without process restart.
- Updated `.github/workflows/retrain.yml` to call the new unified retrain/reload endpoints.

## 2026-04-12 10:07 - GitHub Copilot (GPT-5.3-Codex)
- Added snapshot-backed student feature endpoint at `GET /student/features/{learner_id}` so learner-specific dashboard data is no longer dependent on shared defaults.
- Updated `/history/{learner_id}` and `/explain/me/history` to fall back to snapshot-backed records when explanation history is empty.
- Rewired `StudentDashboard` and `StudentView` to fetch learner-specific features/explanations from the active learner ID instead of `DEFAULT_FEATURES`.
- Added admin identity explorer controls for selecting both students and instructors from seeded data.
- Added `scripts/seed_oulad_users.py` to seed a medium OULAD cohort with student/instructor accounts, snapshots, and history records plus exportable credentials.

## 2026-04-12 10:24 - GitHub Copilot (GPT-5.4-mini)
- Executed `scripts/seed_oulad_users.py --cohort-size 50` against the PostgreSQL-backed auth store.
- Seed completed successfully with 50 student accounts and 5 instructor accounts.
- Verified the exported credentials file contains 55 records total and the seeded learner snapshots differ across learners.

## 2026-04-12 10:31 - GitHub Copilot (GPT-5.4-mini)
- Ran authenticated smoke tests against `http://127.0.0.1:8000/auth/login` and `/auth/me` for three seeded students and one seeded instructor.
- All four accounts returned HTTP 200 for login and profile retrieval.
- Confirmed students receive learner profiles plus learner IDs, while the instructor receives an instructor profile and no learner profile.

## 2026-04-12 10:39 - GitHub Copilot (GPT-5.4-mini)
- Extended `scripts/seed_oulad_users.py` to support `--snapshot-week` and additive credential merging.
- Seeded 10 additional students from snapshot week 6 using `--cohort-size 10 --snapshot-week 6 --sample-offset 50`.
- Verified the combined credential export now contains 65 records total, including 10 week-6 students, and the new week-6 logins authenticate successfully.

## 2026-04-12 10:47 - GitHub Copilot (GPT-5.4-mini)
- Normalized the StudentDashboard explanation bars so widths are relative to the strongest signal in the active tab instead of using raw values.
- Added fallback rendering for Feature and Concept tabs when the backend returns no causal annotations or interaction terms.
- Validated the change with `src/tests/xaiService.test.js` and `src/tests/transforms.test.js` (32/32 passing).

## 2026-04-12 10:51 - GitHub Copilot (GPT-5.4-mini)
- Hardened the StudentDashboard Concept tab so it synthesizes a visible fallback item instead of rendering an empty panel when interaction data is missing.
- Kept the explanation tab tests green after the concept fallback change.

## 2026-04-12 10:58 - GitHub Copilot (GPT-5.4-mini)
- Added backend synthesis for concept interactions in `/explain` so interaction pairs are emitted even when Archipelago is unavailable.
- Updated the frontend explanation transformer to read interaction payloads shaped as `feature_a`/`feature_b` with `interaction_score`.
- Verified `/explain` now returns 5 interaction pairs for a seeded learner and the frontend explanation tests pass (33/33).

## 2026-04-12 11:04 - GitHub Copilot (GPT-5.4-mini)
- Humanized Concept tab labels in the frontend transformer so interaction pair names render in title case instead of raw snake_case identifiers.
- Kept the explanation contract tests green after the label update.

## 2026-04-12 11:12 - GitHub Copilot (GPT-5.4-mini)
- Updated the What-If simulator panel to derive delta from explicit `base_risk` and `new_risk` values when available.
- Increased delta display precision so small real changes do not collapse into a misleading `0% no change` badge.
- Revalidated the focused frontend contract tests after the UI formatting fix.

## 2026-04-12 11:20 - GitHub Copilot (GPT-5.3-Codex)
- Renamed frontend product branding from LearnLens to Kaliedoscope.
- Updated visible brand labels in navbar, login, register, and HTML page title.
- Verified no remaining `LearnLens` references in frontend source paths.

## 2026-04-15 20:32 - GitHub Copilot (GPT-5.3-Codex)
- Implemented S3 sync support for precomputed recommendation CSV artifacts in `backend/app/model/s3_loader.py`.
- Added `_PRECOMPUTED_FILES` mapping and integrated download/upload logic for `student_topk.csv` and `instructor_topk.csv` under `models/precomputed/` on S3.
- Updated `Makefile` MLOps targets so `make models-upload` and `make models-download` also sync `data/recommendations/precomputed/`.
- Verified backend module syntax with `python -m py_compile backend/app/model/s3_loader.py` in project `.venv`.

## 2026-04-15 23:30 - GitHub Copilot (GPT-5.3-Codex)
- Executed Phase B cloud rollout operationally on AWS: Terraform infra apply, model artifact upload to S3, and EC2 container deployment retries.
- Updated `Dockerfile` deployment path for EC2 constraints: removed frontend build stage, switched Docker dependency resolution to CPU torch wheels, and added explicit `psycopg` installation.
- Completed live deployment using the EC2 runtime image with corrected runtime env wiring (`DATABASE_URL` credentials/db name + SSL and `AWS_S3_BUCKET` export).
- Verified production endpoints on `http://3.110.142.118:8000`: `/health` returned `status=ok` with loaded models, and `/predict` returned a valid risk response payload.
