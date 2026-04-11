# Human In The Loop Log

## 2026-04-11 21:08 - GitHub Copilot (GPT-5.3-Codex)
- User request applied: use conda astro env for execution.
- Command run under astro:
  conda run -n astro python scripts/generate_synthetic_recommendation_data.py --n-students 3000 --n-instructors 150 --top-k 3
- Outcome: generation completed successfully.

## 2026-04-11 21:18 - GitHub Copilot (GPT-5.3-Codex)
- User request applied: generate around 20k datapoints and include stronger student-data usage from dropout dataset.
- Command run under astro:
  conda run -n astro python scripts/generate_synthetic_recommendation_data.py --n-students 7000 --n-instructors 220 --top-k 3
- Outcome: generation completed with 21,000 student rows and 20,994 instructor rows.

## 2026-04-11 21:28 - GitHub Copilot (GPT-5.3-Codex)
- User request applied: add more explicit/implicit features and generate around 50k rows.
- Command run under astro:
  conda run -n astro python scripts/generate_synthetic_recommendation_data.py --n-students 17000 --n-instructors 350 --top-k 3
- Outcome: generation completed with 51,000 student rows and 50,991 instructor rows.

## 2026-04-11 22:15 - GitHub Copilot (GPT-5.3-Codex)
- User request applied: start implementation of recommendation engines.
- Commands run under astro:
  conda run -n astro python backend/app/recommender/train_recommenders.py
  conda run -n astro python backend/app/recommender/precompute_recommendations.py --top-k 5
- Outcome: student/instructor rankers trained and precomputed top-k outputs generated.

## 2026-04-12 01:34 - GitHub Copilot (GPT-5.3-Codex)
- User request applied: repopulate stale dev prefill logins that were returning 401.
- Commands run:
  `"/media/smayan/500GB SSD/Datahack 4.0/.venv/bin/python" -m pip install --force-reinstall "bcrypt==4.0.1"`
  `"/media/smayan/500GB SSD/Datahack 4.0/.venv/bin/python" scripts/seed_demo_accounts.py`
  Login validation for `demo_student`, `demo_instructor`, `demo_admin` via `POST /auth/login`.
- Outcome: all three demo logins now authenticate successfully (HTTP 200).
