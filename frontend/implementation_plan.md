# Frontend to Backend API Mapping Plan

This document outlines the strategy for connecting our React frontend prototyping environment (`xaiService.js`) to the live FastAPI Python backend (`main.py`).

**Core Constraint**: The backend architecture (schemas, endpoints, explainers) must remain completely untouched. All mapping, translation, and adaptation must occur within the frontend code.

## User Review Required

> [!WARNING]
> Because there is no persistent database storing multiple students yet, we will map our client-side "Student" and "Instructor" logins to a **static, hardcoded baseline `LearnerFeatures` configuration** in the frontend. If the user uses the sliders in the "What-If" Explorer, we will track the deltas and send them to the backend dynamically. 

## Proposed Changes

We will refactor the frontend codebase to use `axios` and completely rewrite `xaiService.js` to act as an "Adapter Layer" that translates UI requirements into the Pydantic schemas expected by FastAPI.

### [Component: Networking]

#### [NEW] `src/config.js`
- Create a configuration file exporting the backend base URL (e.g., `http://localhost:8000`).

#### [MODIFY] `package.json`
- Install `axios` as a dependency for making HTTP requests.

### [Component: API Adapter (`xaiService.js`)]

#### [MODIFY] `src/services/xaiService.js`

We will replace the file entirely and establish the following translations:

1. **Base Features Mapping (`LearnerFeatures`)**:
   - Create a static dictionary mapping our frontend UI labels (e.g., "Weekly Study Hours") to the exact backend schema parameters (`login_frequency_weekly`, `video_completion_rate`, etc.).
   
2. **`getStudentData(...)` → mapped to `POST /explain`**:
   - Send `POST /explain` with `audience: "learner"`.
   - **Risk Mapping**: Return `resp.risk_score` and `resp.risk_label`.
   - **Feature Mapping**: Iterate over `resp.top_features` and fetch matching metadata from `resp.shap_values`. Convert them to `{ name, value, direction: value > 0 ? "increases" : "decreases" }`.
   - **Narrative Mapping**: Return `resp.narratives.markdown` as `aiNarrative`.
   - **Lookalikes**: Extract from `resp.prototypes.nearest_neighbors` if available.

3. **`getInstructorData(...)` → mapped to `POST /explain`**:
   - Send `POST /explain` with `audience: "instructor"`.
   - Extract `resp.trust_score`, `resp.stability`, and `resp.anchor_rule`.
   - Pull `resp.interaction_narrative` provided by Archipelago.

4. **`simulateWhatIf(...)` → mapped to `POST /whatif`**:
   - Capture frontend slider inputs (deltas).
   - Convert the UI slider names to backend dictionary keys (e.g., `Sliders['Assignments Completed']` -> `assignment_submission_rate: 0.8`).
   - Send them via `WhatIfRequest` overrides list.
   - Return `newRisk: resp.risk_score` and `impactFactors: resp.top_features`.

5. **`getActionPlan(...)` → mapped to `POST /counterfactual`**:
   - Make a call to `/counterfactual`.
   - Parse `resp.ranked_actions`.
   - Map keys (e.g., `feature`, `target`, `recommendation`, `riskReduction` based on DiCE/Causal Ranker output) into the list of actionable objectives expected by `ActionPlan.jsx`.

## Open Questions

> [!IMPORTANT]
> The backend models expect precise ranges for parameters. For instance, `video_completion_rate` must be a float between `0.0` and `1.0`. The current frontend "What-If" mock has sliders moving arbitrary scales (e.g., hours and post counts). I will manually configure step sizes and ranges for these sliders to match the exact `0` to `1.0` boundaries requested by the backend. Is this acceptable?

## Verification Plan

### Automated Tests
1. Verify that `npm run dev` compiles successfully.

### Manual Verification
1. Boot the FastAPI backend locally using Uvicorn (`python -m backend.app.main` or via Makefile).
2. Boot the frontend using `npm run dev`.
3. Login as a student and observe actual AI narratives generated (via Groq/LLM if configured, otherwise fallback text), and ensure SHAP lists accurate parameters.
4. Execute the "What-If" sliders; verify the backend logs receive `POST /whatif` and the UI updates precisely. 
5. Login as an instructor and verify the Trust Score, ML Drift indicators, and Anchor Rules appear correctly mapped from the JSON payload.
