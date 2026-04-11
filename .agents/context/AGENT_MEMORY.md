# Agent Memory & Active State Tracking

> **Purpose:** This file acts as the active context boundary (memory) for cross-IDE agents. It tracks the current focus, major architectural decisions, and open problems. Agents must READ this file upon initialization and UPDATE it when changing major contexts.

## Current Sprint Goal
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

---
*(Agents: Update the `Active Context` and `Open Problems` sections as you progress. Ensure changes are atomic and narrative.)*
