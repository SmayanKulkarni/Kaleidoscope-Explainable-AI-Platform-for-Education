/**
 * Canonical event types for implicit data collection.
 * Used by useEventTracker / eventTracker.js → POST /events → EventStore.
 * The MLOps pipeline aggregates these for engagement signals during retraining.
 */

// ── Session lifecycle ──────────────────────────────────────────────────────
export const SESSION_START = 'session_start';
export const SESSION_END = 'session_end';

// ── Navigation ─────────────────────────────────────────────────────────────
export const PAGE_VIEW = 'page_view';

// ── Student dashboard ──────────────────────────────────────────────────────
export const RECOMMENDATION_VIEWED = 'recommendation_viewed';
export const RECOMMENDATION_CLICKED = 'recommendation_clicked';
export const EXPLAIN_REQUESTED = 'explain_requested';

// ── What-If / Simulate ─────────────────────────────────────────────────────
export const WHATIF_RUN = 'whatif_run';
export const SIMULATE_RUN = 'simulate_run';

// ── Action Plan ────────────────────────────────────────────────────────────
export const ACTION_VIEWED = 'action_viewed';
export const ACTION_TAKEN = 'action_taken';

// ── History / Compare ──────────────────────────────────────────────────────
export const HISTORY_VIEWED = 'history_viewed';
export const COMPARE_RUN = 'compare_run';

// ── Instructor ─────────────────────────────────────────────────────────────
export const INSTRUCTOR_STUDENT_SELECTED = 'instructor_student_selected';
export const INSTRUCTOR_RECO_VIEWED = 'instructor_reco_viewed';
export const INSTRUCTOR_WHATIF_RUN = 'instructor_whatif_run';

// ── Feedback ───────────────────────────────────────────────────────────────
export const FEEDBACK_SUBMITTED = 'feedback_submitted';
