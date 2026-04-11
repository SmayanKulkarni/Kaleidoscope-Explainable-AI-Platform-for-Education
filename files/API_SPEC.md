# API Specification — XAI Learning Recommendation System

> Base URL: `http://localhost:8000` (dev)
>
> All responses return raw JSON objects (no envelope wrapper).
> Errors: FastAPI default `{detail: "..."}` for 4xx/5xx.

---

## Health

### `GET /health`
```json
Response 200:
{
  "status": "ok",
  "model_version": "gbm-v1",
  "gbm_loaded": true,
  "lstm_loaded": true,
  "device": "cuda",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4070 SUPER",
    "cuda_version": "12.6",
    "vram_total_mb": 12562,
    "vram_free_mb": 12562
  },
  "timestamp": "2026-04-11T10:00:00+00:00"
}
```

---

## Core Prediction

### `POST /predict`
**Query param:** `model=gbm` (default) | `model=lstm`

```json
Request:
{
  "login_frequency_weekly": 3.0,
  "avg_session_duration_min": 45.0,
  "forum_posts_count": 2,
  "video_completion_rate": 0.6,
  "quiz_avg_score": 55.0,
  "quiz_completion_rate": 0.7,
  "assignment_submission_rate": 0.65,
  "days_since_last_activity": 7,
  "prior_course_completions": 1,
  "current_week_in_course": 6,
  "missed_deadlines_count": 2,
  "help_requests_count": 1,
  "engagement_latent_1": 0.0,
  "engagement_latent_2": 0.0,
  "engagement_latent_3": 0.0
}

Response 200:
{
  "risk_score": 0.5017,
  "risk_label": "medium",
  "model_used": "gbm",
  "uncertainty": {
    "prediction": 1,
    "risk_score": 0.5017,
    "prediction_set": [0],
    "confidence_width": 0.5,
    "uncertainty_label": "Low"
  }
}
```

**risk_label:** `"low"` (< 0.4) | `"medium"` (0.4–0.7) | `"high"` (≥ 0.7)

---

## Full Explanation

### `POST /explain`

```json
Request:
{
  "features": { /* same as /predict body */ },
  "learner_id": "learner_042",
  "model": "gbm",
  "audience": "both"
}
```

**audience:** `"learner"` | `"instructor"` | `"both"` (default)

```json
Response 200:
{
  "risk_score": 0.5017,
  "risk_label": "medium",
  "shap_values": {
    "login_frequency_weekly": 0.374,
    "assignment_submission_rate": -0.291,
    "current_week_in_course": 2.204,
    "..."
  },
  "base_value": -2.315,
  "top_features": [
    { "name": "current_week_in_course", "shap": 2.204, "direction": "risk" },
    { "name": "login_frequency_weekly", "shap": 0.374, "direction": "risk" },
    { "name": "assignment_submission_rate", "shap": -0.291, "direction": "protective" }
  ],
  "stability": 0.9479,
  "interactions": [
    {
      "feature_a": "current_week_in_course",
      "feature_b": "help_requests_count",
      "interaction_score": -0.558,
      "direction": "dampening"
    }
  ],
  "interaction_narrative": {
    "interaction_summary": "...",
    "top_interaction": "...",
    "n_interactions": 5
  },
  "anchor_rule": {
    "anchor_rule": "IF current_week_in_course > 4 AND assignment_submission_rate <= 0.7 THEN dropout_risk = HIGH",
    "precision": 0.91,
    "coverage": 0.18,
    "human_readable": "..."
  },
  "prototypes": {
    "matches": [
      {
        "learner_id": "learner_218",
        "outcome": "dropped_out",
        "similarity_score": 0.89,
        "distinguishing_features": ["assignment_submission_rate", "days_since_last_activity"]
      }
    ],
    "narrative": "Your profile closely resembles learner_218 who dropped out..."
  },
  "counterfactual": {
    "actions": [...],
    "counterfactuals": [...],
    "best_cf": { "...": "..." },
    "changed_features": { "assignment_submission_rate": {"from": 0.65, "to": 0.85} },
    "n_generated": 3
  },
  "ranked_actions": [
    {
      "rank": 1,
      "feature": "assignment_submission_rate",
      "current_value": 0.65,
      "target_value": 0.85,
      "plain_language": "Submit 85% of assignments (currently at 65%)",
      "priority_score": 0.72,
      "estimated_impact": 0.18,
      "is_causal": true,
      "causal_badge": "Causal"
    }
  ],
  "causal_annotations": [
    {
      "feature": "assignment_submission_rate",
      "shap_value": -0.291,
      "shap_direction": "protective",
      "causal_type": "causal",
      "ate": -0.14,
      "is_causal": true,
      "note": "DoWhy confirms causal effect (ATE=-0.1400). SHAP and causal direction agree."
    }
  ],
  "trust_score": {
    "trust_score": 0.81,
    "fidelity": 0.94,
    "stability": 0.9479,
    "completeness": 0.71,
    "label": "High"
  },
  "uncertainty": {
    "prediction": 1,
    "risk_score": 0.5017,
    "prediction_set": [0],
    "confidence_width": 0.5,
    "uncertainty_label": "Low"
  },
  "explanation_drift": {
    "drift_detected": false,
    "jsd": 0.04,
    "rank_shift": 0,
    "severity": "none",
    "flag": null
  },
  "narratives": {
    "learner": "You're on track but falling behind on assignments...",
    "instructor": "Learner 042 is MEDIUM risk (0.50). Top factor: current_week_in_course..."
  }
}
```

> **LSTM temporal attributions** (only returned when `model=lstm`):
> `temporal_attributions`: `{week_importances, feature_importances, most_important_week}`

---

## What-If Scenario

### `POST /whatif`
```json
Request:
{
  "features": { /* LearnerFeatures */ },
  "overrides": { "quiz_avg_score": 80.0, "days_since_last_activity": 2 }
}

Response 200:
{
  "shap_values": { "...": 0.0 },
  "base_value": -2.315,
  "risk_score": 0.31,
  "risk_delta": -0.19,
  "top_features": [...]
}
```

---

## Counterfactual

### `POST /counterfactual`
```json
Request: { /* LearnerFeatures flat object */ }

Response 200:
{
  "actions": [...],
  "counterfactuals": [...],
  "best_cf": { "...": "..." },
  "changed_features": { "assignment_submission_rate": {"from": 0.65, "to": 0.85} },
  "n_generated": 3,
  "ranked_actions": [
    {
      "rank": 1,
      "feature": "assignment_submission_rate",
      "current_value": 0.65,
      "target_value": 0.85,
      "plain_language": "Submit 85% of assignments (currently at 65%)",
      "priority_score": 0.72,
      "estimated_impact": 0.18,
      "is_causal": true,
      "causal_badge": "Causal"
    }
  ]
}
```

---

## Monte Carlo Simulation

### `POST /simulate`
```json
Request:
{
  "features": { /* LearnerFeatures */ },
  "current_week": 6,
  "target_week": 12,
  "n_simulations": 1000
}

Response 200 (transitions.pkl available):
{
  "outcome_distribution": { "mean_risk": 0.62, "std_risk": 0.09, "p_dropout": 0.71 },
  "feature_distributions": { "assignment_submission_rate": {"mean": 0.61, "std": 0.07} },
  "risk_percentiles": { "p10": 0.45, "p50": 0.63, "p90": 0.78 },
  "current_week": 6,
  "target_week": 12,
  "n_simulations": 1000
}

Response 200 (transitions.pkl missing):
{
  "feature_distributions": {},
  "outcome_distribution": {},
  "current_week": 6,
  "target_week": 12,
  "n_simulations": 1000,
  "message": "transitions.pkl not found — run temporal_builder.py first"
}
```

---

## History

### `GET /history/{learner_id}`
```json
Response 200:
{
  "learner_id": "learner_042",
  "timeline": [
    {
      "timestamp": "2026-04-11T10:19:53",
      "risk_score": 0.50,
      "top3_features": [
        { "name": "current_week_in_course", "shap": 2.204, "direction": "risk" },
        { "name": "login_frequency_weekly", "shap": 0.374, "direction": "risk" },
        { "name": "assignment_submission_rate", "shap": -0.291, "direction": "protective" }
      ]
    }
  ],
  "history": [ { "timestamp": "...", "risk_score": 0.50, "model_used": "gbm" } ],
  "drift_flags": []
}
```

---

## Feedback

### `POST /feedback`
```json
Request:
{
  "learner_id": "learner_042",
  "explanation_id": "exp_a3f9c2",
  "rating": 4,
  "followed_recommendation": true,
  "top_action_feature": "assignment_submission_rate",
  "correction_feature": "days_since_last_activity",
  "correction_comment": "I was on a planned break",
  "audience": "learner"
}

Response 200:
{ "recorded": true, "feedback_id": "fb_x9q1", "learner_id": "learner_042" }
```

### `GET /feedback/stats`
```json
Response 200:
{
  "total_records": 142,
  "avg_rating": 3.8,
  "recommendation_follow_rate": 0.61,
  "top_corrected_features": ["days_since_last_activity", "help_requests_count"],
  "high_trust_follow_rate": 0.74,
  "low_trust_follow_rate": 0.38
}
```

### `GET /feedback/{learner_id}`
```json
Response 200:
{
  "learner_id": "learner_042",
  "records": [{ "timestamp": "...", "rating": 4, "followed_recommendation": true }]
}
```

---

## Interaction Events (Implicit Feedback)

### `POST /events`
```json
Request:
{
  "events": [
    {
      "learner_id": "learner_042",
      "event_type": "whatif_slider",
      "feature": "quiz_avg_score",
      "value": 75.0,
      "timestamp": "2026-04-11T10:00:00Z"
    }
  ]
}

Response 200:
{ "recorded": 1, "message": "ok" }
```

### `GET /events/{learner_id}`
```json
Response 200:
{
  "learner_id": "learner_042",
  "count": 12,
  "events": [{ "event_type": "whatif_slider", "feature": "quiz_avg_score", "value": 75.0 }]
}
```

---

## MLOps

### `GET /mlops/health`
```json
Response 200:
{
  "status": "ok",
  "model_version": "gbm-v1",
  "drift_status": "ok",
  "prediction_count": 1247,
  "timestamp": "2026-04-11T10:00:00+00:00"
}
```

### `GET /mlops/metrics`
```json
Response 200:
{
  "training": {
    "gbm": {
      "test_auc_roc": 0.9427, "test_f1": 0.8035, "test_brier_score": 0.0873
    },
    "rf": {
      "test_auc_roc": 0.9425, "test_f1": 0.7851, "test_brier_score": 0.0902
    }
  },
  "model_version": "gbm-v1"
}
```

### `GET /mlops/drift-report`
```json
Response 200 (report available):
{
  "dataset_drift": false,
  "n_drifted_features": 1,
  "n_total_features": 12,
  "drift_share": 0.083,
  "feature_drifts": [
    { "feature": "days_since_last_activity", "drift_detected": true, "drift_score": 0.21 }
  ]
}

Response 200 (not enough data):
{
  "dataset_drift": false,
  "n_drifted_features": 0,
  "message": "Not enough predictions logged yet"
}
```

### `POST /mlops/retrain`
Triggers full retraining pipeline from accumulated events + feedback.
```json
Response 200:
{ "success": true, "message": "Retrain complete", "new_version": "gbm-v2" }
```

### `POST /mlops/reload`
Hot-swap live model (call after retrain succeeds).
```json
Response 200:
{ "success": true, "reloaded_components": ["gbm", "shap", "anchors", "dice", "uncertainty"] }

Response 500 (reload failed):
{ "detail": { "success": false, "error": "SHAP explainer init failed: ..." } }
```

---

## Auth

### `POST /auth/register`
```json
Request: { "username": "learner_042", "password": "secret", "role": "student" }
Response 200: { "id": 1, "username": "learner_042", "role": "student" }
```

### `POST /auth/token`
```json
Request (form data): username=learner_042&password=secret
Response 200: { "access_token": "eyJ...", "token_type": "bearer" }
```

---

## Authenticated Student Endpoints

All require `Authorization: Bearer <token>`.

### `POST /explain/me`
Returns latest explanation summary from the learner's own profile.

### `GET /explain/me/history`
Returns explanation timeline for the authenticated learner.

---

## Error Responses

```json
// 422 Validation Error
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "quiz_completion_rate"],
      "msg": "Input should be less than or equal to 1"
    }
  ]
}

// 503 Model Not Loaded
{ "detail": "Model not loaded — run trainer.py first" }

// 503 Module Not Initialised
{ "detail": "DiCE explainer not initialised" }

// 404 Not Found
{ "detail": "No explanation found for learner_999" }
```
