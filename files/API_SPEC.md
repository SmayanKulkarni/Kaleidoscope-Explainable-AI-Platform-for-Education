# API Specification — XAI Learning System

> All endpoints follow the envelope pattern: `{status, data, meta}`
> Base URL: `http://localhost:8000` (dev) | `https://your-deployment.com` (prod)

---

## Health

### `GET /health`
```json
Response 200:
{
  "status": "ok",
  "model_version": "gbm-v1.0",
  "model_accuracy": 0.91,
  "timestamp": "2025-04-11T10:00:00Z"
}
```

---

## Core Prediction + Explanation

### `POST /predict`
**Purpose:** Predict dropout risk for a learner  
**Used by:** Dashboard on load, any feature change

```json
Request:
{
  "learner_id": "learner_042",
  "features": {
    "login_frequency_weekly": 2.0,
    "avg_session_duration_min": 18.0,
    "forum_posts_count": 1,
    "video_completion_rate": 0.45,
    "quiz_avg_score": 52.0,
    "quiz_completion_rate": 0.35,
    "assignment_submission_rate": 0.4,
    "days_since_last_activity": 9,
    "prior_course_completions": 1,
    "current_week_in_course": 5,
    "missed_deadlines_count": 3,
    "help_requests_count": 0
  }
}

Response 200:
{
  "status": "ok",
  "data": {
    "learner_id": "learner_042",
    "risk_label": "HIGH",
    "risk_score": 0.78,
    "recommended_topic": "Module 5: Algebra Fundamentals",
    "model_confidence": 0.82,
    "uncertainty": 0.14,
    "uncertainty_label": "Medium",
    "timestamp": "2025-04-11T10:00:00Z"
  }
}
```

---

### `POST /explain`
**Purpose:** Full multi-layer explanation for a prediction  
**Used by:** Explanation card, learner/instructor views

```json
Request:
{
  "learner_id": "learner_042",
  "features": { /* same as /predict */ },
  "audience": "learner"  // "learner" | "instructor" | "both"
}

Response 200:
{
  "status": "ok",
  "data": {
    "prediction": { /* same as /predict response data */ },

    "feature_level": {
      "shap_values": {
        "assignment_submission_rate": -0.31,
        "days_since_last_activity": 0.22,
        "quiz_completion_rate": -0.18,
        "quiz_avg_score": -0.12,
        "login_frequency_weekly": -0.09,
        "missed_deadlines_count": 0.08,
        "forum_posts_count": -0.04,
        "video_completion_rate": -0.03,
        "help_requests_count": -0.01,
        "avg_session_duration_min": -0.01,
        "prior_course_completions": 0.0,
        "current_week_in_course": 0.0
      },
      "base_value": 0.28,
      "top3_features": [
        "assignment_submission_rate",
        "days_since_last_activity",
        "quiz_completion_rate"
      ]
    },

    "concept_level": {
      "concept_scores": {
        "engagement": 0.31,
        "assessment_performance": 0.55,
        "consistency": 0.42
      },
      "prototype_match": {
        "prototypes": [
          {
            "learner_id": "learner_218",
            "outcome": "dropped_out",
            "similarity_score": 0.89,
            "distinguishing_features": ["assignment_submission_rate", "days_since_last_activity"]
          },
          {
            "learner_id": "learner_455",
            "outcome": "completed",
            "similarity_score": 0.76,
            "distinguishing_features": ["quiz_completion_rate", "login_frequency_weekly"]
          }
        ],
        "narrative": "Your profile is similar to learner 455 who completed this course after improving quiz completion."
      },
      "top_interaction": {
        "feature_a": "quiz_completion_rate",
        "feature_b": "days_since_last_activity",
        "interaction_score": 0.14,
        "direction": "amplifying",
        "narrative": "Low quiz completion combined with inactivity amplifies dropout risk beyond either factor alone."
      }
    },

    "prediction_level": {
      "global_feature_importance": {
        "assignment_submission_rate": 0.28,
        "quiz_completion_rate": 0.22,
        "days_since_last_activity": 0.18
      },
      "anchor_rule": "IF assignment_submission_rate <= 0.4 AND days_since_last_activity > 7 THEN dropout_risk = HIGH",
      "anchor_precision": 0.93,
      "anchor_coverage": 0.21
    },

    "diagnosis": {
      "pertinent_positives": {
        "assignment_submission_rate": 0.4,
        "days_since_last_activity": 9
      },
      "pertinent_negatives": {
        "login_frequency_weekly": 4.0,
        "quiz_completion_rate": 0.65
      },
      "pp_narrative": "Your dropout risk is primarily driven by assignment submission rate and inactivity.",
      "pn_narrative": "Logging in 4x/week and completing 65% of quizzes would reduce your risk significantly."
    },

    "trust": {
      "trust_score": 0.81,
      "fidelity": 0.94,
      "stability": 0.78,
      "completeness": 0.71,
      "label": "High"
    },

    "causal_annotations": {
      "assignment_submission_rate": { "type": "causal", "note": "Directly influences dropout risk" },
      "days_since_last_activity": { "type": "causal", "note": "Directly influences dropout risk" },
      "quiz_avg_score": { "type": "correlational", "note": "Associated with risk but mediated by submission rate" }
    },

    "narratives": {
      "learner": "You're doing okay, but we noticed you haven't submitted 3 assignments and haven't logged in for 9 days. Your biggest win right now: submit your pending assignments this week — our analysis suggests this could cut your risk by ~31%.",
      "instructor": "Learner 042 is HIGH risk (score: 0.78, confidence: 0.82). Top causal factors: assignment_submission_rate (SHAP: -0.31), days_since_last_activity (SHAP: +0.22). Anchor rule holds at 93% precision. Recommend: direct outreach + assignment deadline extension."
    },

    "consistency": {
      "drift_detected": false,
      "jsd": 0.04,
      "rank_shift": 0,
      "flag": null
    }
  },
  "meta": {
    "explanation_id": "exp_a3f9c2",
    "model_version": "gbm-v1.0",
    "computation_ms": 142
  }
}
```

---

### `POST /whatif`
**Purpose:** Real-time prediction + SHAP update as user moves sliders  
**Used by:** What-If UI — called on every slider change (debounced 300ms)  
**⚠️ Must call model.predict_proba() — no heuristic**

```json
Request:
{
  "learner_id": "learner_042",
  "original_features": { /* baseline */ },
  "modified_features": {
    "login_frequency_weekly": 5.0,
    "quiz_completion_rate": 0.70,
    "days_since_last_activity": 2
    // only changed features needed; unchanged will be filled from original
  }
}

Response 200:
{
  "status": "ok",
  "data": {
    "original": {
      "risk_score": 0.78,
      "risk_label": "HIGH",
      "shap_values": { /* original SHAP */ }
    },
    "modified": {
      "risk_score": 0.31,
      "risk_label": "LOW",
      "shap_values": { /* recomputed SHAP — FastSHAP for speed */ }
    },
    "delta": {
      "risk_score_change": -0.47,
      "label_change": "HIGH → LOW",
      "biggest_impact_feature": "quiz_completion_rate"
    }
  }
}
```

---

### `POST /counterfactual`
**Purpose:** Compute minimal feature changes to flip prediction  
**Used by:** Counterfactual card, action recommendations

```json
Request:
{
  "learner_id": "learner_042",
  "features": { /* current features */ },
  "desired_outcome": "low_risk",  // "low_risk" | "medium_risk"
  "n_counterfactuals": 3
}

Response 200:
{
  "status": "ok",
  "data": {
    "original_risk_score": 0.78,
    "counterfactual_risk_score": 0.22,
    "flip_achieved": true,

    "changed_features": {
      "assignment_submission_rate": { "from": 0.4, "to": 0.75 },
      "days_since_last_activity": { "from": 9, "to": 2 },
      "quiz_completion_rate": { "from": 0.35, "to": 0.60 }
    },

    "actions": [
      {
        "rank": 1,
        "feature": "assignment_submission_rate",
        "current_value": 0.4,
        "target_value": 0.75,
        "plain_language": "Submit 75% of assignments (currently submitting 40%)",
        "estimated_impact": 0.34,
        "is_causal": true,
        "causal_badge": "🔗 Causal"
      },
      {
        "rank": 2,
        "feature": "days_since_last_activity",
        "current_value": 9,
        "target_value": 2,
        "plain_language": "Return to the platform within 2 days",
        "estimated_impact": 0.19,
        "is_causal": true,
        "causal_badge": "🔗 Causal"
      },
      {
        "rank": 3,
        "feature": "quiz_completion_rate",
        "current_value": 0.35,
        "target_value": 0.60,
        "plain_language": "Complete 60% of quizzes (currently at 35%)",
        "estimated_impact": 0.14,
        "is_causal": false,
        "causal_badge": "〰️ Correlated"
      }
    ],

    "human_readable_summary": "If you submit more assignments and log back in within 2 days, your dropout risk drops from HIGH to LOW."
  }
}
```

---

### `GET /history/{learner_id}`
**Purpose:** Explanation consistency timeline  
**Used by:** Consistency timeline chart

```json
Response 200:
{
  "status": "ok",
  "data": {
    "learner_id": "learner_042",
    "sessions": [
      {
        "session_id": "exp_a1b2c3",
        "timestamp": "2025-04-07T09:00:00Z",
        "risk_score": 0.65,
        "risk_label": "MEDIUM",
        "top3_features": ["quiz_completion_rate", "assignment_submission_rate", "login_frequency_weekly"],
        "trust_score": 0.84,
        "drift_from_previous": false
      },
      {
        "session_id": "exp_a3f9c2",
        "timestamp": "2025-04-11T10:00:00Z",
        "risk_score": 0.78,
        "risk_label": "HIGH",
        "top3_features": ["assignment_submission_rate", "days_since_last_activity", "quiz_completion_rate"],
        "trust_score": 0.81,
        "drift_from_previous": true,
        "drift_flag": "⚠️ Top feature changed: quiz_completion_rate → days_since_last_activity"
      }
    ]
  }
}
```

---

### `POST /feedback`
**Purpose:** Human-in-the-loop feedback collection

```json
Request:
{
  "learner_id": "learner_042",
  "explanation_id": "exp_a3f9c2",
  "rating": 4,
  "followed_recommendation": true,
  "correction": {
    "feature": "days_since_last_activity",
    "comment": "I was on a planned break, this shouldn't count as inactivity"
  }
}

Response 200:
{ "status": "ok", "data": { "feedback_id": "fb_x9q1" } }
```

---

### `GET /feedback/stats`
```json
Response 200:
{
  "status": "ok",
  "data": {
    "total_explanations": 142,
    "avg_rating": 3.8,
    "recommendation_follow_rate": 0.61,
    "top_corrected_features": ["days_since_last_activity", "help_requests_count"],
    "high_trust_follow_rate": 0.74,
    "low_trust_follow_rate": 0.38
  }
}
```

---

## Error Responses

```json
// 422 Validation Error
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "quiz_completion_rate must be between 0.0 and 1.0",
    "field": "quiz_completion_rate"
  }
}

// 503 Model Not Loaded
{
  "status": "error",
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "Model is loading, retry in 5 seconds"
  }
}
```

---

## FastAPI Implementation Skeleton

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncio

# Global state loaded once at startup
MODEL_STATE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything at startup
    MODEL_STATE["gbm"] = load_model("models/gbm.pkl")
    MODEL_STATE["rf"] = load_model("models/rf.pkl")
    MODEL_STATE["X_train"] = load_data("data/train.pkl")
    MODEL_STATE["shap"] = SHAPExplainer(MODEL_STATE["gbm"], MODEL_STATE["X_train"])
    MODEL_STATE["fastshap"] = FastSHAPExplainer(MODEL_STATE["gbm"], MODEL_STATE["X_train"])
    MODEL_STATE["dice"] = DiCEExplainer(MODEL_STATE["gbm"], MODEL_STATE["X_train"])
    MODEL_STATE["anchors"] = AnchorsExplainer(MODEL_STATE["gbm"], MODEL_STATE["X_train"])
    MODEL_STATE["cem"] = CEMExplainer(MODEL_STATE["keras_surrogate"], MODEL_STATE["X_train"])
    MODEL_STATE["proto"] = PrototypeExplainer(MODEL_STATE["X_train"], ...)
    MODEL_STATE["arch"] = ArchipelagoExplainer(MODEL_STATE["shap"].explainer)
    MODEL_STATE["store"] = ExplanationStore("sqlite:///explanations.db")
    MODEL_STATE["drift"] = ExplanationDriftDetector()
    MODEL_STATE["trust"] = TrustScorer()
    MODEL_STATE["narrator"] = LLMNarrator()
    yield
    # Cleanup on shutdown

app = FastAPI(lifespan=lifespan)

@app.post("/explain")
async def explain(request: ExplainRequest):
    features = request.features.dict()
    X = np.array([list(features.values())])

    # Run SHAP + Anchors + Prototypes in parallel
    shap_task = asyncio.to_thread(MODEL_STATE["shap"].explain, features, MODEL_STATE["gbm"])
    anchors_task = asyncio.to_thread(MODEL_STATE["anchors"].explain, features)
    proto_task = asyncio.to_thread(MODEL_STATE["proto"].explain, features)
    arch_task = asyncio.to_thread(MODEL_STATE["arch"].get_interactions, features)

    shap_result, anchors_result, proto_result, arch_result = await asyncio.gather(
        shap_task, anchors_task, proto_task, arch_task
    )

    stability = MODEL_STATE["shap"].stability_score(features)
    trust = MODEL_STATE["trust"].score(shap_result["shap_values"], MODEL_STATE["gbm"], features, stability)
    causal = MODEL_STATE["causal"].annotate_shap(shap_result["shap_values"])

    # CEM runs synchronously (slower)
    cem_result = MODEL_STATE["cem"].explain(features)

    # LLM narrates ONLY pre-computed data
    learner_narrative = await MODEL_STATE["narrator"].narrate_for_learner(
        shap_result, anchors_result, proto_result, trust
    )

    explanation = build_explanation_result(
        shap_result, anchors_result, proto_result,
        arch_result, cem_result, trust, causal, learner_narrative
    )

    # Store for consistency tracking
    drift = MODEL_STATE["drift"].check_learner_drift(
        request.learner_id, MODEL_STATE["store"]
    )
    MODEL_STATE["store"].save(request.learner_id, explanation)

    return {"status": "ok", "data": {**explanation, "consistency": drift}}
```
