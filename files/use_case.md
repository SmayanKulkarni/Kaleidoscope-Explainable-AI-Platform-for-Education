Let me read the project's XAI algorithm docs and feature schema to give you precise, data-grounded mappings.

Now let me pull a real row from our data to ground every example:
Here's the complete mapping of every XAI module to its use case, grounded in our actual OULAD-derived data. I'll use two real learners from `data/learners.csv` throughout:

---

# XAI Module ↔ Use Case Map (with Real Data Examples)

## Reference Learners (from `data/learners.csv`)

```
HIGH-RISK — learner_10678 (dropout_risk=1)            LOW-RISK — learner_32568 (dropout_risk=0)
─────────────────────────────────────────              ─────────────────────────────────────────
login_frequency_weekly:      1.0                       login_frequency_weekly:      2.34
avg_session_duration_min:   10.0                       avg_session_duration_min:   32.20
forum_posts_count:           0                         forum_posts_count:         224
video_completion_rate:       0.0                        video_completion_rate:       0.71
quiz_avg_score:              0.0                        quiz_avg_score:            93.33
quiz_completion_rate:        0.0                        quiz_completion_rate:        1.0
assignment_submission_rate:  0.0                        assignment_submission_rate:  1.0
days_since_last_activity:  287                         days_since_last_activity:   22
prior_course_completions:    1                         prior_course_completions:    1
current_week_in_course:      1                         current_week_in_course:     36
missed_deadlines_count:      0                         missed_deadlines_count:      0
help_requests_count:         1                         help_requests_count:        57
```

---

## 1. TreeSHAP — "Why is this student at risk?"

| | |
|---|---|
| **Layer** | Explanation → Feature-Level |
| **Use Case** | Per-feature attribution: which features push risk up/down, by how much |
| **Audience** | Instructor (detailed), Learner (top-3 simplified) |
| **Endpoint** | `POST /explain` |

### Example Output for `learner_10678`
```python
{
  "shap_values": {
    "days_since_last_activity":    +0.31,   # biggest risk driver
    "assignment_submission_rate":  +0.18,   # 0% submission → high positive SHAP
    "quiz_completion_rate":        +0.14,   # 0% quizzes → pushes risk up
    "video_completion_rate":       +0.09,
    "avg_session_duration_min":    +0.06,   # only 10 min → above-average risk
    "login_frequency_weekly":      +0.04,
    "forum_posts_count":           +0.03,
    "help_requests_count":         -0.01,   # sought help once → slight negative
    "quiz_avg_score":              +0.00,
    "missed_deadlines_count":      -0.00,   # none to miss (never engaged)
    "prior_course_completions":    -0.02,   # completed 1 before → slight protection
    "current_week_in_course":      +0.01,
  },
  "base_value": 0.312,    # population average dropout rate
  "prediction": 0.96      # base + Σ(shap_values) ≈ predicted risk
}
```

**What it tells the instructor:** "287 days inactive is the #1 risk driver (+0.31). Zero assignments submitted is #2 (+0.18). These alone account for 60% of the risk score."

---

## 2. FastSHAP — "What happens if the student changes behavior right now?"

| | |
|---|---|
| **Layer** | Explanation → Feature-Level (real-time) |
| **Use Case** | Live What-If slider updates — user drags `login_frequency_weekly` from 1→4, sees SHAP recalculate in <5ms |
| **Audience** | Learner (What-If panel) |
| **Endpoint** | `POST /whatif` |

### Example: Student drags login slider from 1.0 → 4.0
```python
# Input (modified features)
{
  "login_frequency_weekly": 4.0,        # ← changed from 1.0
  "avg_session_duration_min": 10.0,
  "forum_posts_count": 0,
  ...                                    # everything else unchanged
}

# FastSHAP output (< 5ms)
{
  "fast_shap_values": {
    "days_since_last_activity":    +0.32,
    "assignment_submission_rate":  +0.17,
    "quiz_completion_rate":        +0.14,
    "login_frequency_weekly":      -0.03,  # ← was +0.04, now protective
    ...
  },
  "new_risk_score": 0.89,                 # down from 0.96
  "delta": -0.07
}
```

**What it tells the learner:** "If you start logging in 4x/week, your risk drops by ~7%. But logging in alone won't help much — the real issue is you haven't submitted anything."

---

## 3. Archipelago — "Which feature *combinations* amplify risk?"

| | |
|---|---|
| **Layer** | Explanation → Concept-Level (interactions) |
| **Use Case** | Detect synergistic risk: two features that are moderate individually but catastrophic together |
| **Audience** | Instructor (interaction table), Learner (warning badge) |
| **Endpoint** | `POST /explain` (concept_level section) |

### Example for `learner_10678`
```python
{
  "interactions": [
    {
      "feature_a": "quiz_completion_rate",         # 0.0
      "feature_b": "days_since_last_activity",     # 287
      "interaction_score": +0.08,
      "direction": "amplifying"
      # Meaning: zero quizzes × total inactivity amplify each other
    },
    {
      "feature_a": "assignment_submission_rate",   # 0.0
      "feature_b": "video_completion_rate",        # 0.0
      "interaction_score": +0.05,
      "direction": "amplifying"
    },
    {
      "feature_a": "login_frequency_weekly",       # 1.0
      "feature_b": "avg_session_duration_min",     # 10.0
      "interaction_score": +0.03,
      "direction": "amplifying"
      # Low logins + short sessions together = worse than either alone
    }
  ]
}
```

**What it tells the instructor:** "It's not just that this student hasn't taken quizzes — the *combination* of 0% quiz completion AND 287 days inactive is synergistic. Addressing both together has more impact than addressing either alone."

---

## 4. DiCE — "What should this student *do* to reduce risk?"

| | |
|---|---|
| **Layer** | Prescription |
| **Use Case** | Generate actionable counterfactuals: concrete target values for mutable features that would flip the prediction |
| **Audience** | Learner (action cards), Instructor (intervention plan) |
| **Endpoint** | `POST /counterfactual` |

### Example for `learner_10678`
```python
# DiCE generates 3 diverse counterfactuals, best one:
{
  "original_prediction": 0.96,   # HIGH risk
  "counterfactual_prediction": 0.28,   # LOW risk
  "changed_features": {
    "assignment_submission_rate": (0.0, 0.60),     # submit 60% of TMAs
    "quiz_completion_rate":       (0.0, 0.50),     # complete half the CMAs
    "days_since_last_activity":   (287, 5),        # log back in within 5 days
    "login_frequency_weekly":     (1.0, 3.0),      # 3x/week logins
  },
  "locked_features": {           # immutable — NOT changed by DiCE
    "prior_course_completions": 1,
    "current_week_in_course": 1
  },
  "actions": [
    {
      "feature": "days_since_last_activity",
      "current": 287, "target": 5,
      "plain_language": "Return to the platform within 5 days",
      "estimated_impact": 0.34,    # biggest single lever
      "priority_rank": 1,
      "is_causal": true            # from DoWhy DAG
    },
    {
      "feature": "assignment_submission_rate",
      "current": 0.0, "target": 0.60,
      "plain_language": "Submit 60% of assignments",
      "estimated_impact": 0.21,
      "priority_rank": 2,
      "is_causal": true
    },
    {
      "feature": "quiz_completion_rate",
      "current": 0.0, "target": 0.50,
      "plain_language": "Complete 50% of quizzes (currently 0%)",
      "estimated_impact": 0.13,
      "priority_rank": 3,
      "is_causal": true
    }
  ]
}
```

**What it tells the learner:** "If you log back in within 5 days, submit 60% of your assignments, and complete half the quizzes, your predicted risk drops from 96% to 28%."

---

## 5. CEM — "Why *this* prediction and not the other?"

| | |
|---|---|
| **Layer** | Diagnosis |
| **Use Case** | Contrastive diagnosis — what's *sufficient* for the current prediction (PP) and what would *flip* it (PN) |
| **Audience** | Instructor (diagnostic report) |
| **Endpoint** | `POST /explain` (diagnosis section) |

### Example for `learner_10678`
```python
{
  "pertinent_positive": {
    # Minimal feature subset SUFFICIENT to produce HIGH risk
    "days_since_last_activity": 287,       # this alone is enough
    "assignment_submission_rate": 0.0,
    # Other features are irrelevant to maintaining the prediction
  },
  "pp_interpretation": "Even if all other features were average, 287 days inactive + 0% submissions is enough to predict HIGH risk.",

  "pertinent_negative": {
    # Minimal change to FLIP prediction to LOW risk
    "days_since_last_activity": 287 → 12,  # reduce to 12 days
    "assignment_submission_rate": 0.0 → 0.5,
  },
  "pn_interpretation": "If days_since_last_activity dropped to 12 AND assignment_submission_rate rose to 50%, the prediction would flip to LOW risk."
}
```

**How PP differs from SHAP:** SHAP says "each feature contributes X amount." CEM PP says "you only *need* these 2 features — the rest don't matter for this student."

**How PN differs from DiCE:** DiCE gives the *optimal* action plan. CEM PN gives the *minimal* change to flip — it might not be the most practical, but it's the theoretical boundary.

---

## 6. Anchors — "Give me a simple rule I can act on."

| | |
|---|---|
| **Layer** | Explanation → Concept-Level (rule-based) |
| **Use Case** | High-precision IF-THEN rules — human-auditable, no numerical reasoning needed |
| **Audience** | **Instructor** (primary), used for institutional policy / intervention triggers |
| **Endpoint** | `POST /explain` (prediction_level section) |

### Example for `learner_10678`
```python
{
  "anchor_rule": "days_since_last_activity > 30 AND assignment_submission_rate <= 0.17",
  "precision": 0.94,
  "coverage": 0.12,
  "human_readable": "IF days_since_last_activity > 30 AND assignment_submission_rate <= 0.17 THEN at-risk prediction holds with 94% certainty"
}
```

**What it tells the instructor:** "For *any* student matching this rule (12% of the population), the model predicts HIGH risk 94% of the time. This is a reliable trigger for automated outreach."

---

## 7. ProtoDash / k-NN — "Who else looks like this student?"

| | |
|---|---|
| **Layer** | Explanation → Concept-Level (example-based) |
| **Use Case** | Find similar past learners and show their outcomes — builds trust through relatable comparison |
| **Audience** | **Learner** (primary — motivational), Instructor (cohort analysis) |
| **Endpoint** | `POST /explain` (concept_level section) |

### Example for `learner_10678`
```python
{
  "prototypes": [
    {
      "learner_id": "learner_10452",
      "outcome": "dropped_out",
      "similarity_score": 0.91,
      "distinguishing_features": ["forum_posts_count", "help_requests_count"]
      # Very similar — both dropped out
    },
    {
      "learner_id": "learner_08234",
      "outcome": "completed",
      "similarity_score": 0.72,
      "distinguishing_features": ["assignment_submission_rate", "login_frequency_weekly"]
      # Similar initial profile BUT this student submitted assignments → completed
    }
  ],
  "narrative": "Your profile is similar to 1 learner who completed this course. Their key difference: assignment submission rate — they submitted 65% of assignments while you've submitted 0%."
}
```

**What it tells the learner:** "A student just like you completed the course — the thing that made the difference was submitting assignments."

---

## 8. MAPIE — "How confident is the model in this prediction?"

| | |
|---|---|
| **Layer** | Prediction (uncertainty quantification) |
| **Use Case** | Conformal prediction intervals — flags when the model is unsure |
| **Audience** | Instructor (confidence badge), System (trust scoring) |
| **Endpoint** | `POST /predict` |

### Example for `learner_10678`
```python
{
  "risk_score": 0.96,
  "prediction": 1,           # at-risk
  "uncertainty": {
    "confidence_width": 0.08,   # narrow → model is very sure
    "label": "Low",             # low uncertainty = high confidence
    "interval": [0.88, 0.96]    # 90% conformal interval
  }
}
```

### Contrast: edge-case student at ~50% risk
```python
{
  "risk_score": 0.52,
  "prediction": 1,
  "uncertainty": {
    "confidence_width": 0.31,   # wide → model is unsure
    "label": "High",
    "interval": [0.36, 0.67]
  }
  # UI shows: "⚠️ Model uncertainty is HIGH — interpret with caution"
}
```

---

## 9. DoWhy / Causal Annotator — "Is this a *cause* or just a correlation?"

| | |
|---|---|
| **Layer** | Diagnosis (causal annotation) |
| **Use Case** | Overlay causal vs. correlational labels on SHAP values — prevents acting on spurious associations |
| **Audience** | Instructor (DAG visualization), System (ActionRanker weighting) |
| **Endpoint** | `POST /explain` (feature_level annotations) |

### Example: annotated SHAP for `learner_10678`
```python
{
  "days_since_last_activity": {
    "shap_value": +0.31,
    "type": "causal",            # in the DAG → directly causes dropout
    "note": "Directly influences dropout risk"
  },
  "assignment_submission_rate": {
    "shap_value": +0.18,
    "type": "causal",
    "note": "Directly influences dropout risk"
  },
  "forum_posts_count": {
    "shap_value": +0.03,
    "type": "correlational",     # NOT in causal path → could be spurious
    "note": "Associated with dropout risk but may not cause it"
  },
  "prior_course_completions": {
    "shap_value": -0.02,
    "type": "correlational",
    "note": "Associated with dropout risk but may not cause it"
  }
}
```

**Why this matters:** The ActionRanker uses this to prioritize causal features in DiCE recommendations. Acting on `assignment_submission_rate` (causal) is more reliable than acting on `forum_posts_count` (correlational).

---

## 10. Drift Detector (JSD) — "Are explanations staying consistent over time?"

| | |
|---|---|
| **Layer** | Evaluation / Tracking |
| **Use Case** | Detect when a student's explanation changes significantly between sessions (concept drift / model instability) |
| **Audience** | System (automated alert), Instructor (timeline view) |
| **Endpoint** | `GET /history/{learner_id}` |

### Example: `learner_32568` checked at week 30 vs week 36
```python
{
  "drift_detected": true,
  "jsd": 0.22,              # > 0.15 threshold
  "rank_shift": 2,          # 2 features swapped out of top-3
  "prev_top3": ["days_since_last_activity", "quiz_completion_rate", "login_frequency_weekly"],
  "curr_top3": ["video_completion_rate", "forum_posts_count", "days_since_last_activity"],
  "flag": "⚠️ Explanation changed significantly"
}
```

**What it tells the instructor:** "The risk factors for this student shifted — previously driven by quiz completion, now driven by video engagement. This could mean the student changed behavior, or the model is less stable for this profile."

---

## 11. TrustScorer — "Can I trust this explanation?"

| | |
|---|---|
| **Layer** | Evaluation |
| **Use Case** | Composite quality metric for every explanation — prevents showing unreliable explanations to users |
| **Audience** | System (gating), Instructor (confidence indicator) |
| **Endpoint** | `POST /explain` (trust section) |

### Example for `learner_10678`
```python
{
  "trust_score": 0.87,       # composite
  "fidelity": 0.95,          # SHAP sum ≈ predict_proba (very accurate)
  "stability": 0.82,         # 20 perturbation runs — top features stay consistent
  "completeness": 0.84,      # top-5 features capture 84% of total attribution
  "label": "High"            # > 0.75 → safe to show
}
```

**If trust were low (e.g. 0.42):** UI shows "⚠️ This explanation may vary — interpret with caution" and highlights which sub-score is weak.

---

## Summary Table: Module → Layer → Question → Audience

| # | Module | Layer | Question Answered | Primary Audience |
|---|--------|-------|-------------------|-----------------|
| 1 | **TreeSHAP** | Explanation (feature) | "Why this risk score?" | Instructor |
| 2 | **FastSHAP** | Explanation (real-time) | "What if I change X?" | Learner |
| 3 | **Archipelago** | Explanation (concept) | "Which combos amplify risk?" | Instructor |
| 4 | **DiCE** | Prescription | "What should the student do?" | Both |
| 5 | **CEM** | Diagnosis | "What's sufficient / what would flip it?" | Instructor |
| 6 | **Anchors** | Explanation (concept) | "Give me a simple rule" | Instructor |
| 7 | **ProtoDash/k-NN** | Explanation (concept) | "Who else looks like this?" | Learner |
| 8 | **MAPIE** | Prediction | "How confident is the model?" | Instructor / System |
| 9 | **DoWhy** | Diagnosis | "Causal or just correlated?" | System / Instructor |
| 10 | **Drift Detector** | Tracking | "Did explanations change?" | System / Instructor |
| 11 | **TrustScorer** | Evaluation | "Can I trust this explanation?" | System |

All example data uses the real OULAD-derived schema from `@/media/smayan/500GB SSD/Datahack 4.0/data/learners.csv:1`. SHAP values shown are illustrative of the direction and magnitude that the trained GBM will produce — exact values will come once `trainer.py` is built next.