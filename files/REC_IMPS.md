Task: Implement 10 Features for the XAI Learning Recommendation System
You are working on a FastAPI backend for an Explainable AI Learning Recommendation System. The system has TWO engines: a Dropout Risk Engine (fully featured) and a Recommendation Engine (missing several XAI features). Your job is to bring the Recommendation Engine to parity with the Dropout Engine, and add 3 new recommendation-specific features.

Codebase Context
Key Files:

backend/app/main.py — FastAPI app, all endpoints, AppState holds all models/explainers. Lifespan function loads everything at startup.
backend/app/recommender/ranker_explainer.py — RankerExplainer class wrapping a LightGBM LambdaMART ranker. Has score_items(), explain(), whatif(), health(). Uses pred_contrib for native SHAP.
backend/app/narrator/llm_narrator.py — LLMNarrator class using Groq llama-3.3-70b-versatile. Has .narrate(explain_resp, learner_id, audience) method. Currently only used by the dropout /explain endpoint.
backend/app/evaluator/trust_scorer.py — TrustScorer class computing fidelity + stability + completeness → composite 0-1 trust score. Currently only used by dropout engine.
backend/app/explainers/anchors_explainer.py — AnchorsExplainer using Alibi's AnchorTabular. Currently only used by dropout engine.
backend/app/explainers/prototype_explainer.py — PrototypeExplainer using KNN on training data. Currently only used by dropout engine.
backend/app/tracker/consistency_store.py — ExplanationStore persisting per-learner explanation records with top-3 features to SQLite. Currently only used by dropout engine.
backend/app/tracker/drift_detector.py — ExplanationDriftDetector comparing top-3 features session-to-session via Jaccard similarity. Currently only used by dropout engine.
backend/app/causal/causal_annotator.py — CausalAnnotator using DoWhy. Builds a DAG defined in _build_causal_graph() with node groups: confounders, engagement, performance, risk_signals, outcome. Has .get_causal_effects() returning {feature: CausalEffect}.
AppState attributes (in main.py):

state.student_ranker_explainer: RankerExplainer — loaded from models/recommenders/student_ranker.pkl
state.instructor_ranker_explainer: RankerExplainer — loaded from models/recommenders/instructor_ranker.pkl
state.llm_narrator: LLMNarrator
state.trust_scorer: TrustScorer
state.causal_annotator: CausalAnnotator
state.explanation_store: ExplanationStore
state.drift_detector: ExplanationDriftDetector
Student ranker features (39 total): learner_id, current_module, current_presentation, recommended_module, recommended_presentation, implicit_clicks_14d, implicit_video_watch_ratio_14d, implicit_forum_events_14d, implicit_quiz_attempts_14d, implicit_avg_dwell_time_min_14d, implicit_save_events_14d, implicit_search_events_14d, implicit_last_recommendation_interaction_days, explicit_interest_level, learning_momentum, dropout_risk, base_login_frequency_weekly, base_video_completion_rate, base_quiz_completion_rate, base_quiz_avg_score, base_assignment_submission_rate, base_missed_deadlines_count, base_prior_course_completions, base_days_since_last_activity, base_studied_credits, base_num_of_prev_attempts, base_avg_session_duration_min, base_help_requests_count, base_current_week_in_course, explicit_gender, explicit_region, explicit_highest_education, explicit_imd_band, explicit_age_band, explicit_disability, explicit_education_score, explicit_imd_score, explicit_age_score, synthetic_segment

Encoder maps (categorical features): learner_id, current_module, current_presentation, recommended_module, recommended_presentation, explicit_gender, explicit_region, explicit_highest_education, explicit_imd_band, explicit_age_band, explicit_disability, synthetic_segment

Instructor ranker features (9 total): instructor_archetype, student_id, learner_id, student_current_module, student_current_presentation, recommended_module, recommended_presentation, student_affinity_score, cohort_signal_score

Existing recommendation endpoints in main.py:

GET /recommend/health
POST /recommend/student — scores & ranks items, returns ScoredItem list
POST /recommend/student/explain — full XAI explanation for one item, returns RecommendationExplanation
POST /recommend/student/whatif — what-if counterfactual
POST /recommend/instructor — same for instructor
POST /recommend/instructor/explain
POST /recommend/instructor/whatif
Existing RecommendationExplanation dataclass fields: item_id, score, shap_values, top_features, anchor_rule, feature_interactions, causal_annotations, shap_stability, plain_language

Features to Implement
Feature 1: LLM Narration for Recommendations
Wire state.llm_narrator into /recommend/student/explain and /recommend/instructor/explain. After computing the RecommendationExplanation, call state.llm_narrator.narrate() with the explanation data, passing audience="learner" for student endpoints and audience="instructor" for instructor endpoints. Add a narratives field to the RecommendationExplanation dataclass and return it. The narrator needs a recommendation-specific context string (not dropout context) — modify the narrate call or add a context_type="recommendation" parameter so the LLM knows it's explaining a content recommendation, not a dropout risk prediction. If state.llm_narrator is None, set narratives = None.

Feature 2: Trust Score for Recommendations
Add a trust_score field to RecommendationExplanation. Compute it inside RankerExplainer.explain() as a composite of:

Fidelity (40%): Check if SHAP values sum approximately to the prediction score. The pred_contrib call returns SHAP values + a bias term (last column). Fidelity = 1.0 - min(abs(sum(shap) + bias - score) / max(abs(score), 1e-8), 1.0).
Stability (35%): Already computed as shap_stability — reuse it.
Completeness (25%): Fraction of total absolute SHAP accounted for by top 5 features. = sum(|top5_shap|) / sum(|all_shap|).
Return as {"trust_score": float, "fidelity": float, "stability": float, "completeness": float}.

Feature 3: Better Anchor Rules (Alibi-based)
Replace the simplified _anchor_rule() method in RankerExplainer with a proper Alibi AnchorTabular approach. Create a pseudo-classifier wrapper around the ranker: def _pseudo_predict(X): return (model.predict(X) >= median_score).astype(int). Initialize AnchorTabular with the training data at startup. In explain(), use AnchorsExplainer.explain(features_dict) to get a validated anchor with a precision score. Fall back to the current template-based approach if Alibi fails. Add anchor_precision float field to RecommendationExplanation. Note: this requires storing a small sample of training data inside RankerExplainer — modify the constructor to accept an optional X_train_sample: np.ndarray parameter and pass it during loading in main.py lifespan.

Feature 4: Real Feature Interactions (LightGBM Native)
Replace the fake |shap_i × shap_j| interaction calculation in _feature_interactions(). LightGBM supports model.predict(X, pred_interact=True) which returns a (n, n_features, n_features) interaction matrix. Use this to compute the true pairwise interaction values. Extract the top-3 pairs by absolute interaction value. Update _feature_interactions() to use this, and fall back to the current product-based approximation if pred_interact raises an exception.

Feature 5: KNN Prototype Explainer
Add a prototypes field to RecommendationExplanation. Inside RankerExplainer, add a _prototypes() method that finds the K=3 nearest neighbors from a reference pool based on feature similarity (Euclidean distance on numeric features only). For each neighbor, report {item_id, score, similarity, outcome}. The reference pool should be loaded from data/recommendations/precomputed/student_topk.csv (already exists, 14MB) at startup and stored in RankerExplainer. Add a reference_pool: Optional[pd.DataFrame] parameter to the constructor. If the reference pool is None, return an empty list. Modify the lifespan in main.py to load and pass this CSV.

Feature 6: Drift Detection for Recommendations
Create a new RecommendationExplanationStore class in backend/app/tracker/reco_consistency_store.py following the exact same pattern as consistency_store.py. It should have the same schema: learner_id, timestamp, score (instead of risk_score), top3_features, shap_values, extra. Add .save(), .get_history(), .get_top3_timeline(), .get_latest() methods.

Initialize it in the main.py lifespan as state.reco_explanation_store = RecommendationExplanationStore(). Add a state.reco_explanation_store attribute to AppState.

In /recommend/student/explain, after computing the explanation, save the top-3 features and SHAP values to the store. Also run state.drift_detector.check_learner_drift(learner_id, state.reco_explanation_store) and add explanation_drift to the response.

Feature 10: /causal/graph Endpoint
Add a GET /causal/graph endpoint to main.py that returns the causal DAG as a JSON structure the frontend can render. The endpoint should:

Guard with if state.causal_annotator is None: raise HTTPException(503, ...).
Get causal effects via state.causal_annotator.get_causal_effects().
Define node groups from the existing DAG structure in causal_annotator.py:
confounder: ["prior_course_completions", "current_week_in_course"]
engagement: ["login_frequency_weekly", "avg_session_duration_min", "forum_posts_count", "help_requests_count"]
performance: ["quiz_avg_score", "quiz_completion_rate", "assignment_submission_rate", "video_completion_rate"]
risk_signal: ["days_since_last_activity", "missed_deadlines_count"]
outcome: ["dropout_risk"]
For each node, include: id, label (title-cased, underscores replaced with spaces), group, is_causal (bool from CausalEffect), ate (float), effect_direction (str).
Build edges from the same logic in _build_causal_graph(): confounders→engagement, confounders→performance, confounders→outcome, engagement→performance, engagement→outcome, engagement→risk_signals, performance→outcome, risk_signals→outcome.
Return {"nodes": [...], "edges": [{"from": src_id, "to": tgt_id}, ...]}.
Feature 11: Diversity Score
Add a diversity_score field to the /recommend/student response. After scoring and ranking items with score_items(), compute diversity as:

Extract the recommended_module value from each top-K item's features.
n_unique_modules = len(set(modules))
diversity_score = n_unique_modules / len(top_k_items)
Return this as a float 0-1. Also add a diversity_warning string: if diversity_score < 0.5, return "Low diversity: recommendations are concentrated in few modules. Consider exploring other topics.", else null.

Feature 12: Fairness Audit
Create a new file backend/app/recommender/fairness_auditor.py. Implement a FairnessAuditor class that checks whether recommendation scores systematically differ across protected groups.

The class should:

Accept feature_columns: list[str] and protected_features: list[str] (default: ["explicit_gender", "explicit_age_band", "explicit_disability"]).
Have an audit(items: list[dict], scores: list[float]) -> FairnessReport method.
For each protected feature present in the items, group the items by their value of that feature, compute the mean score per group, and flag if any group's mean score deviates >15% from the overall mean.
Return a FairnessReport dataclass with fields: overall_fair: bool, group_scores: dict[str, dict[str, float]] (feature → {group_value → mean_score}), flagged_disparities: list[dict] (each with feature, group, score, deviation_pct, direction).
Include a .to_dict() method.
Initialize it at startup in main.py lifespan and store as state.fairness_auditor. Wire it into /recommend/student: after scoring items, run the audit and return fairness_audit in the response. If state.fairness_auditor is None, omit it.

Feature 13: Recommendation Explanation Consistency Store
This is the storage foundation that Feature 6 depends on. Create backend/app/tracker/reco_consistency_store.py as described in Feature 6. This is already covered by Feature 6's instructions — implement them together.

Rules
Do NOT modify any existing dropout engine endpoints or explainers.
Do NOT remove any existing fields from response schemas — only add new fields.
All new dataclass fields must have defaults (= None, = 0.0, = []) so existing callers don't break.
Guard all new features with if state.X is not None checks — graceful degradation.
Preserve all existing comments and docstrings.
Update the AppState class to include any new attributes with Optional typing and None defaults.
All new files must include a module docstring with Public API documentation.
Log initialization of all new components at INFO level.