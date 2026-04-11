# System Architecture — XAI Learning Recommendation System

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                          │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Learner    │  │  Instructor  │  │    What-If Panel       │  │
│  │  Dashboard  │  │  Dashboard   │  │   (FastSHAP live)      │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘  │
│         └────────────────┴──────────────────────┘               │
│                          │ HTTP / REST                           │
└──────────────────────────┼───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    FASTAPI BACKEND                               │
│                                                                  │
│  /predict  /explain  /whatif  /counterfactual  /history          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              PREDICTION LAYER                           │    │
│  │   GBM Model (primary) + RF Model (comparison)           │    │
│  │   MAPIE Uncertainty Estimator                           │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │              EXPLANATION LAYER                          │    │
│  │                                                         │    │
│  │  Feature Level:   TreeSHAP + FastSHAP (live)            │    │
│  │                   Archipelago (interactions)            │    │
│  │                                                         │    │
│  │  Concept Level:   LIME over concept groups              │    │
│  │                   Prototype Match (NearestNeighbors)    │    │
│  │                   Anchors (rule extraction)             │    │
│  │                                                         │    │
│  │  Prediction Level: Global SHAP + PDP                    │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │              DIAGNOSIS LAYER                            │    │
│  │   CEM: Pertinent Positives + Pertinent Negatives        │    │
│  │   Causal Annotator (DoWhy SCM)                          │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │              PRESCRIPTION LAYER                         │    │
│  │   DiCE Counterfactuals (actionable constraints)         │    │
│  │   Action Ranker (SHAP × actionability × causality)      │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │              EVALUATION + TRACKING                      │    │
│  │   Trust Scorer (fidelity + stability + completeness)    │    │
│  │   Drift Detector (JSD on SHAP distributions)            │    │
│  │   Explanation Store (SQLite history)                    │    │
│  └──────────────────────┬──────────────────────────────────┘    │
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────┐    │
│  │              NARRATION LAYER (LLM)                      │    │
│  │   Claude Sonnet — narrates ONLY computed outputs        │    │
│  │   Learner view: motivational + plain language           │    │
│  │   Instructor view: technical + precise                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Full Explanation Request

```
User (learner or instructor)
        │
        ▼
POST /explain {learner_id, features, audience}
        │
        ├──────────────────────────────────────────────────────┐
        │                                                      │
        ▼  [async parallel]                                    ▼
TreeSHAP(features)                              AnchorsTabular(features)
    → shap_values dict                              → anchor_rule, precision
    → stability_score (n=20 runs)                   → coverage
    → global_importance                             │
        │                                           │
        ├── Archipelago(shap_interaction_values)    │
        │       → top_interactions list             │
        │                                           │
        ├── PrototypeExplainer(features)             │
        │       → similar_learners + narrative      │
        │                                           │
        └───────────────────────────────────────────┘
                                │
                                ▼  [synchronous — slower]
                        CEM(features)
                            → pertinent_positives
                            → pertinent_negatives
                                │
                                ▼
                        CausalAnnotator(shap_values)
                            → causal/correlational tags
                                │
                                ▼
                        TrustScorer(shap, stability)
                            → trust_score {fidelity, stability, completeness}
                                │
                                ▼
                        ExplanationStore.save()
                        DriftDetector.check()
                                │
                                ▼
                        LLMNarrator(all_computed_data)
                            → learner_narrative (≤100 words)
                            → instructor_narrative (≤150 words)
                                │
                                ▼
                        ExplanationResult (full)
                                │
                                ▼
                        Response to Frontend
```

---

## Algorithm Decision Tree

```
Incoming request type?
│
├── /predict or /whatif
│       └── Use: GBM.predict_proba() + FastSHAP (real-time)
│
├── /explain
│       ├── Feature level   → TreeSHAP + Archipelago
│       ├── Concept level   → LIME over concepts + Prototypes + Anchors
│       ├── Prediction level → Global SHAP summary
│       ├── Diagnosis       → CEM (PP + PN)
│       └── Prescription    → DiCE + ActionRanker
│
├── /counterfactual
│       └── DiCE with actionability_constraints → ranked PrescriptiveActions
│
└── /history/{id}
        └── ExplanationStore.get_history() + DriftDetector.check()
```

---

## Module Dependencies

```
main.py
    ├── model/trainer.py          (GBM, RF, calibration)
    ├── model/predictor.py        (predict_proba wrapper)
    ├── explainers/
    │   ├── shap_explainer.py     (depends on: model)
    │   ├── fastshap_explainer.py (depends on: model, shap_explainer)
    │   ├── archipelago.py        (depends on: shap_explainer)
    │   ├── lime_explainer.py     (depends on: model)
    │   ├── anchors_explainer.py  (depends on: model, X_train)
    │   ├── dice_explainer.py     (depends on: model, X_train)
    │   ├── cem_explainer.py      (depends on: keras_surrogate)
    │   └── prototype_explainer.py(depends on: X_train, y_train)
    ├── causal/
    │   └── causal_annotator.py   (depends on: DAG definition)
    ├── prescriptor/
    │   └── action_ranker.py      (depends on: dice_explainer, shap_explainer)
    ├── evaluator/
    │   ├── trust_scorer.py       (depends on: shap_explainer)
    │   └── uncertainty_estimator.py (depends on: model)
    ├── tracker/
    │   ├── consistency_store.py  (depends on: SQLAlchemy)
    │   └── drift_detector.py     (depends on: consistency_store)
    └── narrator/
        └── llm_narrator.py       (depends on: anthropic SDK, all explainers)
```

---

## Startup Sequence

```python
# All models and explainers loaded ONCE at startup via lifespan context
# Order matters — some explainers depend on others

1. Load GBM model (models/gbm.pkl)
2. Load RF model (models/rf.pkl)
3. Load X_train (data/train.pkl)
4. Initialize SHAPExplainer(gbm, X_train)         # fit TreeExplainer
5. Initialize FastSHAPExplainer(gbm, X_train)      # train surrogate (2-3 min)
6. Initialize ArchipelagoExplainer(shap_explainer)
7. Initialize LIMEExplainer(gbm, X_train)
8. Initialize AnchorsExplainer(gbm, X_train)       # fit discretizer
9. Initialize DiCEExplainer(gbm, X_train)
10. Build Keras surrogate from GBM
11. Initialize CEMExplainer(keras_surrogate, X_train)
12. Initialize PrototypeExplainer(X_train, y_train, learner_ids)
13. Initialize UncertaintyEstimator(gbm).fit(X_train, y_train)
14. Initialize CausalAnnotator(data)
15. Initialize ExplanationStore(db_url)
16. Initialize ExplanationDriftDetector()
17. Initialize TrustScorer()
18. Initialize LLMNarrator()
# Ready — /health returns 200
```

---

## Key Design Decisions

### Why GradientBoostingClassifier?
- Native TreeSHAP support → exact, fast SHAP values
- Calibratable with `CalibratedClassifierCV`
- Strong performance on tabular data (AUC > 0.90 on educational datasets)
- `shap_interaction_values()` available via TreeExplainer

### Why FastSHAP for What-If?
- Standard SHAP: ~50-200ms per call (too slow for slider UI)
- FastSHAP surrogate: ~3-5ms (real-time feel)
- Trade-off: ~5% accuracy loss — acceptable for interactive UI

### Why CEM over pure SHAP for Diagnosis?
- SHAP tells you *how much* each feature contributed
- CEM tells you *which features are sufficient* (PP) and *which removal flips prediction* (PN)
- PP → diagnosis ("here's the core problem")
- PN → prescription preview ("here's the minimal fix")
- Together: the most complete diagnostic picture

### Why Archipelago for Interactions?
- Standard SHAP treats features as independent
- Educational dropout has strong synergistic risks: low engagement × missed deadlines
- SHAP interaction values via TreeExplainer give exact interaction attribution

### Why Prototypes alongside SHAP?
- SHAP values are numbers — hard for learners to interpret
- "Similar to 2 students who completed" is immediately comprehensible
- Dual-audience requirement: prototypes serve the learner; SHAP serves the instructor

### Why DoWhy for Causal Layer?
- SHAP/LIME are correlational — high `days_since_activity` may correlate with but not *cause* dropout
- DoWhy SCM + backdoor adjustment identifies true causal effects
- Annotation: marks features as causal vs correlational in every explanation
- Crucial for action prioritization: causal actions get higher priority weight

---

## Trust Score Design

```
Trust Score = 0.40 × Fidelity
            + 0.35 × Stability
            + 0.25 × Completeness

Fidelity:     1 - |predict_proba(x) - (SHAP_sum + base_value)|
              Measures: does the explanation accurately reflect the model?

Stability:    1 - normalized_std(SHAP_rank_under_noise, n=20)
              Measures: does the explanation change wildly with small input changes?

Completeness: |top_k_SHAP_sum| / |total_SHAP_sum|  (k=5)
              Measures: how much of the prediction does the explanation cover?

Score bands:
  0.75–1.00 → High   (green badge) — explanation is reliable
  0.50–0.74 → Medium (yellow badge) — use with caution
  0.00–0.49 → Low    (red badge) — ⚠️ explanation may be misleading
```

---

## Deployment

```yaml
# docker-compose.yml
services:
  backend:
    build: .
    dockerfile: Dockerfile.backend
    ports: ["8000:8000"]
    environment:
      - MODEL_PATH=/app/models
      - DB_URL=sqlite:///explanations.db
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s

  frontend:
    build: .
    dockerfile: Dockerfile.frontend
    ports: ["3000:80"]
    depends_on: [backend]
```
