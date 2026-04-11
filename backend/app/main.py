"""
XAI Learning Recommendation System — FastAPI Backend
======================================================
Models, explainers, and MLOps modules are loaded once at startup via lifespan.

Endpoints:
    GET  /health
    POST /predict
    POST /explain
    POST /whatif
    POST /counterfactual
    POST /simulate
    GET  /history/{learner_id}
    GET  /mlops/health
    GET  /mlops/drift-report
    GET  /mlops/metrics
    POST /feedback
"""

from __future__ import annotations

import json
import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.causal.causal_annotator import CausalAnnotator
from backend.app.evaluator.trust_scorer import TrustScorer
from backend.app.evaluator.uncertainty_estimator import UncertaintyEstimator
from backend.app.explainers.anchors_explainer import AnchorsExplainer
from backend.app.explainers.archipelago import ArchipelagoExplainer
from backend.app.explainers.dice_explainer import DiCEExplainer
from backend.app.explainers.prototype_explainer import PrototypeExplainer
from backend.app.explainers.shap_explainer import SHAPExplainer
from backend.app.mlops.drift_monitor import DriftMonitor
from backend.app.mlops.prediction_logger import PredictionLogger
from backend.app.model.lstm_trainer import DropoutLSTM
from backend.app.prescriptor.action_ranker import ActionRanker
from backend.app.tracker.consistency_store import ExplanationStore
from backend.app.tracker.drift_detector import ExplanationDriftDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "models"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEEK_ORDER = [2, 4, 6, 8, 10, 12]


# ──────────────────────────────────────────────────────────────────────────────
# Global state (loaded once at startup)
# ──────────────────────────────────────────────────────────────────────────────

class AppState:
    gbm_model       = None
    rf_model        = None
    lstm_model      = None
    feature_names:  list[str] = []
    X_train         = None
    y_train         = None
    learner_ids     = None
    lstm_config:    dict = {}
    shap_explainer: Optional[SHAPExplainer] = None
    dice_explainer: Optional[DiCEExplainer] = None
    anchors_explainer: Optional[AnchorsExplainer] = None
    prototype_explainer: Optional[PrototypeExplainer] = None
    archipelago_explainer: Optional[ArchipelagoExplainer] = None
    uncertainty_estimator: Optional[UncertaintyEstimator] = None
    causal_annotator: Optional[CausalAnnotator] = None
    action_ranker: Optional[ActionRanker] = None
    trust_scorer: Optional[TrustScorer] = None
    explanation_store: Optional[ExplanationStore] = None
    drift_detector: Optional[ExplanationDriftDetector] = None
    prediction_logger: Optional[PredictionLogger] = None
    drift_monitor: Optional[DriftMonitor] = None
    model_version:  str = "unknown"


state = AppState()


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


DATA_DIR = PROJECT_ROOT / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models from %s …", MODELS_DIR)

    # ── Load training data (needed by explainers) ──
    train_path = DATA_DIR / "train.pkl"
    if train_path.exists():
        blob = _load_pkl(train_path)
        state.X_train     = blob["X"]
        state.y_train     = blob["y"]
        state.learner_ids = blob.get("learner_ids")
        log.info("Training data loaded  X=%s", state.X_train.shape)

    # ── GBM ──
    gbm_path = MODELS_DIR / "gbm.pkl"
    if gbm_path.exists():
        blob = _load_pkl(gbm_path)
        state.gbm_model     = blob["model"]
        state.feature_names = blob["feature_names"]
        state.model_version = "gbm-v1"
        log.info("GBM loaded  features=%d", len(state.feature_names))
    else:
        log.warning("gbm.pkl not found — run trainer.py first")

    # ── RF ──
    rf_path = MODELS_DIR / "rf.pkl"
    if rf_path.exists():
        state.rf_model = _load_pkl(rf_path)["model"]
        log.info("RF loaded")

    # ── LSTM ──
    lstm_config_path  = MODELS_DIR / "lstm_config.json"
    lstm_weights_path = MODELS_DIR / "lstm.pt"
    if lstm_config_path.exists() and lstm_weights_path.exists():
        with open(lstm_config_path) as f:
            state.lstm_config = json.load(f)
        lstm = DropoutLSTM(
            n_features = state.lstm_config["n_features"],
            hidden_dim = state.lstm_config["hidden_dim"],
            n_layers   = state.lstm_config["n_layers"],
            dropout    = state.lstm_config["dropout"],
        ).to(DEVICE)
        sd = torch.load(lstm_weights_path, map_location=DEVICE, weights_only=True)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        lstm.load_state_dict(sd)
        lstm.eval()
        state.lstm_model = lstm
        log.info("LSTM loaded  val_auc=%.4f", state.lstm_config.get("best_val_auc", 0))

    # ── Phase 2: Explainers + Evaluators ──
    if state.gbm_model is not None and state.feature_names:
        state.shap_explainer = SHAPExplainer(
            gbm_model     = state.gbm_model,
            feature_names = state.feature_names,
            lstm_model    = state.lstm_model,
        )
        log.info("SHAPExplainer initialised")

        state.archipelago_explainer = ArchipelagoExplainer(
            tree_explainer = state.shap_explainer.tree_explainer,
            feature_names  = state.feature_names,
        )
        log.info("ArchipelagoExplainer initialised")

    if state.gbm_model is not None and state.X_train is not None:
        state.dice_explainer = DiCEExplainer(
            model         = state.gbm_model,
            X_train       = state.X_train,
            feature_names = state.feature_names,
            y_train       = state.y_train,
        )
        log.info("DiCEExplainer initialised")

        def _predict_fn(X):
            return (state.gbm_model.predict_proba(X)[:, 1] >= 0.5).astype(int)

        state.anchors_explainer = AnchorsExplainer(
            predict_fn    = _predict_fn,
            X_train       = state.X_train,
            feature_names = state.feature_names,
        )
        log.info("AnchorsExplainer initialised")

        state.prototype_explainer = PrototypeExplainer(
            X_train       = state.X_train,
            y_train       = state.y_train,
            learner_ids   = state.learner_ids if state.learner_ids is not None else np.arange(len(state.X_train)),
            feature_names = state.feature_names,
        )
        log.info("PrototypeExplainer initialised")

        state.uncertainty_estimator = UncertaintyEstimator(
            model = state.gbm_model,
            X_cal = state.X_train,
            y_cal = state.y_train,
        )
        log.info("UncertaintyEstimator initialised")

        state.causal_annotator = CausalAnnotator(
            X_train       = state.X_train,
            y_train       = state.y_train,
            feature_names = state.feature_names,
        )
        log.info("CausalAnnotator initialised (DoWhy)")

        state.action_ranker = ActionRanker(
            feature_names    = state.feature_names,
            X_train          = state.X_train,
            causal_annotator = state.causal_annotator,
        )
        log.info("ActionRanker initialised")

    state.trust_scorer    = TrustScorer()
    state.explanation_store = ExplanationStore(
        db_url=f"sqlite:///{DATA_DIR / 'explanations.db'}"
    )
    state.drift_detector  = ExplanationDriftDetector()
    state.prediction_logger = PredictionLogger(
        db_url=f"sqlite:///{DATA_DIR / 'predictions.db'}"
    )

    # ── Phase 2A: Drift Monitor ──
    if state.X_train is not None:
        import pandas as pd
        state.drift_monitor = DriftMonitor(
            reference_data = state.X_train,
            feature_names  = state.feature_names,
        )
        log.info("DriftMonitor initialised")

    log.info("Startup complete — device=%s", DEVICE)
    yield
    log.info("Shutdown")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="XAI Learning Recommendation System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────

class LearnerFeatures(BaseModel):
    login_frequency_weekly:     float = Field(ge=0,   le=14)
    avg_session_duration_min:   float = Field(ge=0,   le=300)
    forum_posts_count:          int   = Field(ge=0)
    video_completion_rate:      float = Field(ge=0.0, le=1.0)
    quiz_avg_score:             float = Field(ge=0.0, le=100.0)
    quiz_completion_rate:       float = Field(ge=0.0, le=1.0)
    assignment_submission_rate: float = Field(ge=0.0, le=1.0)
    days_since_last_activity:   int   = Field(ge=0)
    prior_course_completions:   int   = Field(ge=0)
    current_week_in_course:     int   = Field(ge=1,   le=52)
    missed_deadlines_count:     int   = Field(ge=0)
    help_requests_count:        int   = Field(ge=0)


class WhatIfRequest(BaseModel):
    features: LearnerFeatures
    overrides: dict[str, float]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _features_to_array(features: LearnerFeatures) -> np.ndarray:
    fd = features.model_dump()
    return np.array([fd[f] for f in state.feature_names], dtype=np.float32).reshape(1, -1)


def _risk_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    return "low"


def _require_model():
    if state.gbm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run trainer.py first")
    if state.shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not initialised")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints — Phase 1
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name":         torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
            "vram_free_mb":  round((torch.cuda.get_device_properties(0).total_memory
                                    - torch.cuda.memory_allocated(0)) / 1e6),
        }
    return {
        "status":        "ok",
        "model_version": state.model_version,
        "gbm_loaded":    state.gbm_model is not None,
        "lstm_loaded":   state.lstm_model is not None,
        "device":        str(DEVICE),
        "gpu":           gpu_info,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict")
def predict(features: LearnerFeatures, bg: BackgroundTasks, model: str = "gbm"):
    _require_model()
    fd = features.model_dump()
    X = _features_to_array(features)

    if model == "lstm" and state.lstm_model is not None:
        seq = np.zeros((1, len(WEEK_ORDER), len(state.feature_names)), dtype=np.float32)
        week_idx = min(features.current_week_in_course // 2 - 1, len(WEEK_ORDER) - 1)
        for i in range(week_idx + 1):
            seq[0, i] = X[0]
        proba = state.lstm_model.predict_proba(seq)
        risk_score = float(proba[0, 1])
        model_used = "lstm"
    else:
        risk_score = float(state.gbm_model.predict_proba(X)[0][1])
        model_used = "gbm"

    resp = {
        "risk_score":  round(risk_score, 4),
        "risk_label":  _risk_label(risk_score),
        "model_used":  model_used,
    }

    # Uncertainty estimation (MAPIE)
    if state.uncertainty_estimator is not None and model_used == "gbm":
        unc = state.uncertainty_estimator.predict_with_uncertainty(X)
        resp["uncertainty"] = unc.to_dict()

    # Background: log prediction
    if state.prediction_logger is not None:
        bg.add_task(
            state.prediction_logger.log,
            features=fd, risk_score=risk_score,
            model_version=state.model_version, model_used=model_used,
        )

    return resp


@app.post("/whatif")
def whatif(req: WhatIfRequest):
    _require_model()
    fd = req.features.model_dump()
    fd.update(req.overrides)

    result = state.shap_explainer.explain_whatif(fd)
    base_risk = float(state.gbm_model.predict_proba(
        _features_to_array(req.features)
    )[0][1])

    return {
        "shap_values":  result.shap_values,
        "base_value":   result.base_value,
        "risk_score":   result.risk_score,
        "risk_delta":   round(result.risk_score - base_risk, 4),
        "top_features": result.top_features,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints — Phase 2: Full XAI Engine
# ──────────────────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    features:   LearnerFeatures
    learner_id: str = "anonymous"
    model:      str = "gbm"


@app.post("/explain")
def explain(req: ExplainRequest):
    _require_model()
    fd = req.features.model_dump()
    X  = _features_to_array(req.features)

    # Core SHAP
    shap_result = state.shap_explainer.explain(fd)
    stability   = state.shap_explainer.stability_score(fd)
    risk_score  = shap_result.risk_score

    resp = {
        "risk_score":   round(risk_score, 4),
        "risk_label":   _risk_label(risk_score),
        "shap_values":  shap_result.shap_values,
        "base_value":   shap_result.base_value,
        "top_features": shap_result.top_features,
        "stability":    stability,
    }

    # Interactions (Archipelago)
    if state.archipelago_explainer is not None:
        interactions = state.archipelago_explainer.get_interactions(fd)
        resp["interactions"] = [ix.to_dict() for ix in interactions]
        resp["interaction_narrative"] = state.archipelago_explainer.to_narrative(interactions)

    # Anchor rule
    if state.anchors_explainer is not None:
        try:
            anchor = state.anchors_explainer.explain(fd)
            resp["anchor_rule"] = anchor.to_dict()
        except Exception as e:
            log.warning("Anchors failed: %s", e)
            resp["anchor_rule"] = None

    # Prototypes
    if state.prototype_explainer is not None:
        proto = state.prototype_explainer.explain(fd)
        resp["prototypes"] = proto.to_dict()

    # Counterfactual actions
    if state.dice_explainer is not None:
        cf_result = state.dice_explainer.get_counterfactuals(fd)
        resp["counterfactual"] = cf_result.to_dict()
        # Rank actions with causal + actionability weights
        if state.action_ranker is not None and cf_result.actions:
            ranked = state.action_ranker.rank(
                dice_actions=cf_result.actions,
                shap_values=shap_result.shap_values,
                base_risk=risk_score,
            )
            resp["ranked_actions"] = [a.to_dict() for a in ranked]

    # Causal annotations
    if state.causal_annotator is not None:
        annotated = state.causal_annotator.annotate_shap(shap_result.shap_values)
        resp["causal_annotations"] = [a.to_dict() for a in annotated]

    # Trust score
    if state.trust_scorer is not None:
        trust = state.trust_scorer.score(
            shap_values=shap_result.shap_values,
            base_value=shap_result.base_value,
            risk_score=risk_score,
            stability=stability,
        )
        resp["trust_score"] = trust.to_dict()

    # Uncertainty
    if state.uncertainty_estimator is not None:
        unc = state.uncertainty_estimator.predict_with_uncertainty(X)
        resp["uncertainty"] = unc.to_dict()

    # LSTM temporal attributions
    if req.model == "lstm" and state.shap_explainer.deep_explainer is not None:
        week_idx = min(req.features.current_week_in_course // 2 - 1, len(WEEK_ORDER) - 1)
        seq = np.zeros((1, len(WEEK_ORDER), len(state.feature_names)), dtype=np.float32)
        for i in range(week_idx + 1):
            seq[0, i] = X[0]
        temporal = state.shap_explainer.explain_temporal(seq)
        resp["temporal_attributions"] = temporal.to_dict()

    # Persist to explanation store + drift check
    if state.explanation_store is not None:
        top3 = shap_result.top_features[:3] if shap_result.top_features else []
        state.explanation_store.save(
            learner_id=req.learner_id,
            risk_score=risk_score,
            shap_values=shap_result.shap_values,
            top3_features=top3,
            trust_score=resp.get("trust_score", {}).get("trust_score"),
            model_used=req.model,
            anchor_rule=resp.get("anchor_rule", {}).get("anchor_rule") if resp.get("anchor_rule") else None,
        )
        if state.drift_detector is not None:
            drift = state.drift_detector.check_learner_drift(req.learner_id, state.explanation_store)
            resp["explanation_drift"] = drift.to_dict() if drift else None

    resp["narratives"] = None  # Placeholder — Phase 3 (Groq narrator)
    return resp


@app.post("/counterfactual")
def counterfactual(features: LearnerFeatures):
    _require_model()
    if state.dice_explainer is None:
        raise HTTPException(503, "DiCE explainer not initialised")
    fd = features.model_dump()
    cf_result = state.dice_explainer.get_counterfactuals(fd)

    resp = cf_result.to_dict()
    if state.action_ranker is not None and cf_result.actions:
        shap_result = state.shap_explainer.explain(fd)
        ranked = state.action_ranker.rank(
            dice_actions=cf_result.actions,
            shap_values=shap_result.shap_values,
            base_risk=shap_result.risk_score,
        )
        resp["ranked_actions"] = [a.to_dict() for a in ranked]
    return resp


@app.post("/simulate")
def simulate(features: LearnerFeatures):
    return {"feature_distributions": {}, "outcome_distribution": {}, "message": "Rust MC — Phase 5 pending"}


@app.get("/history/{learner_id}")
def history(learner_id: str):
    if state.explanation_store is None:
        return {"learner_id": learner_id, "timeline": [], "drift_flags": []}

    timeline = state.explanation_store.get_top3_timeline(learner_id)
    full_history = state.explanation_store.get_history(learner_id, n=10)

    drift_flags = []
    if state.drift_detector is not None:
        drift = state.drift_detector.check_learner_drift(learner_id, state.explanation_store)
        if drift is not None:
            drift_flags.append(drift.to_dict())

    return {
        "learner_id":   learner_id,
        "timeline":     timeline,
        "history":      full_history,
        "drift_flags":  drift_flags,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints — Phase 2A: MLOps
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/mlops/health")
def mlops_health():
    drift_status = "unknown"
    if state.drift_monitor is not None:
        cached = state.drift_monitor.get_cached_report()
        if cached is not None:
            drift_status = "drift_detected" if cached.dataset_drift else "ok"
    return {
        "status":        "ok",
        "model_version": state.model_version,
        "drift_status":  drift_status,
        "prediction_count": state.prediction_logger.count() if state.prediction_logger else 0,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


@app.get("/mlops/drift-report")
def mlops_drift_report():
    if state.drift_monitor is None:
        raise HTTPException(503, "Drift monitor not initialised")

    # Try cached report first
    cached = state.drift_monitor.get_cached_report()
    if cached is not None:
        return cached.to_dict()

    # Run fresh report from recent predictions
    if state.prediction_logger is None:
        raise HTTPException(503, "Prediction logger not initialised")

    current = state.prediction_logger.get_feature_matrix(n=500)
    if current.empty:
        return {"dataset_drift": False, "n_drifted_features": 0,
                "message": "Not enough predictions logged yet"}

    report = state.drift_monitor.check_drift(current)
    return report.to_dict()


@app.get("/mlops/metrics")
def mlops_metrics():
    """Current production model metrics from training summary."""
    summary_path = MODELS_DIR / "training_summary.json"
    tuning_path  = MODELS_DIR / "tuning_summary.json"
    result = {}
    if summary_path.exists():
        with open(summary_path) as f:
            result["training"] = json.load(f)
    if tuning_path.exists():
        with open(tuning_path) as f:
            result["tuning"] = json.load(f)
    result["model_version"] = state.model_version
    return result


class FeedbackRequest(BaseModel):
    learner_id: str
    actual_outcome: int = Field(ge=0, le=1)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Log ground-truth outcome for concept drift analysis."""
    if state.explanation_store is None:
        raise HTTPException(503, "Explanation store not initialised")
    latest = state.explanation_store.get_latest(req.learner_id)
    if latest is None:
        raise HTTPException(404, f"No explanation found for {req.learner_id}")
    return {
        "learner_id":     req.learner_id,
        "actual_outcome": req.actual_outcome,
        "predicted_risk": latest["risk_score"],
        "error":          abs(latest["risk_score"] - req.actual_outcome),
        "recorded":       True,
    }
