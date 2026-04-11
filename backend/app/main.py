"""
XAI Learning Recommendation System — FastAPI Backend
======================================================
Models and explainers are loaded once at startup via lifespan.

Endpoints (Phase 1):
    GET  /health
    POST /predict
    POST /whatif

All remaining endpoints (explain, counterfactual, simulate, history, mlops/*)
are stubbed and will be filled in during Phase 2+.
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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.explainers.shap_explainer import SHAPExplainer
from backend.app.model.lstm_trainer import DropoutLSTM

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
    gbm_model     = None
    rf_model      = None
    lstm_model    = None
    feature_names: list[str] = []
    lstm_config:  dict = {}
    shap_explainer: Optional[SHAPExplainer] = None
    model_version: str = "unknown"


state = AppState()


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models from %s …", MODELS_DIR)

    gbm_path = MODELS_DIR / "gbm.pkl"
    if gbm_path.exists():
        blob = _load_pkl(gbm_path)
        state.gbm_model    = blob["model"]
        state.feature_names = blob["feature_names"]
        state.model_version = "gbm-v1"
        log.info("GBM loaded  features=%d", len(state.feature_names))
    else:
        log.warning("gbm.pkl not found — run trainer.py first")

    rf_path = MODELS_DIR / "rf.pkl"
    if rf_path.exists():
        state.rf_model = _load_pkl(rf_path)["model"]
        log.info("RF loaded")

    lstm_config_path = MODELS_DIR / "lstm_config.json"
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

    if state.gbm_model is not None and state.feature_names:
        state.shap_explainer = SHAPExplainer(
            gbm_model     = state.gbm_model,
            feature_names = state.feature_names,
            lstm_model    = state.lstm_model,
        )
        log.info("SHAPExplainer initialised")

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
def predict(features: LearnerFeatures, model: str = "gbm"):
    _require_model()
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

    return {
        "risk_score":  round(risk_score, 4),
        "risk_label":  _risk_label(risk_score),
        "model_used":  model_used,
    }


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
# Endpoints — Stubs (Phase 2+)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/explain")
def explain(features: LearnerFeatures, model: str = "gbm"):
    _require_model()
    fd = features.model_dump()
    result = state.shap_explainer.explain(fd)
    stability = state.shap_explainer.stability_score(fd)

    resp = {
        "shap_values":  result.shap_values,
        "base_value":   result.base_value,
        "risk_score":   result.risk_score,
        "top_features": result.top_features,
        "stability":    stability,
        "interactions":       None,
        "anchor_rule":        None,
        "prototypes":         None,
        "counterfactual_actions": None,
        "causal_annotations": None,
        "trust_score":        None,
        "narratives":         None,
    }

    if model == "lstm" and state.shap_explainer.deep_explainer is not None:
        week_idx = min(features.current_week_in_course // 2 - 1, len(WEEK_ORDER) - 1)
        X = _features_to_array(features)
        seq = np.zeros((1, len(WEEK_ORDER), len(state.feature_names)), dtype=np.float32)
        for i in range(week_idx + 1):
            seq[0, i] = X[0]
        temporal = state.shap_explainer.explain_temporal(seq)
        resp["temporal_attributions"] = temporal.to_dict()

    return resp


@app.post("/counterfactual")
def counterfactual(features: LearnerFeatures):
    _require_model()
    return {"actions": [], "message": "DiCE counterfactual — Phase 2 pending"}


@app.post("/simulate")
def simulate(features: LearnerFeatures):
    return {"feature_distributions": {}, "outcome_distribution": {}, "message": "Rust MC — Phase 5 pending"}


@app.get("/history/{learner_id}")
def history(learner_id: str):
    return {"learner_id": learner_id, "timeline": [], "drift_flags": [], "message": "Phase 2 pending"}


@app.get("/mlops/health")
def mlops_health():
    return {"model_version": state.model_version, "drift_status": "unknown", "message": "Phase 2A pending"}


@app.get("/mlops/drift-report")
def mlops_drift_report():
    return {"drift_detected": False, "message": "Evidently Phase 2A pending"}
