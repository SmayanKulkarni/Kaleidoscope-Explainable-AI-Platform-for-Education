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

import hmac
import json
import logging
import os
import pickle
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
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
from backend.app.model.lstm_trainer import DropoutLSTM, build_sequences
from backend.app.prescriptor.action_ranker import ActionRanker
from backend.app.auth.auth import get_current_active_student, get_current_user_optional
from backend.app.auth.database import init_db as init_auth_db
from backend.app.auth.models import User
from backend.app.auth.router import router as auth_router
from backend.app.tracker.feedback_store import FeedbackInput, FeedbackStore
from backend.app.tracker.event_store import EventStore
from backend.app.tracker.event_schemas import EventBatchRequest, EventBatchResponse
from backend.app.model.retrain_pipeline import RetrainPipeline
from backend.app.model.hot_reload import hot_reload
from backend.app.model.s3_loader import download_models, upload_models
from backend.app.narrator.llm_narrator import LLMNarrator
from backend.app.tracker.consistency_store import ExplanationStore
from backend.app.tracker.drift_detector import ExplanationDriftDetector
from backend.app.model.temporal_builder import MonteCarloSimulator

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
    feedback_store: Optional[FeedbackStore] = None
    event_store: Optional[EventStore] = None
    llm_narrator: Optional[LLMNarrator] = None
    mc_simulator: Optional[MonteCarloSimulator] = None
    model_version:   str = "unknown"
    canary_gbm_model = None
    canary_fraction: float = 0.0


state = AppState()
MLOPS_CONTROL_LOCK = Lock()


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


DATA_DIR = PROJECT_ROOT / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Download model artifacts from S3 (no-op if AWS_S3_BUCKET not set) ──
    downloaded = download_models(MODELS_DIR)
    if downloaded:
        log.info("S3: downloaded %d model artifact(s)", len(downloaded))

    log.info("Loading models from %s …", MODELS_DIR)

    # ── Load training data (needed by explainers) ──
    train_path = DATA_DIR / "train.pkl"
    if train_path.exists():
        blob = _load_pkl(train_path)
        state.X_train     = blob["X"]
        state.y_train     = blob["y"]
        _ids = blob.get("learner_ids")
        state.learner_ids = np.asarray(_ids, dtype=object) if _ids is not None else None
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

    # ── Background sequences for DeepSHAP (LSTM) ──
    _X_background = None
    _snapshots_path = DATA_DIR / "temporal" / "snapshots.pkl"
    if state.lstm_model is not None and _snapshots_path.exists():
        try:
            _bg_seqs, _, _ = build_sequences(_snapshots_path)
            _X_background  = _bg_seqs[:200]  # subsample — DeepSHAP only needs reference
            log.info("LSTM background sequences loaded  shape=%s", _X_background.shape)
        except Exception as _e:
            log.warning("Failed to load LSTM background sequences: %s", _e)

    # ── Phase 2: Explainers + Evaluators ──
    if state.gbm_model is not None and state.feature_names:
        state.shap_explainer = SHAPExplainer(
            gbm_model     = state.gbm_model,
            feature_names = state.feature_names,
            lstm_model    = state.lstm_model,
            X_background  = _X_background,
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
    state.explanation_store = ExplanationStore()
    state.drift_detector  = ExplanationDriftDetector()
    state.prediction_logger = PredictionLogger()
    state.feedback_store = FeedbackStore()
    log.info("FeedbackStore initialised")

    state.event_store = EventStore()
    log.info("EventStore initialised")

    # ── Phase 2A: Drift Monitor ──
    if state.X_train is not None:
        import pandas as pd
        state.drift_monitor = DriftMonitor(
            reference_data = state.X_train,
            feature_names  = state.feature_names,
        )
        log.info("DriftMonitor initialised")

    # ── Auth DB ──
    init_auth_db()
    log.info("Auth DB initialised (data/auth.db)")

    # ── Phase 3: LLM Narrator ──
    state.llm_narrator = LLMNarrator()
    if state.llm_narrator.available:
        log.info("LLMNarrator initialised  model=%s", state.llm_narrator.model)
    else:
        log.warning("LLMNarrator disabled — set GROQ_API_KEY to enable narration")

    # ── Monte Carlo Simulator ──
    transitions_path = DATA_DIR / "temporal" / "transitions.pkl"
    if transitions_path.exists():
        with open(transitions_path, "rb") as f:
            transitions = pickle.load(f)
        state.mc_simulator = MonteCarloSimulator(transitions)
        log.info("MonteCarloSimulator loaded — %d transition rows", len(transitions.get("deltas", [])))
    else:
        log.warning("transitions.pkl not found — /simulate will return empty distributions")

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

app.include_router(auth_router)


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
    # Latent engagement features — auto-filled by server when model has been retrained
    # with implicit feedback. Default 0.0 = cold-start / pre-retraining baseline.
    engagement_latent_1:        float = Field(default=0.0)
    engagement_latent_2:        float = Field(default=0.0)
    engagement_latent_3:        float = Field(default=0.0)


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


def _build_lstm_sequence(
    history: list[LearnerFeatures],
    current: LearnerFeatures,
    feature_names: list[str],
) -> np.ndarray:
    """
    Build a real (1, T=6, F) sequence for the LSTM from per-week snapshots.

    history  : list of LearnerFeatures from past weeks, oldest first.
               Each must carry the correct current_week_in_course.
    current  : the current-week snapshot (always placed at its week slot).

    Week slots are aligned to WEEK_ORDER = [2,4,6,8,10,12].
    Missing slots stay zero (cold-start / not-yet-observed).
    Falls back to tiling current features if history is empty.
    """
    seq = np.zeros((1, len(WEEK_ORDER), len(feature_names)), dtype=np.float32)

    def _to_vec(f: LearnerFeatures) -> np.ndarray:
        fd = f.model_dump()
        return np.array([fd[n] for n in feature_names], dtype=np.float32)

    if history:
        for snap in history:
            week = snap.current_week_in_course
            if week in WEEK_ORDER:
                slot = WEEK_ORDER.index(week)
                seq[0, slot] = _to_vec(snap)
        # Place current in its slot (overwrites if duplicate)
        curr_week = current.current_week_in_course
        curr_slot = min(
            WEEK_ORDER.index(curr_week) if curr_week in WEEK_ORDER
            else len(WEEK_ORDER) - 1,
            len(WEEK_ORDER) - 1,
        )
        seq[0, curr_slot] = _to_vec(current)
    else:
        # Fallback: tile current snapshot up to the current week slot
        week_idx = min(
            max(current.current_week_in_course // 2 - 1, 0),
            len(WEEK_ORDER) - 1,
        )
        vec = _to_vec(current)
        for i in range(week_idx + 1):
            seq[0, i] = vec

    return seq


def _require_model():
    if state.gbm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run trainer.py first")
    if state.shap_explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not initialised")


def require_mlops_operator(
    current_user: Optional[User] = Depends(get_current_user_optional),
    x_mlops_token: Optional[str] = Header(default=None),
) -> str:
    """
    Guard for control-plane MLOps operations.

    Access is granted if either:
    - caller is an authenticated admin user, or
    - caller provides a valid automation token in X-MLOPS-Token header.
    """
    if current_user is not None:
        if current_user.role == "admin":
            return f"admin:{current_user.username}"
        raise HTTPException(status_code=403, detail="Admin access required")

    automation_token = os.getenv("MLOPS_AUTOMATION_TOKEN", "")
    if automation_token and x_mlops_token and hmac.compare_digest(x_mlops_token, automation_token):
        return "automation-token"

    raise HTTPException(status_code=401, detail="Unauthorized MLOps operation")


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
        seq    = _build_lstm_sequence([], features, state.feature_names)
        proba  = state.lstm_model.predict_proba(seq)
        risk_score = float(proba[0, 1])
        model_used = "lstm"
    elif (
        state.canary_gbm_model is not None
        and state.canary_fraction > 0.0
        and random.random() < state.canary_fraction
    ):
        risk_score = float(state.canary_gbm_model.predict_proba(X)[0][1])
        model_used = "gbm-canary"
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


class CompareRequest(BaseModel):
    features:   LearnerFeatures
    history:    list[LearnerFeatures] = []   # past weekly snapshots, oldest-first
    learner_id: str = "anonymous"


@app.post("/compare")
def compare_models(req: CompareRequest):
    """
    Side-by-side GBM vs LSTM comparison with cross-model disagreement analysis.

    Always returns GBM results. LSTM sections are omitted gracefully if the
    LSTM or DeepSHAP explainer is not loaded.
    """
    _require_model()
    fd = req.features.model_dump()
    X  = _features_to_array(req.features)

    # ── GBM ──────────────────────────────────────────────────────────────────
    gbm_score  = round(float(state.gbm_model.predict_proba(X)[0][1]), 4)
    shap_result = state.shap_explainer.explain(fd)
    gbm_top3   = list(shap_result.top_features[:3])

    # ── LSTM ─────────────────────────────────────────────────────────────────
    lstm_score:              Optional[float] = None
    lstm_risk_label:         Optional[str]   = None
    lstm_temporal_dict:      Optional[dict]  = None
    lstm_top3_temporal:      list[str]       = []

    if state.lstm_model is not None:
        seq        = _build_lstm_sequence(req.history, req.features, state.feature_names)
        lstm_score = round(float(state.lstm_model.predict_proba(seq)[0][1]), 4)
        lstm_risk_label = _risk_label(lstm_score)

        if state.shap_explainer.deep_explainer is not None:
            temporal_result  = state.shap_explainer.explain_temporal(seq)
            lstm_temporal_dict = temporal_result.to_dict()

            # Find highest-magnitude week → extract its top-3 features
            try:
                attr_array = np.array(temporal_result.temporal_attributions)  # (T, F)
                week_magnitudes = np.abs(attr_array).sum(axis=1)      # (T,)
                best_week_idx   = int(np.argmax(week_magnitudes))
                top_indices     = np.argsort(np.abs(attr_array[best_week_idx]))[::-1][:3]
                lstm_top3_temporal = [
                    state.feature_names[i]
                    for i in top_indices
                    if i < len(state.feature_names)
                ]
            except Exception:
                lstm_top3_temporal = []

    # ── Disagreement analysis ─────────────────────────────────────────────────
    score_delta       = round(abs(gbm_score - (lstm_score or gbm_score)), 4)
    disagreement_flag = score_delta > 0.15
    feature_disagreement = (
        [f for f in gbm_top3 if f not in lstm_top3_temporal]
        if lstm_top3_temporal else []
    )

    # ── Interpretation ────────────────────────────────────────────────────────
    if lstm_score is None:
        interpretation = "LSTM not loaded — GBM aggregate risk only."
    elif disagreement_flag and lstm_score > gbm_score:
        interpretation = (
            "GBM sees moderate aggregate risk; LSTM detected a recent decline pattern "
            "that elevates the temporal risk estimate."
        )
    elif disagreement_flag and gbm_score > lstm_score:
        interpretation = (
            "GBM captures higher aggregate risk; LSTM suggests recent behaviour is improving, "
            "reducing the temporal risk estimate."
        )
    elif feature_disagreement:
        interpretation = (
            f"Models agree on overall risk but diverge on drivers: "
            f"{feature_disagreement} are salient for GBM but not the most active recent week."
        )
    else:
        interpretation = (
            "GBM and LSTM agree on both risk level and key drivers — high model consistency."
        )

    return {
        "learner_id":               req.learner_id,
        "gbm_score":                gbm_score,
        "gbm_risk_label":           _risk_label(gbm_score),
        "gbm_top3":                 gbm_top3,
        "lstm_score":               lstm_score,
        "lstm_risk_label":          lstm_risk_label,
        "lstm_temporal_attributions": lstm_temporal_dict,
        "score_delta":              score_delta,
        "disagreement_flag":        disagreement_flag,
        "feature_disagreement":     feature_disagreement,
        "interpretation":           interpretation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints — Phase 2: Full XAI Engine
# ──────────────────────────────────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    features:   LearnerFeatures
    learner_id: str = "anonymous"
    model:      str = "gbm"
    audience:   str = "both"  # "learner" | "instructor" | "both"
    history:    list[LearnerFeatures] = []  # past weekly snapshots (week 2,4,6...) oldest-first


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
        seq      = _build_lstm_sequence(req.history, req.features, state.feature_names)
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

    # Phase 3: LLM Narration — narrates ONLY pre-computed data above
    if state.llm_narrator is not None:
        narration = state.llm_narrator.narrate(
            explain_resp=resp,
            learner_id=req.learner_id,
            audience=req.audience,
        )
        resp["narratives"] = narration.to_dict()
    else:
        resp["narratives"] = None
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


class SimulateRequest(BaseModel):
    features:      LearnerFeatures
    current_week:  int = Field(default=6,    ge=1, le=52)
    target_week:   int = Field(default=12,   ge=2, le=52)
    n_simulations: int = Field(default=1000, ge=10, le=10_000)


@app.post("/simulate")
def simulate(req: SimulateRequest):
    if state.mc_simulator is None:
        return {
            "feature_distributions": {},
            "outcome_distribution": {},
            "current_week":  req.current_week,
            "target_week":   req.target_week,
            "n_simulations": req.n_simulations,
            "message": "transitions.pkl not found — run temporal_builder.py first",
        }

    if req.current_week >= req.target_week:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="target_week must be greater than current_week")

    current_features = req.features.model_dump(
        exclude={"engagement_latent_1", "engagement_latent_2", "engagement_latent_3"}
    )

    result = state.mc_simulator.simulate(
        current_features=current_features,
        current_week=req.current_week,
        target_week=req.target_week,
        n_simulations=req.n_simulations,
        model=state.gbm_model,
    )
    # Strip individual trajectories from the response (too large)
    result.pop("trajectories", None)
    return result


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


# ──────────────────────────────────────────────────────────────────────────────
# Implicit feedback collection
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/events", response_model=EventBatchResponse)
def record_events(req: EventBatchRequest):
    """
    Batch event ingestion endpoint.
    Accepts up to 500 interaction events per call.
    Events are persisted to data/events.db for retraining aggregation.
    """
    if state.event_store is None:
        raise HTTPException(503, "Event store not initialised")
    n = state.event_store.record_batch([e.model_dump() for e in req.events])
    return EventBatchResponse(recorded=n, message="ok")


@app.get("/events/{learner_id}")
def get_learner_events(learner_id: str, limit: int = 100):
    """Return recent interaction events for a learner (debug/audit use)."""
    if state.event_store is None:
        raise HTTPException(503, "Event store not initialised")
    events = state.event_store.get_learner_events(learner_id)
    return {
        "learner_id": learner_id,
        "count":      len(events),
        "events":     events[-limit:],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Retrain + hot-reload endpoints (admin-only)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/mlops/retrain")
def trigger_retrain(min_events: int = 0, operator: str = Depends(require_mlops_operator)):
    """
    Trigger a full implicit+explicit feedback retraining cycle.
    Steps:
      1. Aggregate events → engagement signals
      2. Train/update engagement autoencoder
      3. Augment OULAD features with 3 latent scores
      4. Retrain GBM + RF (15 features)
      5. Validate against regression gates
      6. Save artifacts if validated

    After success, call POST /mlops/reload to hot-swap the live model.

    Query param `min_events`: override the minimum event count guard (default 0 = no guard).
    """
    if not MLOPS_CONTROL_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another MLOps control operation is already running")
    try:
        if state.event_store is None or state.feedback_store is None:
            raise HTTPException(503, "Event store or feedback store not initialised")

        if min_events > 0 and not (state.event_store.count() >= min_events):
            return {
                "queued":  False,
                "reason":  f"Only {state.event_store.count()} events recorded (need ≥{min_events})",
                "n_events": state.event_store.count(),
            }

        pipeline = RetrainPipeline(
            event_store    = state.event_store,
            feedback_store = state.feedback_store,
            data_dir       = DATA_DIR,
            models_dir     = MODELS_DIR,
        )

        log.info("POST /mlops/retrain triggered by %s", operator)
        result = pipeline.run(trigger="api")
        if result.success:
            uploaded = upload_models(MODELS_DIR)
            log.info("S3: uploaded %d artifact(s) after retrain", len(uploaded))
        return result.to_dict()
    finally:
        MLOPS_CONTROL_LOCK.release()


@app.post("/mlops/reload")
def trigger_hot_reload(
    operator: str = Depends(require_mlops_operator),
    canary_fraction: float = Query(1.0, ge=0.0, le=1.0,
        description="Fraction of /predict traffic routed to the new model. "
                    "0.0=abort canary, (0,1)=partial canary, 1.0=full swap (default)."),
):
    """
    Hot-reload models and all explainers from disk into the running process.
    Call this after POST /mlops/retrain returns success=true.

    - canary_fraction=1.0 (default): full atomic swap, all traffic to new model.
    - canary_fraction=0.2: load new model as canary, serve 20% of /predict calls.
    - canary_fraction=0.0: abort active canary, revert to production.
    """
    if not MLOPS_CONTROL_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another MLOps control operation is already running")
    try:
        log.info("POST /mlops/reload triggered by %s  canary_fraction=%.2f", operator, canary_fraction)
        result = hot_reload(state, DATA_DIR, MODELS_DIR, DEVICE, canary_fraction=canary_fraction)
        if not result.success:
            raise HTTPException(500, detail=result.to_dict())
        return result.to_dict()
    finally:
        MLOPS_CONTROL_LOCK.release()


class FeedbackRequest(BaseModel):
    learner_id:              str
    explanation_id:          Optional[str]  = None
    rating:                  Optional[int]  = Field(default=None, ge=1, le=5)
    followed_recommendation: Optional[bool] = None
    top_action_feature:      Optional[str]  = None
    correction_feature:      Optional[str]  = None
    correction_comment:      Optional[str]  = None
    audience:                Optional[str]  = None


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Submit a user rating, follow-flag, or feature correction for an explanation."""
    if state.feedback_store is None:
        raise HTTPException(503, "Feedback store not initialised")

    # Pull trust/risk scores from latest explanation for correlation tracking
    trust_score: Optional[float] = None
    risk_score:  Optional[float] = None
    if state.explanation_store is not None:
        latest = state.explanation_store.get_latest(req.learner_id)
        if latest:
            trust_score = latest.get("trust_score")
            risk_score  = latest.get("risk_score")

    record_id = state.feedback_store.record(FeedbackInput(
        learner_id              = req.learner_id,
        explanation_id          = req.explanation_id,
        rating                  = req.rating,
        followed_recommendation = req.followed_recommendation,
        top_action_feature      = req.top_action_feature,
        correction_feature      = req.correction_feature,
        correction_comment      = req.correction_comment,
        trust_score_at_time     = trust_score,
        risk_score_at_time      = risk_score,
        audience                = req.audience,
    ))
    return {"recorded": True, "feedback_id": record_id, "learner_id": req.learner_id}


@app.get("/feedback/stats")
def feedback_stats():
    """Aggregate feedback stats: avg rating, follow rates, top corrected features."""
    if state.feedback_store is None:
        raise HTTPException(503, "Feedback store not initialised")
    return state.feedback_store.get_stats().to_dict()


@app.get("/feedback/{learner_id}")
def learner_feedback(learner_id: str):
    """All feedback records for a specific learner."""
    if state.feedback_store is None:
        raise HTTPException(503, "Feedback store not initialised")
    return {
        "learner_id": learner_id,
        "records":    state.feedback_store.get_learner_feedback(learner_id),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Authenticated self-service endpoints (JWT required)
# ──────────────────────────────────────────────────────────────────────────────

class SelfExplainRequest(BaseModel):
    """Student self-service: learner_id auto-filled from JWT token."""
    audience: str = "learner"   # students default to learner view


@app.post("/explain/me")
def explain_me(
    req: SelfExplainRequest,
    current_user: User = Depends(get_current_active_student),
):
    """
    Authenticated student endpoint.
    Pulls the learner's current features from their profile and runs /explain.
    Returns learner-audience narrative only.

    The learner_id is taken from the JWT token — students cannot explain
    other learners' data.
    """
    _require_model()

    if current_user.learner_profile is None:
        raise HTTPException(404, "No learner profile found for this account")

    lp = current_user.learner_profile
    learner_id = lp.learner_id

    # Build feature dict from stored profile context
    # In production these come from the latest OULAD sync; here we pull the
    # most recent explanation record's feature snapshot as a fallback.
    latest = None
    if state.explanation_store is not None:
        latest = state.explanation_store.get_latest(learner_id)

    if latest is None:
        raise HTTPException(
            404,
            "No explanation history found for this learner. "
            "POST /explain first with your current features."
        )

    # Re-run the full explain pipeline on the last recorded SHAP values
    # (features are reconstructed from the explanation store)
    return {
        "learner_id":    learner_id,
        "current_week":  lp.current_week,
        "course_id":     lp.course_id,
        "latest_risk":   latest.get("risk_score"),
        "top3_features": latest.get("top3_features", []),
        "trust_score":   latest.get("trust_score"),
        "history_url":   f"/history/{learner_id}",
        "note":          "Call POST /explain with your current features to refresh the analysis.",
    }


@app.get("/explain/me/history")
def my_history(
    current_user: User = Depends(get_current_active_student),
):
    """Authenticated student endpoint: fetch own explanation timeline."""
    if current_user.learner_profile is None:
        raise HTTPException(404, "No learner profile found")
    learner_id = current_user.learner_profile.learner_id
    if state.explanation_store is None:
        return {"learner_id": learner_id, "timeline": [], "drift_flags": []}
    timeline    = state.explanation_store.get_top3_timeline(learner_id)
    full_history = state.explanation_store.get_history(learner_id, n=10)
    drift_flags = []
    if state.drift_detector is not None:
        drift = state.drift_detector.check_learner_drift(learner_id, state.explanation_store)
        if drift:
            drift_flags.append(drift.to_dict())
    return {
        "learner_id":  learner_id,
        "timeline":    timeline,
        "history":     full_history,
        "drift_flags": drift_flags,
    }
