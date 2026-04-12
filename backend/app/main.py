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

import asyncio
import hmac
import json
import logging
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

# Load .env file for GROQ_API_KEY and other secrets
try:
    from dotenv import load_dotenv
    # Load from project root (parents[2] from backend/app/main.py)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        log_env = logging.getLogger(__name__)
        log_env.info("Loaded .env from %s", env_path)
    else:
        load_dotenv()  # fallback to cwd
except ImportError:
    pass  # python-dotenv not installed, rely on actual env vars

import numpy as np
import pandas as pd
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
from backend.app.auth.auth import get_current_active_student, get_current_user, get_current_user_optional
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
from backend.app.recommender.ranker_explainer import RankerExplainer
from backend.app.recommender.fairness_auditor import FairnessAuditor
from backend.app.tracker.reco_consistency_store import RecommendationExplanationStore

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
    student_ranker_explainer: Optional[RankerExplainer] = None
    instructor_ranker_explainer: Optional[RankerExplainer] = None
    # Startup readiness flags
    core_ready:         bool = False   # /predict and /health available
    fully_initialized:  bool = False   # all XAI explainers ready
    reco_explanation_store: Optional[RecommendationExplanationStore] = None
    fairness_auditor: Optional[FairnessAuditor] = None


state = AppState()
MLOPS_CONTROL_LOCK = Lock()


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


DATA_DIR = PROJECT_ROOT / "data"


# ──────────────────────────────────────────────────────────────────────────────
# Demo user seeder — runs at every startup to ensure demo accounts exist
# ──────────────────────────────────────────────────────────────────────────────

_DEMO_USERS = [
    {"username": "demo_admin",      "password": "admin123",      "email": "demo_admin@xai.local",      "role": "admin",      "learner_id": None},
    {"username": "demo_student",    "password": "student123",    "email": "demo_student@xai.local",    "role": "student",    "learner_id": "learner_001"},
    {"username": "demo_instructor", "password": "instructor123", "email": "demo_instructor@xai.local", "role": "instructor", "learner_id": None},
]

# 6 demo roster students enrolled under demo_instructor (diverse risk profiles)
_DEMO_ROSTER = [
    {
        "username": "roster_alice",   "password": "roster123", "email": "alice@xai.local",
        "full_name": "Alice Chen",    "learner_id": "learner_r01", "course_id": "CCC",
        "module": "CCC", "presentation": "2014J",
        "features": dict(login_frequency_weekly=6, avg_session_duration_min=72, forum_posts_count=8,
                         video_completion_rate=0.92, quiz_avg_score=84, quiz_completion_rate=0.9,
                         assignment_submission_rate=0.95, days_since_last_activity=1,
                         prior_course_completions=3, current_week_in_course=6,
                         missed_deadlines_count=0, help_requests_count=2),
        "risk": 0.12, "trajectory": "improving",
    },
    {
        "username": "roster_bob",     "password": "roster123", "email": "bob@xai.local",
        "full_name": "Bob Martinez",  "learner_id": "learner_r02", "course_id": "CCC",
        "module": "CCC", "presentation": "2014B",
        "features": dict(login_frequency_weekly=3, avg_session_duration_min=45, forum_posts_count=2,
                         video_completion_rate=0.6, quiz_avg_score=62, quiz_completion_rate=0.65,
                         assignment_submission_rate=0.7, days_since_last_activity=6,
                         prior_course_completions=1, current_week_in_course=6,
                         missed_deadlines_count=2, help_requests_count=3),
        "risk": 0.41, "trajectory": "stable",
    },
    {
        "username": "roster_carla",   "password": "roster123", "email": "carla@xai.local",
        "full_name": "Carla Singh",   "learner_id": "learner_r03", "course_id": "CCC",
        "module": "CCC", "presentation": "2013J",
        "features": dict(login_frequency_weekly=1, avg_session_duration_min=20, forum_posts_count=0,
                         video_completion_rate=0.3, quiz_avg_score=42, quiz_completion_rate=0.35,
                         assignment_submission_rate=0.4, days_since_last_activity=18,
                         prior_course_completions=0, current_week_in_course=8,
                         missed_deadlines_count=5, help_requests_count=0),
        "risk": 0.83, "trajectory": "deteriorating",
    },
    {
        "username": "roster_david",   "password": "roster123", "email": "david@xai.local",
        "full_name": "David Okafor",  "learner_id": "learner_r04", "course_id": "CCC",
        "module": "CCC", "presentation": "2014J",
        "features": dict(login_frequency_weekly=5, avg_session_duration_min=55, forum_posts_count=5,
                         video_completion_rate=0.78, quiz_avg_score=74, quiz_completion_rate=0.8,
                         assignment_submission_rate=0.85, days_since_last_activity=3,
                         prior_course_completions=2, current_week_in_course=4,
                         missed_deadlines_count=1, help_requests_count=1),
        "risk": 0.22, "trajectory": "stable",
    },
    {
        "username": "roster_elena",   "password": "roster123", "email": "elena@xai.local",
        "full_name": "Elena Kowalski","learner_id": "learner_r05", "course_id": "CCC",
        "module": "CCC", "presentation": "2013B",
        "features": dict(login_frequency_weekly=2, avg_session_duration_min=30, forum_posts_count=1,
                         video_completion_rate=0.45, quiz_avg_score=53, quiz_completion_rate=0.5,
                         assignment_submission_rate=0.55, days_since_last_activity=12,
                         prior_course_completions=0, current_week_in_course=10,
                         missed_deadlines_count=4, help_requests_count=2),
        "risk": 0.67, "trajectory": "at_risk_stable",
    },
    {
        "username": "roster_felix",   "password": "roster123", "email": "felix@xai.local",
        "full_name": "Felix Andersen","learner_id": "learner_r06", "course_id": "CCC",
        "module": "CCC", "presentation": "2014B",
        "features": dict(login_frequency_weekly=4, avg_session_duration_min=60, forum_posts_count=4,
                         video_completion_rate=0.7, quiz_avg_score=70, quiz_completion_rate=0.72,
                         assignment_submission_rate=0.78, days_since_last_activity=4,
                         prior_course_completions=1, current_week_in_course=6,
                         missed_deadlines_count=1, help_requests_count=2),
        "risk": 0.33, "trajectory": "improving",
    },
]


def _seed_demo_users() -> None:
    from backend.app.auth.auth import hash_password
    from backend.app.auth.database import get_db
    from backend.app.auth.models import (
        User, LearnerProfile, InstructorProfile, CourseEnrollment, StudentSnapshot
    )

    db = next(get_db())
    try:
        # ── Core demo accounts (admin / student / instructor) ──────────────────
        for spec in _DEMO_USERS:
            user = db.query(User).filter(User.username == spec["username"]).first()
            if user is None:
                user = User(
                    username=spec["username"],
                    email=spec["email"],
                    full_name=spec["username"].replace("_", " ").title(),
                    hashed_password=hash_password(spec["password"]),
                    role=spec["role"],
                    is_active=True,
                )
                db.add(user)
                db.flush()
                if spec["role"] == "student" and spec["learner_id"]:
                    lp = LearnerProfile(user_id=user.id, learner_id=spec["learner_id"],
                                        course_id="CCC", current_week=6)
                    db.add(lp)
                    db.flush()
                    db.add(StudentSnapshot(
                        learner_id=spec["learner_id"], display_name=user.full_name,
                        current_module="CCC", current_presentation="2014J",
                        dropout_risk_score=0.38, risk_trajectory="stable",
                    ))
                if spec["role"] == "instructor":
                    db.add(InstructorProfile(
                        user_id=user.id, department="Humanities", bio="Demo instructor"
                    ))
                log.info("Seeded demo user: %s (%s)", spec["username"], spec["role"])
            else:
                user.hashed_password = hash_password(spec["password"])
                user.is_active = True
                log.info("Updated demo user password: %s", spec["username"])

        db.flush()

        # ── Ensure instructor profile exists ───────────────────────────────────
        instr_profile = None
        instr_user = db.query(User).filter(User.username == "demo_instructor").first()
        if instr_user:
            instr_profile = db.query(InstructorProfile).filter(
                InstructorProfile.user_id == instr_user.id
            ).first()
            if instr_profile is None:
                instr_profile = InstructorProfile(
                    user_id=instr_user.id, department="Humanities", bio="Demo instructor"
                )
                db.add(instr_profile)
                db.flush()

        # ── Roster students enrolled under demo_instructor ─────────────────────
        for spec in _DEMO_ROSTER:
            user = db.query(User).filter(User.username == spec["username"]).first()
            if user is None:
                user = User(
                    username=spec["username"],
                    email=spec["email"],
                    full_name=spec["full_name"],
                    hashed_password=hash_password(spec["password"]),
                    role="student",
                    is_active=True,
                )
                db.add(user)
                db.flush()

            lp = db.query(LearnerProfile).filter(
                LearnerProfile.learner_id == spec["learner_id"]
            ).first()
            if lp is None:
                lp = LearnerProfile(
                    user_id=user.id,
                    learner_id=spec["learner_id"],
                    course_id=spec["course_id"],
                    current_week=spec["features"]["current_week_in_course"],
                )
                db.add(lp)
                db.flush()

            # Upsert snapshot
            snap = db.query(StudentSnapshot).filter(
                StudentSnapshot.learner_id == spec["learner_id"]
            ).first()
            if snap is None:
                snap = StudentSnapshot(learner_id=spec["learner_id"])
                db.add(snap)
            snap.display_name        = spec["full_name"]
            snap.current_module      = spec["module"]
            snap.current_presentation = spec["presentation"]
            snap.dropout_risk_score  = spec["risk"]
            snap.risk_trajectory     = spec["trajectory"]
            for k, v in spec["features"].items():
                setattr(snap, k, v)

            # Enroll under demo_instructor if not already enrolled
            if instr_user and instr_profile:
                existing = db.query(CourseEnrollment).filter(
                    CourseEnrollment.learner_profile_id == lp.id,
                    CourseEnrollment.instructor_profile_id == instr_profile.id,
                ).first()
                if existing is None:
                    db.add(CourseEnrollment(
                        learner_profile_id=lp.id,
                        instructor_profile_id=instr_profile.id,
                        course_id=spec["course_id"],
                    ))

        db.commit()
        log.info("Demo seeder complete — %d roster students enrolled", len(_DEMO_ROSTER))
    except Exception as e:
        db.rollback()
        log.warning("Demo user seeding failed: %s", e)
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Startup helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_gbm() -> None:
    gbm_path = MODELS_DIR / "gbm.pkl"
    if gbm_path.exists():
        blob = _load_pkl(gbm_path)
        state.gbm_model     = blob["model"]
        state.feature_names = blob["feature_names"]
        state.model_version = "gbm-v1"
        log.info("GBM loaded  features=%d", len(state.feature_names))
    else:
        log.warning("gbm.pkl not found — run trainer.py first")


def _load_rf() -> None:
    rf_path = MODELS_DIR / "rf.pkl"
    if rf_path.exists():
        state.rf_model = _load_pkl(rf_path)["model"]
        log.info("RF loaded")


def _load_train_data() -> None:
    train_path = DATA_DIR / "train.pkl"
    if train_path.exists():
        blob = _load_pkl(train_path)
        state.X_train     = blob["X"]
        state.y_train     = blob["y"]
        _ids = blob.get("learner_ids")
        state.learner_ids = np.asarray(_ids, dtype=object) if _ids is not None else None
        log.info("Training data loaded  X=%s", state.X_train.shape)


def _load_lstm() -> None:
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


def _load_transitions() -> None:
    transitions_path = DATA_DIR / "temporal" / "transitions.pkl"
    if transitions_path.exists():
        with open(transitions_path, "rb") as f:
            transitions = pickle.load(f)
        state.mc_simulator = MonteCarloSimulator(transitions)
        log.info("MonteCarloSimulator loaded — %d transition rows", len(transitions.get("deltas", [])))
    else:
        log.warning("transitions.pkl not found — /simulate will return empty distributions")


def _load_rankers() -> None:
    _reco_models_dir = MODELS_DIR / "recommenders"
    for _pkl_name, _attr in [
        ("student_ranker.pkl",    "student_ranker_explainer"),
        ("instructor_ranker.pkl", "instructor_ranker_explainer"),
    ]:
        _rp = _reco_models_dir / _pkl_name
        if _rp.exists():
            with open(_rp, "rb") as _f:
                _art = pickle.load(_f)
            # Note: causal_annotator may be None here if background init hasn't finished;
            # RankerExplainer accepts None and degrades gracefully.
            setattr(state, _attr, RankerExplainer(
                model           = _art["model"],
                feature_columns = _art["feature_columns"],
                encoder_maps    = _art["metadata"]["encoder_maps"],
                causal_annotator= state.causal_annotator,
            ))
            log.info("%s loaded  features=%d", _attr, len(_art["feature_columns"]))
        else:
            log.warning("%s not found — run train_recommenders.py", _pkl_name)


def _init_heavy_explainers(X_background) -> None:
    """
    Initialise the slow explainers (DiCE, Anchors, CausalAnnotator) in a
    background thread so the server can start accepting requests immediately.
    Results are written directly to `state`; Python's GIL makes the reference
    assignment atomic and safe.
    """
    if state.gbm_model is None or state.X_train is None:
        log.warning("_init_heavy_explainers: models not loaded — skipping")
        return

    log.info("[bg] Starting heavy XAI explainer initialisation …")

    # ── Phase 2: Explainers + Evaluators ──
    _X_background = X_background  # None on first call; updated later by LSTM block
    if state.gbm_model is not None and state.feature_names:
        try:
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
        except Exception as _shap_err:
            log.warning("SHAPExplainer/ArchipelagoExplainer failed to initialise: %s", _shap_err)

    if state.gbm_model is not None and state.X_train is not None:
        try:
            state.dice_explainer = DiCEExplainer(
                model         = state.gbm_model,
                X_train       = state.X_train,
                feature_names = state.feature_names,
                y_train       = state.y_train,
            )
            log.info("DiCEExplainer initialised")
        except Exception as _dice_err:
            log.warning("DiCEExplainer failed to initialise (dice_ml/pandas compat): %s", _dice_err)

    def _predict_fn(X):
        return (state.gbm_model.predict_proba(X)[:, 1] >= 0.5).astype(int)

    try:
        state.anchors_explainer = AnchorsExplainer(
            predict_fn    = _predict_fn,
            X_train       = state.X_train,
            feature_names = state.feature_names,
        )
        log.info("AnchorsExplainer initialised")
    except Exception as _anchors_err:
        log.warning("AnchorsExplainer failed to initialise: %s", _anchors_err)

    try:
        state.prototype_explainer = PrototypeExplainer(
            X_train       = state.X_train,
            y_train       = state.y_train,
            learner_ids   = state.learner_ids if state.learner_ids is not None else np.arange(len(state.X_train)),
            feature_names = state.feature_names,
        )
        log.info("PrototypeExplainer initialised")
    except Exception as _proto_err:
        log.warning("PrototypeExplainer failed to initialise: %s", _proto_err)

    try:
        state.uncertainty_estimator = UncertaintyEstimator(
            model = state.gbm_model,
            X_cal = state.X_train,
            y_cal = state.y_train,
        )
        log.info("UncertaintyEstimator initialised")
    except Exception as _unc_err:
        log.warning("UncertaintyEstimator failed to initialise: %s", _unc_err)

    try:
        state.causal_annotator = CausalAnnotator(
            X_train       = state.X_train,
            y_train       = state.y_train,
            feature_names = state.feature_names,
        )
        log.info("CausalAnnotator initialised (DoWhy)")
    except Exception as _causal_err:
        log.warning("CausalAnnotator failed to initialise: %s", _causal_err)

    try:
        state.action_ranker = ActionRanker(
            feature_names    = state.feature_names,
            X_train          = state.X_train,
            causal_annotator = state.causal_annotator,
        )
        log.info("ActionRanker initialised")
    except Exception as _ar_err:
        log.warning("ActionRanker failed to initialise: %s", _ar_err)

    import pandas as pd  # noqa: F401 — triggers Evidently's pandas dep lazily
    state.drift_monitor = DriftMonitor(
        reference_data = state.X_train,
        feature_names  = state.feature_names,
    )
    log.info("[bg] DriftMonitor initialised")

    # Wire causal_annotator into already-loaded ranker explainers (preserves reference_pool)
    if state.causal_annotator is not None:
        for _rattr in ("student_ranker_explainer", "instructor_ranker_explainer"):
            _re = getattr(state, _rattr, None)
            if _re is not None:
                _re.causal_annotator = state.causal_annotator

    # LSTM background sequences for DeepSHAP
    if state.lstm_model is not None and X_background is None:
        _snapshots_path = DATA_DIR / "temporal" / "snapshots.pkl"
        if _snapshots_path.exists():
            try:
                _bg_seqs, _, _ = build_sequences(_snapshots_path)
                _X_bg = _bg_seqs[:200]
                state.shap_explainer.deep_explainer  # re-init only if attribute exists
                log.info("[bg] LSTM background sequences loaded  shape=%s", _X_bg.shape)
            except Exception as _e:
                log.warning("[bg] Failed to load LSTM background sequences: %s", _e)

    state.fully_initialized = True
    log.info("[bg] All XAI explainers ready — fully_initialized=True")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Download model artifacts from S3 (no-op if AWS_S3_BUCKET not set) ──
    downloaded = download_models(MODELS_DIR)
    if downloaded:
        log.info("S3: downloaded %d model artifact(s)", len(downloaded))

    log.info("Loading models from %s …", MODELS_DIR)

    # ── Phase 1: Parallel I/O — load all pkl/pt files concurrently ──────────
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="startup") as _ex:
        _futures = {
            _ex.submit(_load_train_data): "train",
            _ex.submit(_load_gbm):        "gbm",
            _ex.submit(_load_rf):         "rf",
            _ex.submit(_load_lstm):       "lstm",
            _ex.submit(_load_transitions): "transitions",
            _ex.submit(_load_rankers):    "rankers",
        }
        for _f in _futures:
            try:
                _f.result()
            except Exception as _e:
                log.error("Startup I/O error [%s]: %s", _futures[_f], _e)

    # ── Phase 1 init: fast explainers (depend on gbm + lstm being loaded) ──
    # LSTM background sequences (build_sequences is slow) are deferred to the
    # background thread; SHAPExplainer starts without them — DeepSHAP is a
    # nice-to-have and not required for /predict or /explain.
    if state.gbm_model is not None and state.feature_names:
        state.shap_explainer = SHAPExplainer(
            gbm_model     = state.gbm_model,
            feature_names = state.feature_names,
            lstm_model    = state.lstm_model,
            X_background  = None,
        )
        log.info("SHAPExplainer initialised")

        state.archipelago_explainer = ArchipelagoExplainer(
            tree_explainer = state.shap_explainer.tree_explainer,
            feature_names  = state.feature_names,
        )
        log.info("ArchipelagoExplainer initialised")

    # ── Phase 1 stores: lightweight, no ML dependency ────────────────────────
    state.trust_scorer      = TrustScorer()
    state.explanation_store = ExplanationStore()
    state.drift_detector    = ExplanationDriftDetector()
    state.prediction_logger = PredictionLogger()
    state.feedback_store    = FeedbackStore()
    state.event_store       = EventStore()
    log.info("Stores initialised")

    init_auth_db()
    log.info("Auth DB initialised (data/auth.db)")
    _seed_demo_users()

    state.llm_narrator = LLMNarrator()
    if state.llm_narrator.available:
        log.info("LLMNarrator initialised  model=%s", state.llm_narrator.model)
    else:
        log.warning("LLMNarrator disabled — set GROQ_API_KEY to enable narration")

    # ── Core ready: /predict and /health available ───────────────────────────
    state.core_ready = True
    log.info("Core ready — device=%s  (XAI explainers initialising in background)", DEVICE)

    # ── Phase 2: Heavy explainers in background thread ───────────────────────
    # DiCE, Anchors, Prototype, CausalAnnotator, ActionRanker, DriftMonitor
    # load from disk caches after first run — typically < 5s on subsequent starts.
    _loop = asyncio.get_event_loop()
    _bg_future = _loop.run_in_executor(None, _init_heavy_explainers, None)

    # ── Monte Carlo Simulator ──
    transitions_path = DATA_DIR / "temporal" / "transitions.pkl"
    if transitions_path.exists():
        with open(transitions_path, "rb") as f:
            transitions = pickle.load(f)
        state.mc_simulator = MonteCarloSimulator(transitions)
        log.info("MonteCarloSimulator loaded — %d transition rows", len(transitions.get("deltas", [])))
    else:
        log.warning("transitions.pkl not found — /simulate will return empty distributions")

    # ── Recommendation Rankers ──
    # Feature 5: load student & instructor reference pools for KNN prototypes
    import pandas as _pd
    _topk_path    = DATA_DIR / "recommendations" / "precomputed" / "student_topk.csv"
    _reco_ref_pool = None
    if _topk_path.exists():
        try:
            _reco_ref_pool = _pd.read_csv(_topk_path)
            log.info("Student reco reference pool loaded  rows=%d", len(_reco_ref_pool))
        except Exception as _e:
            log.warning("Failed to load student_topk.csv: %s", _e)

    _instr_topk_path = DATA_DIR / "recommendations" / "precomputed" / "instructor_topk.csv"
    _instr_ref_pool  = None
    if _instr_topk_path.exists():
        try:
            _instr_ref_pool = _pd.read_csv(_instr_topk_path)
            log.info("Instructor reco reference pool loaded  rows=%d", len(_instr_ref_pool))
        except Exception as _e:
            log.warning("Failed to load instructor_topk.csv: %s", _e)

    _reco_models_dir = MODELS_DIR / "recommenders"
    for _pkl_name, _attr in [
        ("student_ranker.pkl",    "student_ranker_explainer"),
        ("instructor_ranker.pkl", "instructor_ranker_explainer"),
    ]:
        _rp = _reco_models_dir / _pkl_name
        if _rp.exists():
            with open(_rp, "rb") as _f:
                _art = pickle.load(_f)
            _feat_cols = _art["feature_columns"]

            # Feature 3 & 5: build X_train_sample from reference pool
            _X_sample, _pool = None, None
            _ref_source = (
                _reco_ref_pool  if _attr == "student_ranker_explainer"    else
                _instr_ref_pool if _attr == "instructor_ranker_explainer" else None
            )
            if _ref_source is not None:
                _pool  = _ref_source
                _avail = [c for c in _feat_cols if c in _ref_source.columns]
                if len(_avail) >= 3:
                    try:
                        _ps = _ref_source.dropna(subset=_avail).head(500)
                        if len(_ps) > 10:
                            _Xf = _pd.DataFrame(0.0, index=range(len(_ps)), columns=_feat_cols)
                            for _c in _avail:
                                _Xf[_c] = _ps[_c].values
                            _X_sample = _Xf.values.astype(float)
                    except Exception as _xe:
                        log.warning("%s X_train_sample build failed: %s", _attr, _xe)

            setattr(state, _attr, RankerExplainer(
                model           = _art["model"],
                feature_columns = _feat_cols,
                encoder_maps    = _art["metadata"]["encoder_maps"],
                causal_annotator= state.causal_annotator,
                X_train_sample  = _X_sample,
                reference_pool  = _pool,
            ))
            log.info("%s loaded  features=%d", _attr, len(_feat_cols))
        else:
            log.warning("%s not found — run train_recommenders.py", _pkl_name)

    # Feature 6/13: Recommendation explanation store
    state.reco_explanation_store = RecommendationExplanationStore()
    log.info("RecommendationExplanationStore initialised")

    # Feature 12: Fairness auditor
    state.fairness_auditor = FairnessAuditor()
    log.info("FairnessAuditor initialised")

    log.info("Startup complete — device=%s", DEVICE)
    yield

    # Wait briefly for background init to finish cleanly on shutdown
    try:
        await asyncio.wait_for(_bg_future, timeout=30)
    except (asyncio.TimeoutError, Exception):
        pass
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
# Static file serving — React SPA
# ──────────────────────────────────────────────────────────────────────────────
# The built frontend lives at backend/static/ (Vite outputs there via vite.config.js).
# We mount /assets explicitly so hashed JS/CSS/image bundles are served with
# correct MIME types. All other non-API routes fall through to index.html so
# React Router can handle client-side navigation.

_STATIC_DIR = PROJECT_ROOT / "backend" / "static"
_INDEX_HTML  = _STATIC_DIR / "index.html"

if _STATIC_DIR.exists():
    # Serve hashed asset bundles (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR / "assets")), name="assets")
    log.info("Static assets mounted from %s", _STATIC_DIR / "assets")
else:
    log.warning(
        "Frontend static dir not found at %s — run 'npm run build' in frontend/\n"
        "  The API endpoints still work; only the React UI is unavailable.",
        _STATIC_DIR,
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
        "core_ready":    state.core_ready,
        "xai_ready":     state.fully_initialized,
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
        _lstm_n = state.lstm_config.get("n_features", len(state.feature_names))
        seq    = _build_lstm_sequence([], features, state.feature_names[:_lstm_n])
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
    try:
        return _compare_models_impl(req)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("compare_models failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Compare failed: {exc}") from exc


def _compare_models_impl(req: CompareRequest):
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
        lstm_n_features = state.lstm_config.get("n_features", len(state.feature_names))
        lstm_feature_names = state.feature_names[:lstm_n_features]
        seq        = _build_lstm_sequence(req.history, req.features, lstm_feature_names)
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
        _lstm_n = state.lstm_config.get("n_features", len(state.feature_names))
        seq      = _build_lstm_sequence(req.history, req.features, state.feature_names[:_lstm_n])
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
        model_feature_names=state.feature_names,
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
    drift_fallback_reason = ""
    if state.drift_monitor is None:
        try:
            state.drift_monitor = DriftMonitor(
                reference_data=state.X_train,
                feature_names=state.feature_names,
            )
            log.info("DriftMonitor lazily initialised from /mlops/drift-report")
        except Exception as e:
            drift_fallback_reason = f"monitor_init_failed: {e}"
            log.warning("DriftMonitor lazy init failed, using fallback: %s", e)

    # Try cached report first
    if state.drift_monitor is not None:
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

    try:
        if state.drift_monitor is not None:
            report = state.drift_monitor.check_drift(current)
            return report.to_dict()
        if not drift_fallback_reason:
            drift_fallback_reason = "monitor_unavailable"
    except Exception as e:
        # Degrade gracefully so admin dashboard remains usable even when
        # optional drift dependencies are unavailable.
        drift_fallback_reason = f"monitor_check_failed: {e}"
        log.warning("Drift report fallback engaged: %s", e)

    # Lightweight fallback: mark feature drift by standardized mean shift.
    try:
        if isinstance(current, pd.DataFrame):
            cur_df = current.copy()
        else:
            cur_df = pd.DataFrame(current)
        cur_df.columns = [str(c) for c in cur_df.columns]
        cur_df = cur_df.apply(pd.to_numeric, errors="coerce")

        # Reference mode A: training baseline when available.
        if state.X_train is not None and state.feature_names:
            ref_df = pd.DataFrame(state.X_train)
            ref_df = ref_df.apply(pd.to_numeric, errors="coerce")

            n_cols = min(ref_df.shape[1], cur_df.shape[1], len(state.feature_names))
            if n_cols == 0:
                raise ValueError("training/current matrices have no comparable columns")

            # Build stable, unique display names from model feature names.
            base_names = [str(x) for x in state.feature_names[:n_cols]]
            seen = {}
            deduped_names = []
            for name in base_names:
                idx = seen.get(name, 0)
                deduped_names.append(name if idx == 0 else f"{name}__{idx}")
                seen[name] = idx + 1

            ref_df = ref_df.iloc[:, :n_cols].copy()
            cur_df = cur_df.iloc[:, :n_cols].copy()
            ref_df.columns = deduped_names
            cur_df.columns = deduped_names

            avail = [f for f in deduped_names if not cur_df[f].isna().all()]
            if not avail:
                raise ValueError("no comparable columns between current and training reference")

            ref_df = ref_df[avail]
            cur_df = cur_df[avail]
        else:
            # Reference mode B: self-baseline split from current predictions.
            numeric_cols = [c for c in cur_df.columns if not cur_df[c].isna().all()]
            if len(numeric_cols) == 0:
                raise ValueError("current predictions contain no numeric columns")

            cur_df = cur_df[numeric_cols]
            split = max(1, len(cur_df) // 2)
            if len(cur_df) < 2:
                raise ValueError("not enough predictions for self-baseline fallback")

            ref_df = cur_df.iloc[:split]
            cur_df = cur_df.iloc[split:]
            if cur_df.empty:
                raise ValueError("not enough recent rows after self-baseline split")
            avail = list(cur_df.columns)

        ref_mean = ref_df.mean(numeric_only=True)
        ref_std = ref_df.std(numeric_only=True).replace(0, 1e-6)
        cur_mean = cur_df.mean(numeric_only=True)

        shift = ((cur_mean - ref_mean).abs() / ref_std).fillna(0.0)
        drift_mask = shift > 1.0
        n_drifted = int(drift_mask.sum())
        n_total = int(len(avail))
        drift_share = n_drifted / max(n_total, 1)

        feature_drifts = [
            {
                "feature": feat,
                "drift_detected": bool(drift_mask.get(feat, False)),
                "statistic": float(round(shift.get(feat, 0.0), 6)),
                "p_value": 1.0,
                "method": "z_mean_shift_fallback",
            }
            for feat in avail
        ]

        return {
            "dataset_drift": drift_share >= 0.2,
            "n_drifted_features": n_drifted,
            "n_total_features": n_total,
            "drift_share": round(drift_share, 4),
            "feature_drifts": feature_drifts,
            "message": f"Fallback drift report ({drift_fallback_reason or 'monitor_unavailable'})",
        }
    except Exception as fallback_err:
        log.warning("Fallback drift computation failed: %s", fallback_err)
        return {
            "dataset_drift": False,
            "n_drifted_features": 0,
            "n_total_features": 0,
            "drift_share": 0.0,
            "feature_drifts": [],
            "message": f"Drift unavailable ({drift_fallback_reason or 'insufficient_data'}; fallback_failed: {fallback_err})",
        }


@app.get("/mlops/metrics")
def mlops_metrics():
    """Current production model metrics from training summary."""
    summary_path = MODELS_DIR / "training_summary.json"
    tuning_path  = MODELS_DIR / "tuning_summary.json"
    result = {}
    if summary_path.exists():
        with open(summary_path) as f:
            raw = json.load(f)
        # Flatten GBM sub-dict to the keys the dashboard StatCards expect.
        # training_summary.json stores metrics under {"gbm": {"test_auc_roc": ...}}
        gbm = raw.get("gbm", {})

        # Compute SHAP fidelity on-the-fly if not in the summary file.
        shap_fidelity = raw.get("shap_fidelity")
        if shap_fidelity is None and not hasattr(state, "_cached_shap_fidelity"):
            try:
                if (
                    state.shap_explainer is not None
                    and state.gbm_model is not None
                    and state.X_train is not None
                ):
                    import numpy as _np
                    sample_size = min(200, len(state.X_train))
                    X_sample = state.X_train[:sample_size]
                    explainer = state.shap_explainer._explainer
                    shap_vals = explainer.shap_values(X_sample)
                    base = explainer.expected_value
                    if isinstance(base, list):
                        base = base[1]  # binary classification: positive class
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]
                    predicted = state.gbm_model.predict_proba(X_sample)[:, 1]
                    reconstructed = shap_vals.sum(axis=1) + base
                    # Apply sigmoid if SHAP values are in log-odds space
                    if reconstructed.max() > 1.5 or reconstructed.min() < -0.5:
                        reconstructed = 1 / (1 + _np.exp(-reconstructed))
                    fidelity = 1.0 - float(_np.mean(_np.abs(predicted - reconstructed)))
                    state._cached_shap_fidelity = round(max(0.0, min(1.0, fidelity)), 4)
                    log.info("Computed SHAP fidelity on-the-fly: %.4f", state._cached_shap_fidelity)
            except Exception as e:
                log.warning("SHAP fidelity computation failed: %s", e)
                state._cached_shap_fidelity = None
        if shap_fidelity is None:
            shap_fidelity = getattr(state, "_cached_shap_fidelity", None)

        result["training"] = {
            "auc":           gbm.get("test_auc_roc"),
            "f1":            gbm.get("test_f1"),
            "brier":         gbm.get("test_brier_score"),
            "avg_precision": gbm.get("test_avg_precision"),
            "shap_fidelity": shap_fidelity,
            "gbm":           gbm,
            "rf":            raw.get("rf", {}),
        }
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


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints — Recommendation Engine
#
# Explainability matrix vs dropout engine:
#   ✅ TreeSHAP (pred_contrib)   ✅ WhatIf    ✅ Anchor rule
#   ✅ Feature interactions       ✅ Causal annotations
#   ✅ SHAP stability             ✅ Plain-language narration
#   ❌ DiCE (classifier-only)    ❌ MAPIE (no predict_proba)
#   ❌ LSTM temporal (not applicable to ranking)
# ──────────────────────────────────────────────────────────────────────────────

class RecoItem(BaseModel):
    """A single candidate item to score. item_id is echoed back in results."""
    item_id:  str = ""
    features: Dict[str, Any] = {}   # ranker feature dict for this item


class StudentRecoRequest(BaseModel):
    learner_id:   str = "anonymous"
    items:        list[RecoItem]
    top_k:        int = Field(5, ge=1, le=20)
    include_shap: bool = True


class StudentRecoExplainRequest(BaseModel):
    learner_id: str = "anonymous"
    features:   Dict[str, Any]   # combined learner + item features (one row)
    item_id:    str = ""
    audience:   str = "learner"   # "learner" | "instructor" | "both"


class StudentRecoWhatIfRequest(BaseModel):
    learner_id: str = "anonymous"
    features:   Dict[str, Any]   # combined learner + item features (one row)
    overrides:  Dict[str, Any]   # feature values to change


class InstructorRecoRequest(BaseModel):
    instructor_id: str = "anonymous"
    items:         list[RecoItem]
    top_k:         int = Field(5, ge=1, le=20)
    include_shap:  bool = True


class InstructorRecoExplainRequest(BaseModel):
    instructor_id: str = "anonymous"
    features:      Dict[str, Any]
    item_id:       str = ""
    audience:      str = "instructor"   # "learner" | "instructor" | "both"


class InstructorRecoWhatIfRequest(BaseModel):
    instructor_id: str = "anonymous"
    features:      Dict[str, Any]
    overrides:     Dict[str, Any]


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_student_ranker():
    if state.student_ranker_explainer is None:
        raise HTTPException(503, "Student ranker not loaded — run train_recommenders.py")


def _require_instructor_ranker():
    if state.instructor_ranker_explainer is None:
        raise HTTPException(503, "Instructor ranker not loaded — run train_recommenders.py")


def _merge_item(item: RecoItem) -> Dict[str, Any]:
    """Merge item_id into feature dict so the ranker sees a single row."""
    return {"item_id": item.item_id, **item.features}


# ── /causal/graph ─────────────────────────────────────────────────────────

@app.get("/causal/graph")
def causal_graph():
    """Feature 10: return the causal DAG as a JSON structure for frontend rendering."""
    annotator_available = state.causal_annotator is not None
    effects = state.causal_annotator.get_causal_effects() if annotator_available else {}

    _groups = {
        "confounder":  ["prior_course_completions", "current_week_in_course"],
        "engagement":  ["login_frequency_weekly", "avg_session_duration_min",
                        "forum_posts_count", "help_requests_count"],
        "performance": ["quiz_avg_score", "quiz_completion_rate",
                        "assignment_submission_rate", "video_completion_rate"],
        "risk_signal": ["days_since_last_activity", "missed_deadlines_count"],
    }

    nodes = []
    for group, members in _groups.items():
        for node_id in members:
            effect = effects.get(node_id)
            nodes.append({
                "id":              node_id,
                "label":           node_id.replace("_", " ").title(),
                "group":           group,
                "is_causal":       effect.is_causal if effect else False,
                "ate":             round(effect.ate, 6) if effect else 0.0,
                "effect_direction": effect.effect_direction if effect else "neutral",
            })
    nodes.append({
        "id":              "dropout_risk",
        "label":           "Dropout Risk",
        "group":           "outcome",
        "is_causal":       True,
        "ate":             0.0,
        "effect_direction": "neutral",
    })

    _conf = ["prior_course_completions", "current_week_in_course"]
    _eng  = ["login_frequency_weekly", "avg_session_duration_min",
             "forum_posts_count", "help_requests_count"]
    _perf = ["quiz_avg_score", "quiz_completion_rate",
             "assignment_submission_rate", "video_completion_rate"]
    _risk = ["days_since_last_activity", "missed_deadlines_count"]
    _out  = "dropout_risk"

    edges = []
    for c in _conf:
        for e in _eng:  edges.append({"from": c, "to": e})
        for p in _perf: edges.append({"from": c, "to": p})
        edges.append({"from": c, "to": _out})
    for e in _eng:
        for p in _perf: edges.append({"from": e, "to": p})
        for r in _risk: edges.append({"from": e, "to": r})
        edges.append({"from": e, "to": _out})
    for p in _perf: edges.append({"from": p, "to": _out})
    for r in _risk: edges.append({"from": r, "to": _out})

    payload = {"nodes": nodes, "edges": edges}
    if not annotator_available:
        payload["message"] = "Causal annotator not initialised; returning static DAG"
    return payload


# ── /causal/explain ───────────────────────────────────────────────────────

_CAUSAL_EXPLAIN_SYSTEM_PROMPT = """
You are an expert in causal inference and educational data mining, narrating
a pre-computed Causal DAG for a school administrator viewing an XAI dashboard.

STRICT RULES:
- ONLY describe what is present in the data payload. Never invent numbers.
- Use the ATE (Average Treatment Effect) values exactly as given.
- Explain in plain language that a non-technical school admin can understand.
- Return a JSON object with EXACTLY these 3 fields, no markdown wrapper:
    "summary"          : 2-3 sentence high-level overview of what the DAG reveals
                         about dropout risk factors and their causal relationships.
    "key_insights"     : list of 3-5 bullet-point strings, each one specific insight
                         drawn from the graph structure and ATE values.
    "strongest_drivers": list of objects with "feature", "ate", "direction", and
                         "explanation" (1-sentence plain-language interpretation).
                         Include only features where is_causal=true, sorted by |ATE|.
- Total response must be parseable JSON. No explanation outside the JSON.
""".strip()

_CAUSAL_EXPLAIN_USER_TEMPLATE = """
Causal DAG Summary for the XAI Learning Analytics Dashboard:

Node groups:
- Confounders: prior context variables that influence both engagement and performance
- Engagement: student activity and participation metrics
- Performance: academic achievement indicators
- Risk signals: early warning indicators
- Outcome: the dropout_risk prediction target

Nodes (feature → group, is_causal, ATE, direction):
{nodes_text}

Edge count: {edge_count} directed edges connecting the above nodes.

Key edge patterns:
- Confounders → Engagement, Performance, Outcome
- Engagement → Performance, Risk signals, Outcome
- Performance → Outcome
- Risk signals → Outcome

Narrate this causal structure for a school admin. Return JSON only.
""".strip()


@app.post("/causal/explain")
def causal_explain():
    """
    Uses Groq LLM to narrate the Causal DAG in plain English.
    Returns { summary, key_insights, strongest_drivers }.
    """
    # Get the graph data
    graph = causal_graph()
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # Build structured fallback even without LLM
    causal_nodes = [n for n in nodes if n.get("is_causal")]
    strongest = sorted(causal_nodes, key=lambda n: abs(n.get("ate", 0)), reverse=True)

    if state.llm_narrator is None or not state.llm_narrator.available:
        # Generate a simple rule-based explanation as fallback
        drivers = []
        for n in strongest[:5]:
            drivers.append({
                "feature": n["label"],
                "ate": n.get("ate", 0),
                "direction": n.get("effect_direction", "neutral"),
                "explanation": f"{n['label']} has a causal effect (ATE={n.get('ate', 0):.4f}) on dropout risk.",
            })
        return {
            "summary": (
                f"This causal graph models {len(nodes)} features connected by "
                f"{len(edges)} directed edges, showing how student engagement, "
                f"performance, and risk signals causally influence dropout risk. "
                f"{len(causal_nodes)} features have statistically significant causal effects."
            ),
            "key_insights": [
                f"{len(causal_nodes)} out of {len(nodes)} features show significant causal effects on dropout.",
                "Engagement metrics (login frequency, forum posts) causally influence both performance and risk signals.",
                "Performance metrics (quiz scores, submission rates) directly affect dropout risk.",
                "Risk signals (days since activity, missed deadlines) are downstream of engagement and directly predict dropout.",
            ],
            "strongest_drivers": drivers,
            "source": "fallback",
        }

    # Build LLM prompt
    node_lines = []
    for n in nodes:
        node_lines.append(
            f"  {n['label']} ({n['group']}): is_causal={n.get('is_causal', False)}, "
            f"ATE={n.get('ate', 0):.6f}, direction={n.get('effect_direction', 'neutral')}"
        )
    nodes_text = "\n".join(node_lines)

    user_prompt = _CAUSAL_EXPLAIN_USER_TEMPLATE.format(
        nodes_text=nodes_text,
        edge_count=len(edges),
    )

    try:
        import json as _json
        from groq import Groq as _Groq
        _client = _Groq(api_key=state.llm_narrator.api_key)
        _resp = _client.chat.completions.create(
            model=state.llm_narrator.model,
            messages=[
                {"role": "system", "content": _CAUSAL_EXPLAIN_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        raw = (_resp.choices[0].message.content or "{}").strip()
        # Strip markdown fence if present
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = _json.loads(raw)
        parsed["source"] = "groq"
        return parsed
    except Exception as _e:
        log.warning("causal_explain: Groq call failed (%s) — returning fallback", _e)
        drivers = []
        for n in strongest[:5]:
            drivers.append({
                "feature": n["label"],
                "ate": n.get("ate", 0),
                "direction": n.get("effect_direction", "neutral"),
                "explanation": f"{n['label']} has a causal effect (ATE={n.get('ate', 0):.4f}) on dropout risk.",
            })
        top_driver_names = [n["label"] for n in strongest[:3]]
        return {
            "summary": (
                f"This causal graph models {len(nodes)} features across {len(edges)} directed edges. "
                f"{len(causal_nodes)} features show statistically significant causal effects on dropout risk. "
                f"Strongest drivers: {', '.join(top_driver_names) or 'none identified'}."
            ),
            "key_insights": [
                f"{len(causal_nodes)} of {len(nodes)} features have significant causal effects.",
                "Engagement metrics causally influence both performance and risk signals.",
                "Performance metrics directly affect dropout risk.",
            ],
            "strongest_drivers": drivers,
            "source": "rule_based",
        }


# ── /fairness/explain ─────────────────────────────────────────────────────

_FAIRNESS_SYSTEM_PROMPT = """
You are a fairness auditor narrating pre-computed algorithmic fairness results for a school administrator.

STRICT RULES:
- ONLY describe what is present in the data payload. Never invent numbers or group sizes.
- Never say "probably", "might", or "could" — only state what the data shows.
- When protected attribute data is absent, clearly explain WHY (items lacked those fields) and what
  that means for the validity of the result.
- Return a JSON object with EXACTLY these 4 fields, no markdown wrapper:
    "methodology"  : 1-2 sentences on what the check does and what threshold was used.
    "findings"     : 2-3 sentences on group distributions found, and any disparities flagged.
                     If no groups had data, explain that explicitly.
    "verdict"      : 1 sentence — is the system fair, and what does that actually mean here?
    "action_items" : list of 1-3 specific, actionable strings for the admin.
- Total response must be parseable JSON. No explanation outside the JSON.
""".strip()

_FAIRNESS_USER_TEMPLATE = """
Fairness Audit Results:

Methodology:
- Protected groups audited: {protected_features}
- Disparity threshold: {threshold_pct:.0f}% relative mean-score deviation
- Items scored: {item_count}
- Overall mean recommendation score: {overall_mean:.4f}

Data availability:
- Features with demographic data found: {features_with_data}
- Features with NO data (absent from all scored items): {features_without_data}

Result: {result_label}

Group score breakdown (feature → group → mean score):
{group_scores_text}

Flagged disparities:
{flagged_text}

Narrate these audit results for an admin. Return JSON only.
""".strip()


class FairnessExplainRequest(BaseModel):
    report: dict  # the full FairnessReport.to_dict() output


@app.post("/fairness/explain")
def fairness_explain(req: FairnessExplainRequest):
    """
    Uses Groq LLM to narrate a pre-computed FairnessReport in plain English.
    Returns { methodology, findings, verdict, action_items }.
    """
    if state.llm_narrator is None or not state.llm_narrator.available:
        _report    = req.report
        _flagged   = _report.get("flagged_disparities", [])
        _n_flagged = len(_flagged)
        _gs        = _report.get("group_scores", {})
        _overall   = _report.get("overall_fair", True)
        _findings_parts = []
        for feat, groups in list(_gs.items())[:2]:
            for grp, sc in list(groups.items())[:3]:
                _findings_parts.append(f"{feat}[{grp}]: {sc:.3f}")
        _findings  = "; ".join(_findings_parts) if _findings_parts else "No demographic group data available."
        _verdict   = "No significant disparities detected across groups." if _overall else f"{_n_flagged} group(s) show disparity above threshold."
        _actions   = [f"Review group '{d['feature']}[{d['group']}]' — deviation {d.get('deviation_pct', 0):+.1f}%." for d in _flagged[:3]] if _flagged else ["No disparities detected — continue monitoring regularly."]
        return {
            "methodology":  "Fairness check compares mean recommendation scores across demographic groups and flags groups deviating more than the threshold from the overall mean.",
            "findings":     _findings,
            "verdict":      _verdict,
            "action_items": _actions,
            "source":       "rule_based",
        }

    report    = req.report
    meta      = report.get("metadata", {})
    protected = meta.get("protected_features_checked", [])
    with_data = meta.get("features_with_data", [])
    without   = [f for f in protected if f not in with_data]
    overall_fair      = report.get("overall_fair", True)
    flagged           = report.get("flagged_disparities", [])
    group_scores      = report.get("group_scores", {})

    # Build human-readable group scores block
    if group_scores:
        lines = []
        for feat, groups in group_scores.items():
            for grp, score in groups.items():
                lines.append(f"  {feat} [{grp}] → {score:.4f}")
        group_scores_text = "\n".join(lines)
    else:
        group_scores_text = "  (no demographic data found in scored items)"

    # Build flagged block
    if flagged:
        flag_lines = []
        for d in flagged:
            flag_lines.append(
                f"  ⚑ {d['feature']} [{d['group']}]: score={d['score']:.4f}, "
                f"deviation={d['deviation_pct']:+.1f}% ({d['direction']})"
            )
        flagged_text = "\n".join(flag_lines)
    else:
        flagged_text = "  None — no group exceeded the disparity threshold."

    result_label = (
        "FAIR — no significant disparities detected"
        if overall_fair
        else f"DISPARITIES DETECTED — {len(flagged)} group(s) flagged"
    )

    user_prompt = _FAIRNESS_USER_TEMPLATE.format(
        protected_features   = ", ".join(protected) if protected else "none specified",
        threshold_pct        = meta.get("disparity_threshold_pct", 15.0),
        item_count           = meta.get("item_count", 0),
        overall_mean         = meta.get("overall_mean", 0.0),
        features_with_data   = ", ".join(with_data) if with_data else "none",
        features_without_data= ", ".join(without) if without else "none",
        result_label         = result_label,
        group_scores_text    = group_scores_text,
        flagged_text         = flagged_text,
    )

    try:
        import json as _json
        from groq import Groq as _Groq
        _client = _Groq(api_key=state.llm_narrator.api_key)
        _resp   = _client.chat.completions.create(
            model    = state.llm_narrator.model,
            messages = [
                {"role": "system", "content": _FAIRNESS_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = 0.3,
            max_tokens  = 500,
        )
        raw = (_resp.choices[0].message.content or "{}").strip()
        # Strip markdown fence if present
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:])
            if raw.endswith("```"):
                raw = raw[:-3].strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = _json.loads(raw)
        parsed["source"] = "groq"
        return parsed
    except Exception as _e:
        log.warning("fairness_explain: Groq call failed (%s) — returning fallback", _e)
        n_groups = sum(len(g) for g in group_scores.values())
        return {
            "methodology":  (
                f"Checks mean recommendation score per demographic group across "
                f"{', '.join(protected) if protected else 'no configured'} attributes. "
                f"Groups deviating >{meta.get('disparity_threshold_pct', 15):.0f}% from the "
                f"overall mean ({meta.get('overall_mean', 0):.3f}) are flagged."
            ),
            "findings":     (
                f"Scored {meta.get('item_count', 0)} item(s). "
                + (f"Found demographic data for: {', '.join(with_data)}. " if with_data else
                   f"No protected attribute data found in the scored items "
                   f"({', '.join(without) if without else 'all features absent'} were missing). ")
                + (f"{len(flagged)} disparity(ies) flagged." if flagged else "No disparities flagged.")
            ),
            "verdict":      (
                "No disparities detected — but note: the result is only as meaningful as the "
                "demographic data present in the scored items."
                if overall_fair
                else f"{len(flagged)} group(s) show score disparities exceeding the threshold."
            ),
            "action_items": (
                ["Include protected attributes (explicit_gender, explicit_age_band, explicit_disability) "
                 "in the scored item features to get a meaningful fairness analysis."]
                if not with_data
                else ["Review flagged groups and their score deviations in the audit panel above."]
            ),
            "source": "fallback",
        }


# ── /recommend/health ─────────────────────────────────────────────────────

@app.get("/recommend/health")
def recommend_health():
    """Status of both recommendation rankers (and overall XAI readiness)."""
    student_ok    = state.student_ranker_explainer is not None
    instructor_ok = state.instructor_ranker_explainer is not None
    return {
        "loaded":            student_ok or instructor_ok or state.fully_initialized,
        "student_ranker":    (
            state.student_ranker_explainer.health()
            if state.student_ranker_explainer else {"loaded": False}
        ),
        "instructor_ranker": (
            state.instructor_ranker_explainer.health()
            if state.instructor_ranker_explainer else {"loaded": False}
        ),
        "xai_ready":         state.fully_initialized,
        "core_ready":        state.core_ready,
    }


# ── /recommend/student ────────────────────────────────────────────────────────

@app.post("/recommend/student")
def recommend_student(req: StudentRecoRequest):
    """
    Score and rank candidate resources for a student.

    Each item in `items` carries its own feature dict.  The ranker scores all
    items and returns the top-K with SHAP-based explanations.

    Explainability returned per item:
      - shap_values     : feature → score contribution
      - top_features    : top-3 features by |shap|
    """
    _require_student_ranker()
    if not req.items:
        raise HTTPException(422, "Provide at least one item to score")

    rows = [_merge_item(it) for it in req.items]
    scored = state.student_ranker_explainer.score_items(
        items=rows,
        top_k=req.top_k,
        include_shap=req.include_shap,
    )

    # Feature 11: Diversity score
    _id_to_feats = {
        str(it.item_id) if it.item_id else str(i): it.features
        for i, it in enumerate(req.items)
    }
    _modules = [
        str(_id_to_feats.get(s.item_id, {}).get(
            "recommended_module",
            _id_to_feats.get(s.item_id, {}).get("module", ""),
        ))
        for s in scored
    ]
    _modules = [m for m in _modules if m]
    _diversity = round(len(set(_modules)) / len(scored), 4) if scored else 0.0
    _div_warn  = (
        "Low diversity: recommendations are concentrated in few modules. "
        "Consider exploring other topics."
        if _diversity < 0.5 and scored
        else None
    )

    resp = {
        "learner_id":        req.learner_id,
        "top_k":             req.top_k,
        "recommendations":   [s.to_dict() for s in scored],
        "diversity_score":   _diversity,
        "diversity_warning": _div_warn,
    }

    # Feature 12: Fairness audit
    if state.fairness_auditor is not None and scored:
        _s_feats  = [_id_to_feats.get(s.item_id, {}) for s in scored]
        _s_scores = [s.score for s in scored]
        resp["fairness_audit"] = state.fairness_auditor.audit(_s_feats, _s_scores).to_dict()

    return resp


@app.post("/recommend/student/explain")
def recommend_student_explain(req: StudentRecoExplainRequest):
    """
    Full XAI explanation for why an item would be recommended to a student.

    Explainability:
      - shap_values          : per-feature score contribution
      - top_features         : top-5 by |shap|
      - anchor_rule          : IF-THEN rule from top features + actual values
      - feature_interactions : top-3 feature pairs by |shap_i × shap_j|
      - causal_annotations   : causal / correlational / confounder per feature
      - shap_stability       : 0-1 confidence that top features stay stable
      - plain_language       : one-sentence human-readable reason
    """
    _require_student_ranker()
    explanation = state.student_ranker_explainer.explain(
        features=req.features,
        item_id=req.item_id,
    )

    # Feature 6: persist to reco explanation store + drift detection
    if state.reco_explanation_store is not None:
        _top3 = explanation.top_features[:3]
        _ts   = (
            explanation.trust_score.get("trust_score")
            if explanation.trust_score else None
        )
        state.reco_explanation_store.save(
            learner_id    = req.learner_id,
            score         = explanation.score,
            shap_values   = explanation.shap_values,
            top3_features = _top3,
            trust_score   = _ts,
            anchor_rule   = explanation.anchor_rule,
        )
        if state.drift_detector is not None:
            _drift = state.drift_detector.check_learner_drift(
                req.learner_id, state.reco_explanation_store
            )
            explanation.explanation_drift = _drift.to_dict() if _drift else None

    # Feature 1: LLM narration with recommendation context
    if state.llm_narrator is not None:
        _reco_dict = {"learner_id": req.learner_id, **explanation.to_dict()}
        _narration = state.llm_narrator.narrate(
            explain_resp = _reco_dict,
            learner_id   = req.learner_id,
            audience     = req.audience,
            context_type = "recommendation",
        )
        explanation.narratives = _narration.to_dict()

    return {"learner_id": req.learner_id, **explanation.to_dict()}


@app.post("/recommend/student/whatif")
def recommend_student_whatif(req: StudentRecoWhatIfRequest):
    """
    Counterfactual: how does the recommendation score change if feature
    values are modified?

    Returns:
      - original_score / modified_score
      - score_delta + direction ("higher" | "lower" | "unchanged")
      - shap_delta: which features changed attribution most
    """
    _require_student_ranker()
    result = state.student_ranker_explainer.whatif(
        features=req.features,
        overrides=req.overrides,
    )
    return {"learner_id": req.learner_id, **result.to_dict()}


# ── /recommend/instructor ─────────────────────────────────────────────────────

@app.post("/recommend/instructor")
def recommend_instructor(req: InstructorRecoRequest):
    """
    Score and rank candidate learner-intervention assignments for an instructor.
    Same explainability surface as the student endpoint.
    """
    _require_instructor_ranker()
    if not req.items:
        raise HTTPException(422, "Provide at least one item to score")

    rows = [_merge_item(it) for it in req.items]
    scored = state.instructor_ranker_explainer.score_items(
        items=rows,
        top_k=req.top_k,
        include_shap=req.include_shap,
    )

    # Diversity by intervention_type
    _i_id_to_feats = {
        str(it.item_id) if it.item_id else str(i): it.features
        for i, it in enumerate(req.items)
    }
    _int_types = [
        str(_i_id_to_feats.get(s.item_id, {}).get(
            "intervention_type",
            _i_id_to_feats.get(s.item_id, {}).get("recommended_content_type", ""),
        ))
        for s in scored
    ]
    _int_types   = [t for t in _int_types if t]
    _i_diversity = round(len(set(_int_types)) / len(scored), 4) if scored else 0.0
    _i_div_warn  = (
        "Low diversity: all interventions are the same type. Consider diversifying approach."
        if _i_diversity < 0.5 and scored
        else None
    )

    i_resp = {
        "instructor_id":     req.instructor_id,
        "top_k":             req.top_k,
        "recommendations":   [s.to_dict() for s in scored],
        "diversity_score":   _i_diversity,
        "diversity_warning": _i_div_warn,
    }

    # Fairness audit across instructor_department and instructor_archetype
    if state.fairness_auditor is not None and scored:
        _i_feats  = [_i_id_to_feats.get(s.item_id, {}) for s in scored]
        _i_scores = [s.score for s in scored]
        _i_audit  = FairnessAuditor(
            protected_features=["instructor_department", "instructor_archetype",
                                 "instructor_teaching_style"]
        ).audit(_i_feats, _i_scores)
        i_resp["fairness_audit"] = _i_audit.to_dict()

    return i_resp


@app.post("/recommend/instructor/explain")
def recommend_instructor_explain(req: InstructorRecoExplainRequest):
    """
    Full XAI explanation for why an intervention/resource was assigned to a
    learner cohort.  Same explainability stack as the student endpoint.
    """
    _require_instructor_ranker()
    explanation = state.instructor_ranker_explainer.explain(
        features=req.features,
        item_id=req.item_id,
    )

    # Intervention metadata passthrough
    _INSTR_META_KEYS = (
        "intervention_type", "intervention_urgency",
        "recommended_content_type", "estimated_effort_hours",
        "student_dropout_risk_score", "student_risk_trajectory",
        "instructor_archetype", "instructor_teaching_style",
        "instructor_department", "cohort_avg_dropout_rate",
    )
    _instr_meta = {k: req.features[k] for k in _INSTR_META_KEYS if k in req.features}

    # Feature 1: LLM narration with recommendation context
    if state.llm_narrator is not None:
        _reco_dict = {
            "instructor_id": req.instructor_id,
            **explanation.to_dict(),
            **_instr_meta,
        }
        _narration = state.llm_narrator.narrate(
            explain_resp = _reco_dict,
            learner_id   = req.instructor_id,
            audience     = req.audience,
            context_type = "recommendation",
        )
        explanation.narratives = _narration.to_dict()

    return {"instructor_id": req.instructor_id, **explanation.to_dict(), **_instr_meta}


@app.post("/recommend/instructor/whatif")
def recommend_instructor_whatif(req: InstructorRecoWhatIfRequest):
    """
    Counterfactual for instructor recommendations: how does assignment priority
    change if feature values are adjusted?
    """
    _require_instructor_ranker()
    result = state.instructor_ranker_explainer.whatif(
        features=req.features,
        overrides=req.overrides,
    )
    return {"instructor_id": req.instructor_id, **result.to_dict()}


# ──────────────────────────────────────────────────────────────────────────────
# Instructor roster + per-student recommendation endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/instructor/students")
def instructor_students(current_user=Depends(get_current_user)):
    """
    Return enrolled students for the authenticated instructor with their
    latest snapshot features + live GBM risk score.
    """
    from backend.app.auth.database import get_db
    from backend.app.auth.models import (
        User, InstructorProfile, CourseEnrollment, LearnerProfile, StudentSnapshot
    )
    db = next(get_db())
    try:
        if current_user.role not in ("instructor", "admin"):
            raise HTTPException(403, "Instructor or admin access required")

        # Admin sees all snapshots; instructor sees only their enrolled students
        if current_user.role == "admin":
            snaps = db.query(StudentSnapshot).all()
        else:
            ip = db.query(InstructorProfile).filter(
                InstructorProfile.user_id == current_user.id
            ).first()
            if ip is None:
                return {"students": []}
            enrollments = db.query(CourseEnrollment).filter(
                CourseEnrollment.instructor_profile_id == ip.id
            ).all()
            learner_profile_ids = [e.learner_profile_id for e in enrollments]
            learner_ids = [
                lp.learner_id for lp in
                db.query(LearnerProfile).filter(
                    LearnerProfile.id.in_(learner_profile_ids)
                ).all()
            ]
            snaps = db.query(StudentSnapshot).filter(
                StudentSnapshot.learner_id.in_(learner_ids)
            ).all()

        result = []
        for snap in snaps:
            fd = snap.to_features_dict()
            live_risk = snap.dropout_risk_score  # use cached value; avoids heavy model call
            if state.gbm_model is not None:
                try:
                    import numpy as _np
                    _X = _np.array([[fd.get(n, 0.0) for n in state.feature_names[:len(fd)]]])
                    live_risk = round(float(state.gbm_model.predict_proba(_X)[0][1]), 4)
                    # Update cached value in place
                    snap.dropout_risk_score = live_risk
                except Exception:
                    pass
            result.append({
                "learner_id":       snap.learner_id,
                "display_name":     snap.display_name or snap.learner_id,
                "dropout_risk_score": live_risk,
                "risk_label":       _risk_label(live_risk) if live_risk is not None else "Unknown",
                "risk_trajectory":  snap.risk_trajectory,
                "current_module":   snap.current_module,
                "current_presentation": snap.current_presentation,
                "features":         fd,
            })
        db.commit()
        return {"students": result}
    finally:
        db.close()


@app.post("/instructor/recommend/{learner_id}")
def instructor_recommend_for_student(learner_id: str, current_user=Depends(get_current_user)):
    """
    Run the instructor ranker for a specific enrolled student.
    Returns ranked (module, presentation, intervention) recommendations.
    """
    _require_instructor_ranker()
    from backend.app.auth.database import get_db
    from backend.app.auth.models import StudentSnapshot, InstructorProfile
    db = next(get_db())
    try:
        snap = db.query(StudentSnapshot).filter(
            StudentSnapshot.learner_id == learner_id
        ).first()
        if snap is None:
            raise HTTPException(404, f"No snapshot for learner_id={learner_id!r}")

        ip = db.query(InstructorProfile).filter(
            InstructorProfile.user_id == current_user.id
        ).first()
        instr_feats = {
            "instructor_archetype":       "Socratic Guide",
            "instructor_department":      getattr(ip, "department", "Humanities") if ip else "Humanities",
            "instructor_teaching_style":  "socratic",
            "instructor_experience_years": 5,
            "instructor_avg_cohort_size": 30,
            "instructor_intervention_intensity": 0.6,
            "instructor_past_success_rate": 0.7,
            "instructor_confidence_score": 0.75,
        }
        student_feats = {
            "student_id": learner_id,
            "learner_id": learner_id,
            "student_current_module":       snap.current_module or "CCC",
            "student_current_presentation": snap.current_presentation or "2014J",
            "student_dropout_risk_score":   snap.dropout_risk_score or 0.5,
            "student_risk_trajectory":      snap.risk_trajectory or "stable",
            "student_learning_momentum":    snap.video_completion_rate,
            "student_quiz_avg_score":       snap.quiz_avg_score,
            "student_assignment_submission_rate": snap.assignment_submission_rate,
            "student_missed_deadlines":     snap.missed_deadlines_count,
            "student_days_inactive":        snap.days_since_last_activity,
            "student_week_in_course":       snap.current_week_in_course,
            "student_vs_cohort_quiz_delta": snap.quiz_avg_score - 65.0,
            "student_peer_collab_readiness": snap.forum_posts_count / 10.0,
        }
        merged = {**instr_feats, **student_feats,
                  "cohort_signal_score": 0.5, "cohort_avg_quiz_score": 65.0,
                  "cohort_avg_dropout_rate": 0.3, "cohort_size": 30,
                  "cohort_engagement_percentile": 0.5}

        scored = state.instructor_ranker_explainer.score_items(
            items=[merged], top_k=5, include_shap=True
        )
        return {
            "learner_id": learner_id,
            "student_name": snap.display_name or learner_id,
            "recommendations": [s.to_dict() for s in scored],
        }
    finally:
        db.close()


@app.post("/instructor/recommend/{learner_id}/explain")
def instructor_recommend_explain_for_student(
    learner_id: str,
    req: dict,
    current_user=Depends(get_current_user),
):
    """Full XAI explanation for a specific instructor recommendation for a student."""
    _require_instructor_ranker()
    item_id  = req.get("item_id", "")
    features = req.get("features", {})
    explanation = state.instructor_ranker_explainer.explain(
        features=features, item_id=item_id
    )
    resp = explanation.to_dict()
    if state.llm_narrator is not None:
        try:
            narr = state.llm_narrator.narrate_instructor(resp, learner_id=learner_id)
            resp["narratives"] = {"instructor": narr}
        except Exception as _e:
            log.warning("LLM narration failed for instructor reco explain: %s", _e)
    return {"learner_id": learner_id, **resp}


@app.post("/instructor/whatif/{learner_id}")
def instructor_whatif_for_student(
    learner_id: str,
    req: dict,
    current_user=Depends(get_current_user),
):
    """
    What-if for a student under instructor view: apply feature overrides
    to a student's snapshot and return the new predicted risk score.
    """
    _require_model()
    from backend.app.auth.database import get_db
    from backend.app.auth.models import StudentSnapshot
    db = next(get_db())
    try:
        snap = db.query(StudentSnapshot).filter(
            StudentSnapshot.learner_id == learner_id
        ).first()
        if snap is None:
            raise HTTPException(404, f"No snapshot for learner_id={learner_id!r}")
        base_fd = snap.to_features_dict()
        overrides = req.get("overrides", {})
        merged_fd = {**base_fd, **overrides}
        import numpy as _np
        _X = _np.array([[merged_fd.get(n, 0.0) for n in state.feature_names]])
        new_risk = round(float(state.gbm_model.predict_proba(_X)[0][1]), 4)
        delta = round(new_risk - (snap.dropout_risk_score or 0.0), 4)
        shap_result = state.shap_explainer.explain(merged_fd)
        return {
            "learner_id":    learner_id,
            "base_risk":     snap.dropout_risk_score,
            "new_risk":      new_risk,
            "risk_delta":    delta,
            "risk_label":    _risk_label(new_risk),
            "top_features":  list(shap_result.top_features[:5]),
            "shap_values":   {k: round(v, 5) for k, v in shap_result.shap_values.items()},
        }
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
# Student study-tips endpoint (LLM-grounded in SHAP explanation)
# ──────────────────────────────────────────────────────────────────────────────

_STUDY_TIPS_PROMPT = """You are an educational AI assistant helping a student understand their dropout risk.
Given their XAI explanation, provide exactly 4 specific, actionable study tips.
Grounded in the features below — do NOT invent features not listed.

Risk score: {risk_score} ({risk_label})
Top risk factors: {top_features}
Anchor rule: {anchor_rule}
SHAP summary (feature → impact): {shap_summary}

Return ONLY valid JSON: {{"summary": "one sentence", "tips": ["tip1", "tip2", "tip3", "tip4"]}}
"""


@app.post("/student/study-tips")
def student_study_tips(req: dict, current_user=Depends(get_current_user)):
    """
    LLM-generated study tips grounded in the student's SHAP explanation.
    """
    risk_score   = req.get("risk_score", 0.5)
    risk_label   = req.get("risk_label", "Medium")
    top_features = req.get("top_features", [])
    shap_values  = req.get("shap_values", {})
    anchor_rule  = req.get("anchor_rule", {})

    shap_summary = "; ".join(
        f"{k}={v:+.3f}" for k, v in sorted(shap_values.items(), key=lambda x: -abs(x[1]))[:6]
    ) if shap_values else "N/A"
    anchor_text = anchor_rule.get("rule", "N/A") if isinstance(anchor_rule, dict) else str(anchor_rule)

    # Rule-based fallback (always available)
    fallback_tips = []
    for feat in (top_features or list(shap_values.keys()))[:4]:
        sv = shap_values.get(feat, 0)
        if sv > 0:
            fallback_tips.append(f"Focus on improving '{feat.replace('_', ' ')}' — it is currently increasing your dropout risk.")
        else:
            fallback_tips.append(f"Keep up the good work on '{feat.replace('_', ' ')}' — it is helping reduce your risk.")
    while len(fallback_tips) < 4:
        fallback_tips.append("Log in regularly and engage with course materials daily.")

    if state.llm_narrator is None or not state.llm_narrator.available:
        return {
            "summary": f"Your current dropout risk is {risk_label} ({risk_score:.0%}). Here are personalised tips based on your model explanation.",
            "tips": fallback_tips,
            "source": "rule_based",
        }

    prompt = _STUDY_TIPS_PROMPT.format(
        risk_score=f"{risk_score:.0%}", risk_label=risk_label,
        top_features=", ".join(top_features[:5]),
        anchor_rule=anchor_text, shap_summary=shap_summary,
    )
    try:
        import json as _json
        from groq import Groq as _Groq
        _client = _Groq(api_key=state.llm_narrator.api_key)
        _resp = _client.chat.completions.create(
            model=state.llm_narrator.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=512,
        )
        raw = (_resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json.loads(raw)
        return {**parsed, "source": "llm"}
    except Exception as _e:
        log.warning("study-tips LLM call failed: %s", _e)
        return {
            "summary": f"Your dropout risk is {risk_label} ({risk_score:.0%}).",
            "tips": fallback_tips,
            "source": "rule_based",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Compare narration endpoint — LLM explains GBM vs LSTM bipartite graph
# ──────────────────────────────────────────────────────────────────────────────

_COMPARE_NARRATE_PROMPT = """You are an XAI system explaining a model comparison to a {audience}.
GBM (gradient boosting) risk score: {gbm_score} ({gbm_label})
LSTM (sequential/temporal) risk score: {lstm_score} ({lstm_label})
Disagreement flag: {disagreement}
GBM top features: {gbm_top}

Explain in plain language:
1. What each model captures (GBM = tabular patterns; LSTM = temporal trends over weeks)
2. Why they agree or disagree on this student
3. What the disagreement (if any) means practically for the {audience}
4. One concrete recommendation based on this comparison

Return ONLY valid JSON: {{"summary": "...", "gbm_insight": "...", "lstm_insight": "...", "disagreement_meaning": "...", "recommendation": "..."}}
"""


@app.post("/compare/narrate")
def compare_narrate(req: dict):
    """LLM narration of the GBM vs LSTM bipartite comparison graph."""
    gbm_score     = req.get("gbm_score", 0.5)
    lstm_score    = req.get("lstm_score")
    gbm_label     = req.get("gbm_risk_label", _risk_label(gbm_score))
    lstm_label    = req.get("lstm_risk_label", _risk_label(lstm_score) if lstm_score else "N/A")
    disagreement  = req.get("disagreement_flag", False)
    gbm_top       = req.get("gbm_top3_features", [])
    audience      = req.get("audience", "student")

    fallback = {
        "summary": f"GBM predicts {gbm_label} risk ({gbm_score:.0%})" + (
            f", LSTM predicts {lstm_label} risk ({lstm_score:.0%})." if lstm_score else "."
        ),
        "gbm_insight": "The gradient boosting model uses tabular engagement and performance features.",
        "lstm_insight": "The LSTM model tracks weekly engagement trends over time." if lstm_score else "LSTM not available.",
        "disagreement_meaning": "The models disagree — consider both perspectives before acting." if disagreement else "Both models agree on the risk level.",
        "recommendation": "Focus on the top risk features identified by GBM.",
        "source": "rule_based",
    }

    if state.llm_narrator is None or not state.llm_narrator.available or lstm_score is None:
        return fallback

    prompt = _COMPARE_NARRATE_PROMPT.format(
        audience=audience, gbm_score=f"{gbm_score:.0%}", gbm_label=gbm_label,
        lstm_score=f"{lstm_score:.0%}", lstm_label=lstm_label,
        disagreement=disagreement, gbm_top=", ".join(gbm_top),
    )
    try:
        import json as _json
        from groq import Groq as _Groq
        _client = _Groq(api_key=state.llm_narrator.api_key)
        _resp = _client.chat.completions.create(
            model=state.llm_narrator.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4, max_tokens=600,
        )
        raw = (_resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = _json.loads(raw)
        return {**parsed, "source": "llm"}
    except Exception as _e:
        log.warning("compare/narrate LLM call failed: %s", _e)
        return fallback


# ──────────────────────────────────────────────────────────────────────────────
# Admin enrollment management endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/admin/instructors")
def admin_list_instructors(current_user=Depends(get_current_user)):
    """List all instructors with their enrolled student count."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    from backend.app.auth.database import get_db
    from backend.app.auth.models import User, InstructorProfile, CourseEnrollment
    db = next(get_db())
    try:
        instructors = db.query(User).filter(User.role == "instructor", User.is_active == True).all()
        result = []
        for u in instructors:
            ip = db.query(InstructorProfile).filter(InstructorProfile.user_id == u.id).first()
            count = 0
            if ip:
                count = db.query(CourseEnrollment).filter(
                    CourseEnrollment.instructor_profile_id == ip.id
                ).count()
            result.append({
                "user_id":    u.id, "username": u.username,
                "full_name":  u.full_name, "department": getattr(ip, "department", None),
                "student_count": count,
                "profile_id": ip.id if ip else None,
            })
        return {"instructors": result}
    finally:
        db.close()


@app.get("/admin/enrollments")
def admin_list_enrollments(current_user=Depends(get_current_user)):
    """List all course enrollments."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    from backend.app.auth.database import get_db
    from backend.app.auth.models import CourseEnrollment, LearnerProfile, InstructorProfile, User
    db = next(get_db())
    try:
        enrollments = db.query(CourseEnrollment).all()
        result = []
        for e in enrollments:
            lp = db.query(LearnerProfile).filter(LearnerProfile.id == e.learner_profile_id).first()
            ip = db.query(InstructorProfile).filter(InstructorProfile.id == e.instructor_profile_id).first()
            student_user = db.query(User).filter(User.id == lp.user_id).first() if lp else None
            result.append({
                "id": e.id, "course_id": e.course_id,
                "learner_id": lp.learner_id if lp else None,
                "student_name": student_user.full_name if student_user else None,
                "instructor_profile_id": e.instructor_profile_id,
                "enrolled_at": e.enrolled_at.isoformat() if e.enrolled_at else None,
            })
        return {"enrollments": result}
    finally:
        db.close()


@app.post("/admin/enrollments")
def admin_add_enrollment(req: dict, current_user=Depends(get_current_user)):
    """Enroll a student (by learner_id) under an instructor (by profile_id) in a course."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    from backend.app.auth.database import get_db
    from backend.app.auth.models import CourseEnrollment, LearnerProfile
    db = next(get_db())
    try:
        learner_id            = req.get("learner_id")
        instructor_profile_id = req.get("instructor_profile_id")
        course_id             = req.get("course_id", "CCC")
        lp = db.query(LearnerProfile).filter(LearnerProfile.learner_id == learner_id).first()
        if lp is None:
            raise HTTPException(404, f"learner_id={learner_id!r} not found")
        existing = db.query(CourseEnrollment).filter(
            CourseEnrollment.learner_profile_id == lp.id,
            CourseEnrollment.instructor_profile_id == instructor_profile_id,
        ).first()
        if existing:
            raise HTTPException(409, "Enrollment already exists")
        e = CourseEnrollment(
            learner_profile_id=lp.id,
            instructor_profile_id=instructor_profile_id,
            course_id=course_id,
        )
        db.add(e)
        db.commit()
        return {"id": e.id, "learner_id": learner_id, "course_id": course_id}
    finally:
        db.close()


@app.delete("/admin/enrollments/{enrollment_id}")
def admin_delete_enrollment(enrollment_id: str, current_user=Depends(get_current_user)):
    """Remove a course enrollment."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    from backend.app.auth.database import get_db
    from backend.app.auth.models import CourseEnrollment
    db = next(get_db())
    try:
        e = db.query(CourseEnrollment).filter(CourseEnrollment.id == enrollment_id).first()
        if e is None:
            raise HTTPException(404, "Enrollment not found")
        db.delete(e)
        db.commit()
        return {"deleted": enrollment_id}
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────────
# SPA catch-all — MUST be the very last route registered
# ──────────────────────────────────────────────────────────────────────────────
# This serves index.html for any path not matched by an API route above,
# allowing React Router to handle client-side navigation (e.g. /student,
# /what-if, /action-plan). Only active when the built frontend is present.

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    """Serve the React SPA for all non-API routes."""
    if not _INDEX_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Frontend not built. "
                "Run 'npm run build' in the frontend/ directory, "
                "or start the dev server with 'npm run dev'."
            ),
        )
    return FileResponse(str(_INDEX_HTML))

