"""
Hot Reload (Step 5)
====================
Atomically reloads models and ALL dependent explainers from disk into the
running FastAPI AppState — without restarting the uvicorn process.

Safety: builds all new components in a staging dict. Only swaps into
AppState if every component initialises successfully. On any failure,
AppState is untouched and the current production model remains live.

Public API
----------
hot_reload(state, data_dir, models_dir, device) -> HotReloadResult
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


@dataclass
class HotReloadResult:
    success:       bool
    model_version: str
    feature_count: int
    components_reloaded: list[str]
    error:         Optional[str] = None
    elapsed_sec:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "success":             self.success,
            "model_version":       self.model_version,
            "feature_count":       self.feature_count,
            "components_reloaded": self.components_reloaded,
            "error":               self.error,
            "elapsed_sec":         round(self.elapsed_sec, 2),
        }


def hot_reload(state, data_dir: Path, models_dir: Path, device) -> HotReloadResult:
    """
    Reload model artifacts from disk and re-initialise all dependent components.
    Performs an atomic swap: AppState is only mutated after ALL inits succeed.

    Parameters
    ----------
    state      : AppState instance (the FastAPI global state)
    data_dir   : Path to data/ directory (for train.pkl)
    models_dir : Path to models/ directory (gbm.pkl, rf.pkl, lstm.pt, etc.)
    device     : torch.device
    """
    t_start = time.time()
    staging: dict = {}
    reloaded: list[str] = []

    # Lazy imports to avoid circular imports at module load time
    from backend.app.explainers.anchors_explainer import AnchorsExplainer
    from backend.app.explainers.archipelago import ArchipelagoExplainer
    from backend.app.explainers.dice_explainer import DiCEExplainer
    from backend.app.explainers.prototype_explainer import PrototypeExplainer
    from backend.app.explainers.shap_explainer import SHAPExplainer
    from backend.app.evaluator.trust_scorer import TrustScorer
    from backend.app.evaluator.uncertainty_estimator import UncertaintyEstimator
    from backend.app.causal.causal_annotator import CausalAnnotator
    from backend.app.prescriptor.action_ranker import ActionRanker
    from backend.app.mlops.drift_monitor import DriftMonitor
    from backend.app.tracker.drift_detector import ExplanationDriftDetector

    def _load_pkl(name: str) -> dict:
        p = models_dir / name
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        with open(p, "rb") as f:
            return pickle.load(f)

    def _load_data(name: str) -> dict:
        p = data_dir / name
        if not p.exists():
            raise FileNotFoundError(f"{p} not found")
        with open(p, "rb") as f:
            return pickle.load(f)

    try:
        # ── GBM ──────────────────────────────────────────────────
        gbm_blob = _load_pkl("gbm.pkl")
        staging["gbm_model"]     = gbm_blob["model"]
        staging["feature_names"] = gbm_blob["feature_names"]
        reloaded.append("gbm_model")

        # ── RF ───────────────────────────────────────────────────
        try:
            staging["rf_model"] = _load_pkl("rf.pkl")["model"]
            reloaded.append("rf_model")
        except FileNotFoundError:
            staging["rf_model"] = None
            log.warning("hot_reload: rf.pkl not found — RF model not reloaded")

        # ── Training data ────────────────────────────────────────
        train_blob = _load_data("train.pkl")
        staging["X_train"]     = train_blob["X"]
        staging["y_train"]     = train_blob["y"]
        staging["learner_ids"] = train_blob.get("learner_ids")
        reloaded.append("training_data")

        gbm_model     = staging["gbm_model"]
        feature_names = staging["feature_names"]
        X_train       = staging["X_train"]
        y_train       = staging["y_train"]
        learner_ids   = staging["learner_ids"]

        # ── Model version ────────────────────────────────────────
        version_path = models_dir / "model_version.json"
        if version_path.exists():
            with open(version_path) as f:
                staging["model_version"] = f"gbm-v{json.load(f).get('version', 1)}"
        else:
            staging["model_version"] = "gbm-v1"

        # ── SHAP ─────────────────────────────────────────────────
        lstm_model = state.lstm_model   # LSTM doesn't change on retrain
        staging["shap_explainer"] = SHAPExplainer(
            gbm_model     = gbm_model,
            feature_names = feature_names,
            lstm_model    = lstm_model,
        )
        reloaded.append("shap_explainer")

        # ── Archipelago ──────────────────────────────────────────
        staging["archipelago_explainer"] = ArchipelagoExplainer(
            tree_explainer = staging["shap_explainer"].tree_explainer,
            feature_names  = feature_names,
        )
        reloaded.append("archipelago_explainer")

        # ── DiCE ─────────────────────────────────────────────────
        staging["dice_explainer"] = DiCEExplainer(
            model         = gbm_model,
            X_train       = X_train,
            feature_names = feature_names,
            y_train       = y_train,
        )
        reloaded.append("dice_explainer")

        # ── Anchors ──────────────────────────────────────────────
        def _predict_fn(X):
            return (gbm_model.predict_proba(X)[:, 1] >= 0.5).astype(int)

        staging["anchors_explainer"] = AnchorsExplainer(
            predict_fn    = _predict_fn,
            X_train       = X_train,
            feature_names = feature_names,
        )
        reloaded.append("anchors_explainer")

        # ── Prototypes ───────────────────────────────────────────
        staging["prototype_explainer"] = PrototypeExplainer(
            X_train       = X_train,
            y_train       = y_train,
            learner_ids   = learner_ids if learner_ids is not None else np.arange(len(X_train)),
            feature_names = feature_names,
        )
        reloaded.append("prototype_explainer")

        # ── MAPIE Uncertainty ────────────────────────────────────
        staging["uncertainty_estimator"] = UncertaintyEstimator(
            model = gbm_model,
            X_cal = X_train,
            y_cal = y_train,
        )
        reloaded.append("uncertainty_estimator")

        # ── Causal annotator ─────────────────────────────────────
        staging["causal_annotator"] = CausalAnnotator(
            X_train       = X_train,
            y_train       = y_train,
            feature_names = feature_names,
        )
        reloaded.append("causal_annotator")

        # ── Action ranker ─────────────────────────────────────────
        staging["action_ranker"] = ActionRanker(
            feature_names    = feature_names,
            X_train          = X_train,
            causal_annotator = staging["causal_annotator"],
        )
        reloaded.append("action_ranker")

        # ── Drift monitor ─────────────────────────────────────────
        staging["drift_monitor"] = DriftMonitor(
            reference_data = X_train,
            feature_names  = feature_names,
        )
        reloaded.append("drift_monitor")

        # ── Explanation drift detector ────────────────────────────
        staging["drift_detector"] = ExplanationDriftDetector()
        reloaded.append("drift_detector")

        # ── Trust scorer (stateless — always fresh) ───────────────
        staging["trust_scorer"] = TrustScorer()
        reloaded.append("trust_scorer")

    except Exception as exc:
        elapsed = time.time() - t_start
        log.error("hot_reload FAILED (AppState unchanged): %s", exc, exc_info=True)
        return HotReloadResult(
            success=False, model_version=state.model_version,
            feature_count=len(state.feature_names),
            components_reloaded=reloaded,
            error=str(exc),
            elapsed_sec=elapsed,
        )

    # ── Atomic swap into AppState ─────────────────────────────────────────────
    state.gbm_model            = staging["gbm_model"]
    state.rf_model             = staging["rf_model"]
    state.feature_names        = staging["feature_names"]
    state.X_train              = staging["X_train"]
    state.y_train              = staging["y_train"]
    state.learner_ids          = staging["learner_ids"]
    state.model_version        = staging["model_version"]
    state.shap_explainer       = staging["shap_explainer"]
    state.archipelago_explainer = staging["archipelago_explainer"]
    state.dice_explainer       = staging["dice_explainer"]
    state.anchors_explainer    = staging["anchors_explainer"]
    state.prototype_explainer  = staging["prototype_explainer"]
    state.uncertainty_estimator = staging["uncertainty_estimator"]
    state.causal_annotator     = staging["causal_annotator"]
    state.action_ranker        = staging["action_ranker"]
    state.drift_monitor        = staging["drift_monitor"]
    state.drift_detector       = staging["drift_detector"]
    state.trust_scorer         = staging["trust_scorer"]

    elapsed = time.time() - t_start
    log.info(
        "hot_reload COMPLETE  version=%s  features=%d  components=%d  elapsed=%.2fs",
        state.model_version, len(state.feature_names), len(reloaded), elapsed,
    )
    return HotReloadResult(
        success=True,
        model_version=state.model_version,
        feature_count=len(state.feature_names),
        components_reloaded=reloaded,
        elapsed_sec=elapsed,
    )
