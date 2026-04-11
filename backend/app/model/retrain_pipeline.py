"""
Retrain Pipeline (Steps 4 + 9)
================================
Orchestrates the full feedback-driven retraining sequence:

  1. Aggregate implicit events → per-learner engagement signals
  2. Train / update Engagement Autoencoder
  3. Extract latent scores (3-dim) for every known learner
  4. Build augmented feature matrix: [OULAD_12 + latent_3] = 15 features
  5. Retrain GBM + RF on augmented matrix
  6. Validate against held-out test set (regression guard)
  7. If metrics pass: save artifacts, bump version
  8. Signal hot-reload

Cold-start handling (bootstrap):
  - Historical OULAD learners with no interaction events get latent = [0, 0, 0]
  - Autoencoder trains on whatever events exist (even sparse is OK)
  - GBM learns to weight latent features as real events accumulate
  - First retrain: latent columns are mostly zeros → GBM barely changes vs baseline

Public API
----------
RetrainPipeline(event_store, feedback_store, data_dir, models_dir)
    .run(trigger)         -> RetrainResult
    .can_run()            -> bool   (enough new data since last retrain)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from backend.app.model.engagement_model import EngagementModel
from backend.app.model.implicit_aggregator import (
    ImplicitAggregator,
    LATENT_FEATURE_NAMES,
)
from backend.app.mlops.mlflow_config import configure_mlflow

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Validation gates
# ──────────────────────────────────────────────────────────────────────────────
AUC_DROP_TOLERANCE     = 0.02    # allow at most 2% AUC drop vs current production
BRIER_RISE_TOLERANCE   = 0.03    # allow at most 3% Brier rise
SHAP_FIDELITY_MAX      = 0.05    # SHAP sum error threshold
MIN_TRAIN_SAMPLES      = 1_000   # minimum rows required to retrain
MAX_COLD_START_FRAC    = 0.95    # reject if >95% of learners are cold-start
CLASS_BALANCE_MIN      = 0.05    # minority class must be ≥5% of total
CLASS_BALANCE_MAX      = 0.95    # majority class must be ≤95% of total
STRICT_MODE            = os.getenv("RETRAIN_STRICT_MODE", "false").lower() == "true"


@dataclass
class RetrainResult:
    success:          bool
    trigger:          str
    model_version:    str
    feature_names:    list[str]
    n_features:       int
    n_train_samples:  int
    metrics:          dict = field(default_factory=dict)
    prev_metrics:     dict = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    engagement_model_trained: bool = False
    n_learners_with_events:   int  = 0
    cold_start_fraction:      float = 0.0
    training_time_sec:        float = 0.0
    message:          str = ""

    def to_dict(self) -> dict:
        return {
            "success":                   self.success,
            "trigger":                   self.trigger,
            "model_version":             self.model_version,
            "feature_names":             self.feature_names,
            "n_features":                self.n_features,
            "n_train_samples":           self.n_train_samples,
            "metrics":                   self.metrics,
            "prev_metrics":              self.prev_metrics,
            "rejection_reason":          self.rejection_reason,
            "engagement_model_trained":  self.engagement_model_trained,
            "n_learners_with_events":    self.n_learners_with_events,
            "cold_start_fraction":       round(self.cold_start_fraction, 3),
            "training_time_sec":         round(self.training_time_sec, 1),
            "message":                   self.message,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class RetrainPipeline:
    def __init__(
        self,
        event_store,
        feedback_store,
        data_dir:   Path,
        models_dir: Path,
        mlruns_dir: Optional[Path] = None,
    ):
        self._events    = event_store
        self._feedback  = feedback_store
        self._data_dir  = Path(data_dir)
        self._models_dir = Path(models_dir)
        self._mlruns_dir = Path(mlruns_dir) if mlruns_dir else Path(models_dir).parent / "mlruns"
        self._eng_model_dir = self._models_dir / "engagement"

    # ── Public ────────────────────────────────────────────────────────────────

    def can_run(self, min_new_events: int = 50) -> bool:
        """Return True if there are enough new events to warrant a retrain."""
        return self._events.count() >= min_new_events

    def run(self, trigger: str = "manual") -> RetrainResult:
        """
        Execute the full retrain pipeline.
        Returns a RetrainResult describing success/failure + metrics.
        """
        t_start = time.time()
        log.info("RetrainPipeline.run  trigger=%s", trigger)

        # ── 1. Load base training data (OULAD 12-feature split) ──
        try:
            train_blob = self._load_pkl("train.pkl")
            test_blob  = self._load_pkl("test.pkl")
        except FileNotFoundError as e:
            return RetrainResult(
                success=False, trigger=trigger, model_version="",
                feature_names=[], n_features=0, n_train_samples=0,
                rejection_reason=f"Base data missing: {e}",
                message="Run data_loader.py first.",
            )

        X_tr_base = train_blob["X"]
        y_tr      = train_blob["y"]
        X_te_base = test_blob["X"]
        y_te      = test_blob["y"]
        learner_ids_tr = train_blob.get("learner_ids", np.arange(len(X_tr_base)))
        learner_ids_te = test_blob.get("learner_ids",  np.arange(len(X_te_base)))
        base_feature_names: list[str] = train_blob["feature_names"]

        # ── 2. Load current production metrics (for validation gate) ──
        prev_metrics = self._load_prev_metrics()

        # ── 3. Aggregate engagement signals ──
        aggregator = ImplicitAggregator(
            event_store    = self._events,
            feedback_store = self._feedback,
        )
        signals_dict = aggregator.aggregate_all()
        eng_matrix, eng_feature_names, eng_learner_ids = aggregator.to_matrix(signals_dict)

        n_with_events = sum(
            1 for s in signals_dict.values() if not s.is_cold_start()
        )
        log.info(
            "Engagement signals: total=%d  with_real_events=%d  cold_start=%d",
            len(signals_dict), n_with_events, len(signals_dict) - n_with_events,
        )

        # ── 4. Train / update Engagement Autoencoder ──
        eng_model = EngagementModel()
        eng_trained = False
        if len(eng_matrix) >= 5:
            eng_model.fit(eng_matrix, epochs=60)
            eng_model.save(self._eng_model_dir)
            eng_trained = True
        elif self._eng_model_dir.exists():
            log.info("Too few samples for fresh autoencoder — loading existing")
            eng_model = EngagementModel.load(self._eng_model_dir)
            eng_trained = eng_model.is_fitted
        else:
            log.warning(
                "No existing engagement model and too few samples (%d). "
                "Latent features will be zeros for all learners.",
                len(eng_matrix),
            )

        # ── 5. Extract latent scores + augment feature matrices ──
        X_tr_aug, X_te_aug, augmented_feature_names = self._augment_matrices(
            X_tr_base, X_te_base,
            learner_ids_tr, learner_ids_te,
            signals_dict, base_feature_names,
            eng_model,
        )

        log.info(
            "Augmented feature matrix: %d → %d features  "
            "train=%d  test=%d",
            len(base_feature_names), len(augmented_feature_names),
            len(X_tr_aug), len(X_te_aug),
        )

        # ── 6. Retrain GBM + RF ──
        gbm_model, gbm_metrics = self._train_gbm(
            X_tr_aug, y_tr, X_te_aug, y_te, augmented_feature_names
        )
        rf_model, rf_metrics = self._train_rf(
            X_tr_aug, y_tr, X_te_aug, y_te, augmented_feature_names
        )

        # ── 7. Validation gates ──
        cold_frac_pre = (len(signals_dict) - n_with_events) / max(len(signals_dict), 1)
        rejection = self._validate(
            gbm_metrics, prev_metrics,
            X_tr_aug, y_tr,
            cold_frac_pre,
            augmented_feature_names,
        )
        if rejection:
            log.warning("RetrainPipeline REJECTED: %s", rejection)
            return RetrainResult(
                success=False, trigger=trigger,
                model_version="rejected",
                feature_names=augmented_feature_names,
                n_features=len(augmented_feature_names),
                n_train_samples=len(X_tr_aug),
                metrics=gbm_metrics,
                prev_metrics=prev_metrics,
                rejection_reason=rejection,
                engagement_model_trained=eng_trained,
                n_learners_with_events=n_with_events,
                cold_start_fraction=(len(signals_dict) - n_with_events) / max(len(signals_dict), 1),
                training_time_sec=time.time() - t_start,
                message="Validation gate failed — keeping current production model.",
            )

        # ── 8. Save artifacts ──
        new_version = self._bump_version()
        self._save_models(gbm_model, rf_model, augmented_feature_names)
        self._save_augmented_splits(
            X_tr_aug, y_tr, learner_ids_tr,
            X_te_aug, y_te, learner_ids_te,
            augmented_feature_names,
        )
        self._save_summary(gbm_metrics, rf_metrics, augmented_feature_names, new_version)
        self._save_manifest(
            version=new_version,
            metrics=gbm_metrics,
            feature_names=augmented_feature_names,
            train_pkl_path=self._data_dir / "train.pkl",
        )

        elapsed = round(time.time() - t_start, 1)
        cold_frac = cold_frac_pre

        log.info(
            "RetrainPipeline.run COMPLETE  version=%s  features=%d  "
            "auc=%.4f  brier=%.4f  time=%.1fs",
            new_version, len(augmented_feature_names),
            gbm_metrics.get("test_auc_roc", 0), gbm_metrics.get("test_brier_score", 0),
            elapsed,
        )
        return RetrainResult(
            success=True, trigger=trigger,
            model_version=new_version,
            feature_names=augmented_feature_names,
            n_features=len(augmented_feature_names),
            n_train_samples=len(X_tr_aug),
            metrics=gbm_metrics,
            prev_metrics=prev_metrics,
            engagement_model_trained=eng_trained,
            n_learners_with_events=n_with_events,
            cold_start_fraction=cold_frac,
            training_time_sec=elapsed,
            message="Retrain successful. Call POST /mlops/reload to activate.",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_pkl(self, name: str) -> dict:
        path = self._data_dir / name
        if not path.exists():
            raise FileNotFoundError(str(path))
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_prev_metrics(self) -> dict:
        summary_path = self._models_dir / "training_summary.json"
        if not summary_path.exists():
            return {}
        with open(summary_path) as f:
            data = json.load(f)
        return data.get("gbm", {})

    def _bump_version(self) -> str:
        version_path = self._models_dir / "model_version.json"
        version = 1
        if version_path.exists():
            with open(version_path) as f:
                version = json.load(f).get("version", 0) + 1
        with open(version_path, "w") as f:
            json.dump({"version": version}, f)
        return f"gbm-v{version}"

    def _augment_matrices(
        self,
        X_tr_base: np.ndarray,
        X_te_base: np.ndarray,
        learner_ids_tr,
        learner_ids_te,
        signals_dict: dict,
        base_feature_names: list[str],
        eng_model: EngagementModel,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        For each split, build latent score columns and concatenate.
        Cold-start learners (not in signals_dict or all-zero) get [0, 0, 0].
        """
        augmented_names = base_feature_names + LATENT_FEATURE_NAMES

        def _get_latents(learner_ids) -> np.ndarray:
            latents = np.zeros((len(learner_ids), len(LATENT_FEATURE_NAMES)), dtype=np.float32)
            if not eng_model.is_fitted:
                return latents
            for i, lid in enumerate(learner_ids):
                lid_str = str(lid)
                if lid_str in signals_dict:
                    vec = signals_dict[lid_str].to_vector()
                    # Only encode non-cold-start learners
                    if any(v != 0.0 for v in vec):
                        latents[i] = eng_model.encode_learner(vec)
            return latents

        lat_tr = _get_latents(learner_ids_tr)
        lat_te = _get_latents(learner_ids_te)

        X_tr_aug = np.hstack([X_tr_base, lat_tr])
        X_te_aug = np.hstack([X_te_base, lat_te])
        return X_tr_aug, X_te_aug, augmented_names

    def _train_gbm(self, X_tr, y_tr, X_te, y_te, feature_names) -> tuple:
        params = {
            "n_estimators":     200,
            "learning_rate":    0.05,
            "max_depth":        4,
            "subsample":        0.8,
            "min_samples_leaf": 20,
            "random_state":     42,
        }
        log.info("Retraining GBM  features=%d  samples=%d", len(feature_names), len(X_tr))
        configure_mlflow(experiment="xai-dropout-risk-retrain")
        with mlflow.start_run(run_name="gbm-retrain"):
            base  = GradientBoostingClassifier(**params)
            base.fit(X_tr, y_tr)
            model = CalibratedClassifierCV(base, method="isotonic", cv=5)
            model.fit(X_tr, y_tr)
            metrics = self._evaluate(model, X_te, y_te, "test_")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model", registered_model_name="gbm-dropout-risk")
        return model, metrics

    def _train_rf(self, X_tr, y_tr, X_te, y_te, feature_names) -> tuple:
        params = {
            "n_estimators":     200,
            "min_samples_leaf": 10,
            "n_jobs":           -1,
            "random_state":     42,
        }
        configure_mlflow(experiment="xai-dropout-risk-retrain")
        with mlflow.start_run(run_name="rf-retrain"):
            base  = RandomForestClassifier(**params)
            base.fit(X_tr, y_tr)
            model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
            model.fit(X_tr, y_tr)
            metrics = self._evaluate(model, X_te, y_te, "test_")
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model", registered_model_name="rf-dropout-risk")
        return model, metrics

    @staticmethod
    def _evaluate(model, X, y, prefix: str = "") -> dict:
        proba = model.predict_proba(X)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        return {
            f"{prefix}auc_roc":       round(roc_auc_score(y, proba), 4),
            f"{prefix}avg_precision": round(average_precision_score(y, proba), 4),
            f"{prefix}f1":            round(f1_score(y, pred), 4),
            f"{prefix}brier_score":   round(brier_score_loss(y, proba), 4),
        }

    def _validate(
        self,
        new_metrics: dict,
        prev_metrics: dict,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        cold_start_fraction: float,
        feature_names: list[str],
    ) -> Optional[str]:
        """
        Return rejection reason string or None if all gates pass.

        Gates (all active in strict mode, only regression gates in normal mode):
          1. Minimum training data volume
          2. Class balance
          3. Cold-start fraction (warn in normal, reject in strict)
          4. Feature schema consistency
          5. AUC regression vs current production
          6. Brier regression vs current production
        """
        reasons: list[str] = []

        # ── 1. Data volume ────────────────────────────────────────────────────
        if len(X_tr) < MIN_TRAIN_SAMPLES:
            reasons.append(
                f"Insufficient training data: {len(X_tr)} rows < {MIN_TRAIN_SAMPLES} minimum"
            )

        # ── 2. Class balance ──────────────────────────────────────────────────
        if len(y_tr) > 0:
            pos_frac = float(y_tr.mean())
            if pos_frac < CLASS_BALANCE_MIN:
                reasons.append(
                    f"Class imbalance: positive fraction {pos_frac:.3f} < {CLASS_BALANCE_MIN} "
                    f"(only {int(y_tr.sum())} dropout samples)"
                )
            elif pos_frac > CLASS_BALANCE_MAX:
                reasons.append(
                    f"Class imbalance: positive fraction {pos_frac:.3f} > {CLASS_BALANCE_MAX}"
                )

        # ── 3. Cold-start fraction ────────────────────────────────────────────
        if cold_start_fraction > MAX_COLD_START_FRAC:
            msg = (
                f"Cold-start fraction {cold_start_fraction:.2%} > {MAX_COLD_START_FRAC:.0%} — "
                f"latent features are effectively zero for most learners"
            )
            if STRICT_MODE:
                reasons.append(msg)
            else:
                log.warning("RETRAIN WARN: %s (proceeding in non-strict mode)", msg)

        # ── 4. Feature schema consistency ─────────────────────────────────────
        expected_schema_path = self._models_dir / "model_manifest.json"
        if expected_schema_path.exists():
            with open(expected_schema_path) as f:
                prev_manifest = json.load(f)
            prev_hash = prev_manifest.get("feature_schema_hash")
            curr_hash = _feature_schema_hash(feature_names)
            if prev_hash and prev_hash != curr_hash:
                reasons.append(
                    f"Feature schema changed: prev_hash={prev_hash[:12]}  "
                    f"curr_hash={curr_hash[:12]}  "
                    f"(features={feature_names})"
                )

        # ── 5+6. Metric regression gates ─────────────────────────────────────
        if prev_metrics:
            prev_auc   = prev_metrics.get("test_auc_roc")
            prev_brier = prev_metrics.get("test_brier_score")
            new_auc    = new_metrics.get("test_auc_roc")
            new_brier  = new_metrics.get("test_brier_score")

            if prev_auc is not None and new_auc is not None:
                if new_auc < prev_auc - AUC_DROP_TOLERANCE:
                    reasons.append(
                        f"AUC regression: {new_auc:.4f} < {prev_auc:.4f} - {AUC_DROP_TOLERANCE} "
                        f"(drop={prev_auc - new_auc:.4f})"
                    )

            if prev_brier is not None and new_brier is not None:
                if new_brier > prev_brier + BRIER_RISE_TOLERANCE:
                    reasons.append(
                        f"Brier regression: {new_brier:.4f} > {prev_brier:.4f} + {BRIER_RISE_TOLERANCE} "
                        f"(rise={new_brier - prev_brier:.4f})"
                    )
        else:
            log.info("No previous metrics found — skipping regression gate (first retrain)")

        if reasons:
            return "; ".join(reasons)
        return None

    def _save_models(self, gbm_model, rf_model, feature_names: list[str]) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        with open(self._models_dir / "gbm.pkl", "wb") as f:
            pickle.dump({"model": gbm_model, "feature_names": feature_names}, f)
        with open(self._models_dir / "rf.pkl", "wb") as f:
            pickle.dump({"model": rf_model,  "feature_names": feature_names}, f)
        log.info("Models saved → %s", self._models_dir)

    def _save_augmented_splits(
        self,
        X_tr, y_tr, learner_ids_tr,
        X_te, y_te, learner_ids_te,
        feature_names,
    ) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        tr_blob = {"X": X_tr, "y": y_tr, "learner_ids": learner_ids_tr, "feature_names": feature_names}
        te_blob = {"X": X_te, "y": y_te, "learner_ids": learner_ids_te, "feature_names": feature_names}
        with open(self._data_dir / "train.pkl", "wb") as f:
            pickle.dump(tr_blob, f)
        with open(self._data_dir / "test.pkl", "wb") as f:
            pickle.dump(te_blob, f)
        log.info("Augmented splits saved (train=%d  test=%d  features=%d)",
                 len(X_tr), len(X_te), len(feature_names))

    def _save_summary(self, gbm_metrics, rf_metrics, feature_names, version):
        summary = {
            "gbm":           gbm_metrics,
            "rf":            rf_metrics,
            "feature_names": feature_names,
            "model_version": version,
            "n_features":    len(feature_names),
            "latent_features_added": LATENT_FEATURE_NAMES,
        }
        with open(self._models_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _save_manifest(
        self,
        version: str,
        metrics: dict,
        feature_names: list[str],
        train_pkl_path: Path,
    ) -> None:
        """
        Write model_manifest.json — tamper-evident record for artifact governance.

        Fields
        ------
        model_version        : e.g. "gbm-v3"
        training_timestamp   : ISO-8601 UTC
        metrics_snapshot     : dict of GBM test metrics at promotion time
        feature_schema_hash  : SHA-256 of sorted JSON feature list
        data_fingerprint     : SHA-256 of the raw train.pkl bytes
        n_features           : int
        strict_mode          : bool — whether RETRAIN_STRICT_MODE was active
        """
        schema_hash = _feature_schema_hash(feature_names)
        data_fp = _file_sha256(train_pkl_path) if train_pkl_path.exists() else "unavailable"

        manifest = {
            "model_version":       version,
            "training_timestamp":  datetime.now(timezone.utc).isoformat(),
            "metrics_snapshot":    metrics,
            "feature_schema_hash": schema_hash,
            "data_fingerprint":    data_fp,
            "n_features":          len(feature_names),
            "feature_names":       feature_names,
            "strict_mode":         STRICT_MODE,
        }
        manifest_path = self._models_dir / "model_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        log.info(
            "Manifest saved  version=%s  schema_hash=%s  data_fp=%s",
            version, schema_hash[:16], data_fp[:16],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────

def _feature_schema_hash(feature_names: list[str]) -> str:
    """SHA-256 of the JSON-encoded sorted feature name list."""
    canonical = json.dumps(sorted(feature_names), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    """SHA-256 of a file's raw bytes (streaming, 64KB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
