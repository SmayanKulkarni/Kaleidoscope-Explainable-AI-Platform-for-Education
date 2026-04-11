#!/usr/bin/env python3
"""
GBM & RF Trainer — Static Dropout Risk Model
=============================================
Loads data/train.pkl, trains GradientBoostingClassifier (primary) and
RandomForestClassifier (secondary), calibrates both, evaluates on test set,
and saves artifacts to models/.  All runs logged to MLflow.

Usage:
    python backend/app/model/trainer.py
    python backend/app/model/trainer.py --n-estimators 200 --lr 0.05
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from backend.app.mlops.mlflow_config import configure_mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
MLRUNS_DIR   = PROJECT_ROOT / "mlruns"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_split(name: str) -> dict:
    path = DATA_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data_loader.py first")
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, X, y, prefix: str = "") -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    metrics = {
        f"{prefix}auc_roc":           round(roc_auc_score(y, proba), 4),
        f"{prefix}avg_precision":     round(average_precision_score(y, proba), 4),
        f"{prefix}f1":                round(f1_score(y, pred), 4),
        f"{prefix}brier_score":       round(brier_score_loss(y, proba), 4),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────

def train_gbm(X_tr, y_tr, X_te, y_te, feature_names, args):
    params = {
        "n_estimators":     args.n_estimators,
        "learning_rate":    args.lr,
        "max_depth":        args.max_depth,
        "subsample":        0.8,
        "min_samples_leaf": 20,
        "random_state":     42,
    }
    log.info("Training GradientBoostingClassifier  %s", params)
    t0 = time.time()

    configure_mlflow(alias="train")

    with mlflow.start_run(run_name="gbm"):
        base = GradientBoostingClassifier(**params)
        base.fit(X_tr, y_tr)

        model = CalibratedClassifierCV(base, method="isotonic", cv=5)
        model.fit(X_tr, y_tr)

        elapsed = round(time.time() - t0, 1)
        tr_metrics = evaluate(model, X_tr, y_tr, "train_")
        te_metrics  = evaluate(model, X_te, y_te, "test_")
        all_metrics = {**tr_metrics, **te_metrics, "training_time_sec": elapsed}

        mlflow.log_params(params)
        mlflow.log_metrics(all_metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name="gbm-dropout-risk")

        log.info("GBM  test AUC=%.4f  F1=%.4f  Brier=%.4f  [%.1fs]",
                 te_metrics["test_auc_roc"], te_metrics["test_f1"],
                 te_metrics["test_brier_score"], elapsed)
        log.info("\n%s", classification_report(y_te, (model.predict_proba(X_te)[:, 1] >= 0.5).astype(int)))

    out = MODELS_DIR / "gbm.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    log.info("Saved → %s", out)
    return model, all_metrics


def train_rf(X_tr, y_tr, X_te, y_te, feature_names, args):
    params = {
        "n_estimators": args.n_estimators,
        "max_depth":    None,
        "min_samples_leaf": 10,
        "n_jobs":       -1,
        "random_state": 42,
    }
    log.info("Training RandomForestClassifier  %s", params)
    t0 = time.time()

    configure_mlflow(alias="train")

    with mlflow.start_run(run_name="rf"):
        base = RandomForestClassifier(**params)
        base.fit(X_tr, y_tr)

        model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        model.fit(X_tr, y_tr)

        elapsed = round(time.time() - t0, 1)
        te_metrics = evaluate(model, X_te, y_te, "test_")
        all_metrics = {**te_metrics, "training_time_sec": elapsed}

        mlflow.log_params(params)
        mlflow.log_metrics(all_metrics)
        mlflow.sklearn.log_model(model, "model", registered_model_name="rf-dropout-risk")

        log.info("RF   test AUC=%.4f  F1=%.4f  Brier=%.4f  [%.1fs]",
                 te_metrics["test_auc_roc"], te_metrics["test_f1"],
                 te_metrics["test_brier_score"], elapsed)

    out = MODELS_DIR / "rf.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    log.info("Saved → %s", out)
    return model, all_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--lr",           type=float, default=0.05)
    parser.add_argument("--max-depth",    type=int, default=4)
    args = parser.parse_args()

    train_data = load_split("train")
    test_data  = load_split("test")

    X_tr = train_data["X"]
    y_tr = train_data["y"]
    X_te = test_data["X"]
    y_te = test_data["y"]
    feature_names = train_data["feature_names"]

    log.info("Train=%d  Test=%d  Features=%d  Dropout-rate=%.1f%%",
             len(X_tr), len(X_te), len(feature_names), 100 * y_tr.mean())

    gbm_model, gbm_metrics = train_gbm(X_tr, y_tr, X_te, y_te, feature_names, args)
    rf_model,  rf_metrics  = train_rf (X_tr, y_tr, X_te, y_te, feature_names, args)

    summary = {
        "gbm": gbm_metrics,
        "rf":  rf_metrics,
        "feature_names": feature_names,
    }
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary → %s", summary_path)

    log.info("Done. Models saved to %s/", MODELS_DIR)


if __name__ == "__main__":
    main()
