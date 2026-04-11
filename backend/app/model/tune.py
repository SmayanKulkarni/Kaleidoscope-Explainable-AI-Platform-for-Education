#!/usr/bin/env python3
"""
Hyperparameter Tuner — GBM, RF, LSTM
======================================
Light tuning via Optuna (TPE sampler).  Each study runs N_TRIALS trials,
optimises val AUC-ROC, and saves the best model to models/ (overwriting only
if the tuned model beats the baseline).

Usage:
    python backend/app/model/tune.py                  # all models, 20 trials each
    python backend/app/model/tune.py --models gbm --n-trials 30
    python backend/app/model/tune.py --models lstm --n-trials 15
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import mlflow
from backend.app.mlops.mlflow_config import configure_mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from backend.app.model.lstm_trainer import (
    DropoutLSTM,
    build_sequences,
    DEVICE,
    MODELS_DIR,
    MLRUNS_DIR,
    N_TIMESTEPS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data"


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_static():
    with open(DATA_DIR / "train.pkl", "rb") as f:
        tr = pickle.load(f)
    with open(DATA_DIR / "test.pkl", "rb") as f:
        te = pickle.load(f)
    return tr["X"], tr["y"], te["X"], te["y"], tr["feature_names"]


def _load_temporal():
    X, y, feat = build_sequences(DATA_DIR / "temporal" / "snapshots.pkl")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_tr, y_tr, X_val, y_val, feat


# ──────────────────────────────────────────────────────────────────────────────
# GBM study
# ──────────────────────────────────────────────────────────────────────────────

def _tune_gbm(n_trials: int) -> dict:
    X_tr, y_tr, X_te, y_te, feature_names = _load_static()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     100, 400, step=50),
            "learning_rate":    trial.suggest_float("learning_rate",  0.01, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth",        3, 6),
            "subsample":        trial.suggest_float("subsample",      0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8]),
            "random_state": 42,
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_tr, y_tr):
            clf = GradientBoostingClassifier(**params)
            clf.fit(X_tr[tr_idx], y_tr[tr_idx])
            proba = clf.predict_proba(X_tr[val_idx])[:, 1]
            scores.append(roc_auc_score(y_tr[val_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log.info("GBM best params: %s  val_auc=%.4f", best, study.best_value)

    best["random_state"] = 42
    t0 = time.time()
    final = CalibratedClassifierCV(
        GradientBoostingClassifier(**best), method="isotonic", cv=5
    )
    final.fit(X_tr, y_tr)
    test_auc = roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])
    elapsed = round(time.time() - t0, 1)
    log.info("GBM tuned  test_auc=%.4f  [%.1fs]", test_auc, elapsed)

    return {"model": final, "feature_names": feature_names,
            "best_params": best, "cv_auc": study.best_value,
            "test_auc": test_auc, "training_time_sec": elapsed}


# ──────────────────────────────────────────────────────────────────────────────
# RF study
# ──────────────────────────────────────────────────────────────────────────────

def _tune_rf(n_trials: int) -> dict:
    X_tr, y_tr, X_te, y_te, feature_names = _load_static()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     100, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth",        5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 40),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 20),
            "n_jobs": -1, "random_state": 42,
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_tr, y_tr):
            clf = RandomForestClassifier(**params)
            clf.fit(X_tr[tr_idx], y_tr[tr_idx])
            proba = clf.predict_proba(X_tr[val_idx])[:, 1]
            scores.append(roc_auc_score(y_tr[val_idx], proba))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log.info("RF  best params: %s  val_auc=%.4f", best, study.best_value)

    best["n_jobs"] = -1
    best["random_state"] = 42
    t0 = time.time()
    final = CalibratedClassifierCV(
        RandomForestClassifier(**best), method="sigmoid", cv=5
    )
    final.fit(X_tr, y_tr)
    test_auc = roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])
    elapsed = round(time.time() - t0, 1)
    log.info("RF  tuned  test_auc=%.4f  [%.1fs]", test_auc, elapsed)

    return {"model": final, "feature_names": feature_names,
            "best_params": best, "cv_auc": study.best_value,
            "test_auc": test_auc, "training_time_sec": elapsed}


# ──────────────────────────────────────────────────────────────────────────────
# LSTM study
# ──────────────────────────────────────────────────────────────────────────────

def _make_loaders(X_tr, y_tr, X_val, y_val, batch_size: int):
    pin = DEVICE.type == "cuda"
    tr_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=pin, persistent_workers=pin,
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32)),
        batch_size=1024,
        num_workers=2, pin_memory=pin, persistent_workers=pin,
    )
    return tr_dl, val_dl


def _train_lstm_trial(X_tr, y_tr, X_val, y_val, n_features, params, max_epochs=30, patience=4):
    model = DropoutLSTM(
        n_features = n_features,
        hidden_dim = params["hidden_dim"],
        n_layers   = params["n_layers"],
        dropout    = params["dropout"],
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler(device="cuda", enabled=DEVICE.type == "cuda")
    tr_dl, val_dl = _make_loaders(X_tr, y_tr, X_val, y_val, params["batch_size"])

    best_auc, patience_cnt = 0.0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        logits, labels = [], []
        with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            for xb, yb in val_dl:
                logits.append(model(xb.to(DEVICE, non_blocking=True)).cpu())
                labels.append(yb)
        proba = torch.sigmoid(torch.cat(logits)).numpy()
        auc   = roc_auc_score(torch.cat(labels).numpy(), proba)
        scheduler.step(auc)

        if auc > best_auc:
            best_auc, patience_cnt = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_auc


def _tune_lstm(n_trials: int) -> dict:
    X_tr, y_tr, X_val, y_val, feature_names = _load_temporal()
    n_features = X_tr.shape[2]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "n_layers":   trial.suggest_int("n_layers", 1, 3),
            "dropout":    trial.suggest_float("dropout", 0.1, 0.5),
            "lr":         trial.suggest_float("lr", 5e-4, 5e-3, log=True),
            "wd":         trial.suggest_float("wd", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        }
        _, auc = _train_lstm_trial(X_tr, y_tr, X_val, y_val, n_features, params)
        return auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    log.info("LSTM best params: %s  val_auc=%.4f", best, study.best_value)

    log.info("Retraining LSTM with best params (50 epochs) …")
    final_model, best_auc = _train_lstm_trial(
        X_tr, y_tr, X_val, y_val, n_features, best,
        max_epochs=50, patience=5,
    )
    log.info("LSTM tuned  val_auc=%.4f", best_auc)

    return {"model": final_model, "feature_names": feature_names,
            "best_params": best, "best_val_auc": best_auc, "n_features": n_features}


# ──────────────────────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_if_better(name: str, result: dict, baseline_auc_key: str = "test_auc"):
    baseline_path = MODELS_DIR / "training_summary.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            summary = json.load(f)
        baseline = summary.get(name, {}).get("test_auc_roc", 0.0)
        tuned    = result.get(baseline_auc_key, 0.0) or result.get("best_val_auc", 0.0)
        if tuned <= baseline:
            log.info("%s tuned AUC (%.4f) did not beat baseline (%.4f) — keeping original",
                     name.upper(), tuned, baseline)
            return False

    if name in ("gbm", "rf"):
        out = MODELS_DIR / f"{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"model": result["model"],
                         "feature_names": result["feature_names"]}, f)
        log.info("Saved improved %s → %s", name.upper(), out)
    elif name == "lstm":
        torch.save(result["model"].state_dict(), MODELS_DIR / "lstm.pt")
        with open(MODELS_DIR / "lstm_config.json") as f:
            cfg = json.load(f)
        cfg.update({
            "hidden_dim": result["best_params"]["hidden_dim"],
            "n_layers":   result["best_params"]["n_layers"],
            "dropout":    result["best_params"]["dropout"],
            "best_val_auc": result["best_val_auc"],
        })
        with open(MODELS_DIR / "lstm_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        log.info("Saved improved LSTM → models/lstm.pt + lstm_config.json")

    return True


def _log_to_mlflow(name: str, result: dict):
    configure_mlflow(alias="tune")
    with mlflow.start_run(run_name=f"{name}-tuned"):
        mlflow.log_params(result.get("best_params", {}))
        for k, v in result.items():
            if k not in ("model", "feature_names", "best_params") and isinstance(v, (int, float)):
                mlflow.log_metric(k, v)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",   nargs="+", default=["gbm", "rf", "lstm"],
                        choices=["gbm", "rf", "lstm"])
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Optuna trials per model")
    args = parser.parse_args()

    results_summary = {}

    if "gbm" in args.models:
        log.info("─── Tuning GBM (%d trials) ───", args.n_trials)
        r = _tune_gbm(args.n_trials)
        _log_to_mlflow("gbm", r)
        improved = _save_if_better("gbm", r, "test_auc")
        results_summary["gbm"] = {
            "best_params": r["best_params"],
            "cv_auc":      round(r["cv_auc"],  4),
            "test_auc":    round(r["test_auc"], 4),
            "improved":    improved,
        }

    if "rf" in args.models:
        log.info("─── Tuning RF  (%d trials) ───", args.n_trials)
        r = _tune_rf(args.n_trials)
        _log_to_mlflow("rf", r)
        improved = _save_if_better("rf", r, "test_auc")
        results_summary["rf"] = {
            "best_params": r["best_params"],
            "cv_auc":      round(r["cv_auc"],  4),
            "test_auc":    round(r["test_auc"], 4),
            "improved":    improved,
        }

    if "lstm" in args.models:
        log.info("─── Tuning LSTM (%d trials) ───", args.n_trials)
        r = _tune_lstm(args.n_trials)
        _log_to_mlflow("lstm", r)
        improved = _save_if_better("lstm", r)
        results_summary["lstm"] = {
            "best_params": r["best_params"],
            "best_val_auc": round(r["best_val_auc"], 4),
            "improved":    improved,
        }

    out = MODELS_DIR / "tuning_summary.json"
    with open(out, "w") as f:
        json.dump(results_summary, f, indent=2)
    log.info("Tuning complete. Summary → %s", out)
    for name, res in results_summary.items():
        auc_key = "test_auc" if "test_auc" in res else "best_val_auc"
        log.info("  %-6s  AUC=%.4f  improved=%s  params=%s",
                 name.upper(), res[auc_key], res["improved"], res["best_params"])


if __name__ == "__main__":
    main()
