#!/usr/bin/env python3
"""
LSTM Trainer — Temporal Dropout Risk Model
==========================================
Loads data/temporal/snapshots.pkl, builds per-student sequences of shape
(T=6, F=12), trains a 2-layer LSTM with early stopping on val AUC, and saves
the model to models/lstm.pt + models/lstm_config.json.  Run logged to MLflow.

Usage:
    python backend/app/model/lstm_trainer.py
    python backend/app/model/lstm_trainer.py --epochs 80 --hidden 128
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
MLRUNS_DIR   = PROJECT_ROOT / "mlruns"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

WEEK_ORDER  = [2, 4, 6, 8, 10, 12]
N_TIMESTEPS = len(WEEK_ORDER)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
    log.info("GPU: %s  CUDA %s  TF32+benchmark enabled",
             torch.cuda.get_device_name(0), torch.version.cuda)


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

class DropoutLSTM(nn.Module):
    def __init__(self, n_features: int = 12, hidden_dim: int = 64,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers
        self.dropout_p   = dropout

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).squeeze(-1)

    def predict_proba(self, X_sequence: np.ndarray) -> np.ndarray:
        """Sklearn-compatible interface for XAI pipeline. Input: (N, T, F) numpy array."""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X_sequence, dtype=torch.float32).to(DEVICE)
            logits = self.forward(t).cpu().numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - proba, proba])


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def build_sequences(snapshots_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        X — (N_students, T=6, F=12)  float32
        y — (N_students,)             int   (1=dropout)
        feature_names — list[str]
    """
    import pandas as pd

    with open(snapshots_path, "rb") as f:
        raw = pickle.load(f)

    if isinstance(raw, dict):
        df: pd.DataFrame = raw["snapshots"]
        feature_names = raw.get("feature_columns",
                                [c for c in df.columns
                                 if c not in ("learner_id", "snapshot_week", "dropout_risk")])
    else:
        df = raw
        feature_names = [c for c in df.columns
                         if c not in ("learner_id", "snapshot_week", "dropout_risk")]

    sequences, labels = [], []
    for lid, grp in df.sort_values("snapshot_week").groupby("learner_id"):
        week_data = {}
        for _, row in grp.iterrows():
            week_data[int(row["snapshot_week"])] = row[feature_names].values.astype(np.float32)

        seq = np.zeros((N_TIMESTEPS, len(feature_names)), dtype=np.float32)
        for i, w in enumerate(WEEK_ORDER):
            if w in week_data:
                seq[i] = week_data[w]

        sequences.append(seq)
        labels.append(int(grp["dropout_risk"].iloc[-1]))

    X = np.stack(sequences, axis=0)
    y = np.array(labels, dtype=np.int64)
    log.info("Sequences built: X=%s  y=%s  dropout=%.1f%%",
             X.shape, y.shape, 100 * y.mean())
    return X, y, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    snapshots_path = DATA_DIR / "temporal" / "snapshots.pkl"
    if not snapshots_path.exists():
        raise FileNotFoundError(f"{snapshots_path} not found — run temporal_builder.py first")

    X, y, feature_names = build_sequences(snapshots_path)
    n_features = X.shape[2]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32))
    pin = DEVICE.type == "cuda"
    tr_dl  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=pin, persistent_workers=pin)
    val_dl = DataLoader(val_ds, batch_size=1024,
                        num_workers=2, pin_memory=pin, persistent_workers=pin)

    model     = DropoutLSTM(n_features, args.hidden, args.n_layers, args.dropout).to(DEVICE)
    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            log.warning("torch.compile skipped: %s", e)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler(device="cuda", enabled=DEVICE.type == "cuda")

    params = vars(args)
    params["n_features"]   = n_features
    params["n_timesteps"]  = N_TIMESTEPS
    params["device"]       = str(DEVICE)

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment("xai-dropout-risk")

    best_auc       = 0.0
    patience_count = 0
    best_state     = None

    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params(params)
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss = 0.0
            for xb, yb in tr_dl:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_ds)

            model.eval()
            val_logits, val_labels = [], []
            with torch.no_grad(), autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                for xb, yb in val_dl:
                    val_logits.append(model(xb.to(DEVICE, non_blocking=True)).cpu())
                    val_labels.append(yb)
            val_logits = torch.cat(val_logits).numpy()
            val_labels = torch.cat(val_labels).numpy()
            val_proba  = 1.0 / (1.0 + np.exp(-val_logits))
            val_auc    = roc_auc_score(val_labels, val_proba)

            scheduler.step(val_auc)
            mlflow.log_metrics({"train_loss": round(tr_loss, 4),
                                "val_auc": round(val_auc, 4)}, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                log.info("Epoch %3d  train_loss=%.4f  val_auc=%.4f", epoch, tr_loss, val_auc)

            if val_auc > best_auc:
                best_auc    = val_auc
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    log.info("Early stop at epoch %d  best_val_auc=%.4f", epoch, best_auc)
                    break

        elapsed = round(time.time() - t0, 1)
        mlflow.log_metrics({"best_val_auc": best_auc, "training_time_sec": elapsed})
        log.info("Training done  best_val_auc=%.4f  [%.1fs]", best_auc, elapsed)

        model.load_state_dict(best_state)
        mlflow.pytorch.log_model(model, "model", registered_model_name="lstm-dropout-risk")

    model_path = MODELS_DIR / "lstm.pt"
    torch.save(best_state, model_path)

    config = {
        "n_features":   n_features,
        "n_timesteps":  N_TIMESTEPS,
        "hidden_dim":   args.hidden,
        "n_layers":     args.n_layers,
        "dropout":      args.dropout,
        "week_order":   WEEK_ORDER,
        "feature_names": feature_names,
        "best_val_auc": round(best_auc, 4),
    }
    config_path = MODELS_DIR / "lstm_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info("Saved → %s", model_path)
    log.info("Saved → %s", config_path)
    return model, config


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=int,   default=64)
    parser.add_argument("--n-layers",   type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--patience",   type=int,   default=5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
