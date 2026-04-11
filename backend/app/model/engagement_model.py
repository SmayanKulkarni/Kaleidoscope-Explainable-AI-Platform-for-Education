"""
Engagement Autoencoder (Step 3)
================================
Denoising autoencoder that compresses 20 implicit+explicit engagement signals
into 3 latent dimensions (engagement_latent_1/2/3) per learner.

These latent scores are appended to the 12 OULAD features before GBM retraining.
Cold-start learners (no events) always produce latent = [0.0, 0.0, 0.0].

Architecture
------------
Input:   20 signals (standardised)
Encoder: Linear(20→16)→ReLU→Dropout(0.2)→Linear(16→8)→ReLU→Linear(8→3)
Latent:  3 dims
Decoder: Linear(3→8)→ReLU→Linear(8→16)→ReLU→Linear(16→20)
Loss:    MSE(reconstructed, original)

Public API
----------
EngagementModel()
    .fit(X, epochs=50, lr=1e-3)          -> training_losses
    .encode(X) -> np.ndarray             -> (n, 3) latent matrix
    .encode_learner(vector) -> list[float]
    .save(path) / .load(path)
    .is_fitted -> bool
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.app.model.implicit_aggregator import (
    ALL_ENGAGEMENT_FEATURE_NAMES,
    LATENT_FEATURE_NAMES,
    N_TOTAL,
)

log = logging.getLogger(__name__)

LATENT_DIM   = len(LATENT_FEATURE_NAMES)   # 3
N_INPUT      = N_TOTAL                      # 20


# ──────────────────────────────────────────────────────────────────────────────
# Neural net definition
# ──────────────────────────────────────────────────────────────────────────────

class _Autoencoder(nn.Module):
    def __init__(self, n_input: int = N_INPUT, latent_dim: int = LATENT_DIM, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, n_input),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ──────────────────────────────────────────────────────────────────────────────
# EngagementModel wrapper
# ──────────────────────────────────────────────────────────────────────────────

class EngagementModel:
    """
    Wraps the denoising autoencoder with sklearn-style fit/encode API.
    Handles standardization internally so callers pass raw signals.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, dropout: float = 0.2, device: Optional[str] = None):
        self.latent_dim  = latent_dim
        self.dropout     = dropout
        self._device     = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._net: Optional[_Autoencoder] = None
        self._mean: Optional[np.ndarray]  = None
        self._std:  Optional[np.ndarray]  = None
        self.is_fitted   = False
        self.feature_names = ALL_ENGAGEMENT_FEATURE_NAMES

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        noise_factor: float = 0.05,
        min_samples: int = 5,
    ) -> list[float]:
        """
        Train the autoencoder on (n_learners, 20) engagement matrix.

        Parameters
        ----------
        X            : raw engagement signals, shape (n, 20)
        epochs       : training epochs
        batch_size   : mini-batch size
        lr           : learning rate
        noise_factor : Gaussian noise std for denoising (fraction of signal)
        min_samples  : if fewer samples than this, skip training

        Returns
        -------
        list of per-epoch MSE losses
        """
        if len(X) < min_samples:
            log.warning(
                "EngagementModel.fit: only %d samples (need ≥%d). "
                "Initialising with identity-style weights. Latent = first 3 dims.",
                len(X), min_samples,
            )
            self._mean = np.zeros(N_INPUT, dtype=np.float32)
            self._std  = np.ones(N_INPUT,  dtype=np.float32)
            self._net  = _Autoencoder(n_input=N_INPUT, latent_dim=self.latent_dim, dropout=0.0)
            self._net.to(self._device)
            self.is_fitted = True
            return []

        # Standardise
        self._mean = X.mean(axis=0).astype(np.float32)
        self._std  = X.std(axis=0).astype(np.float32)
        self._std[self._std == 0] = 1.0   # avoid div-by-zero for constant features

        X_norm = ((X - self._mean) / self._std).astype(np.float32)

        self._net = _Autoencoder(n_input=N_INPUT, latent_dim=self.latent_dim, dropout=self.dropout)
        self._net.to(self._device)

        dataset    = TensorDataset(torch.from_numpy(X_norm))
        loader     = DataLoader(dataset, batch_size=min(batch_size, len(X)), shuffle=True)
        optimiser  = torch.optim.Adam(self._net.parameters(), lr=lr, weight_decay=1e-5)
        criterion  = nn.MSELoss()

        losses: list[float] = []
        self._net.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                # Denoising: add Gaussian noise to input
                noisy = batch + noise_factor * torch.randn_like(batch)
                reconstructed = self._net(noisy)
                loss = criterion(reconstructed, batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch)
            avg_loss = epoch_loss / len(X)
            losses.append(round(avg_loss, 6))
            if (epoch + 1) % 10 == 0:
                log.info("EngagementModel  epoch=%d/%d  mse=%.5f", epoch + 1, epochs, avg_loss)

        self._net.eval()
        self.is_fitted = True
        log.info(
            "EngagementModel.fit complete  samples=%d  final_mse=%.5f",
            len(X), losses[-1] if losses else 0.0,
        )
        return losses

    # ── Encode ─────────────────────────────────────────────────────────────────

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode (n, 20) → (n, 3) latent matrix.
        Cold-start rows (all-zero input) remain [0, 0, 0].
        """
        if not self.is_fitted or self._net is None:
            raise RuntimeError("EngagementModel not fitted. Call .fit() first.")

        X_arr = np.array(X, dtype=np.float32)
        # Find cold-start rows (all zeros)
        cold_mask = (X_arr == 0).all(axis=1)

        # Standardise
        X_norm = ((X_arr - self._mean) / self._std).astype(np.float32)

        self._net.eval()
        with torch.no_grad():
            t = torch.from_numpy(X_norm).to(self._device)
            latent = self._net.encode(t).cpu().numpy()

        # Zero out cold-start learners
        latent[cold_mask] = 0.0
        return latent.astype(np.float32)

    def encode_learner(self, vector: list[float]) -> list[float]:
        """
        Encode a single learner's 20-dim signal vector to 3 latent scores.
        Returns [0.0, 0.0, 0.0] for cold-start learners.
        """
        X = np.array([vector], dtype=np.float32)
        return self.encode(X)[0].tolist()

    # ── Persist ────────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save model weights + normalisation stats to path (directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._net is not None:
            torch.save(self._net.state_dict(), path / "engagement_net.pt")

        meta = {
            "latent_dim":    self.latent_dim,
            "dropout":       self.dropout,
            "n_input":       N_INPUT,
            "feature_names": self.feature_names,
            "is_fitted":     self.is_fitted,
        }
        with open(path / "engagement_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self._mean is not None:
            np.save(path / "engagement_mean.npy", self._mean)
            np.save(path / "engagement_std.npy",  self._std)

        log.info("EngagementModel saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "EngagementModel":
        """Load from a previously saved directory."""
        path = Path(path)
        with open(path / "engagement_meta.json") as f:
            meta = json.load(f)

        obj = cls(latent_dim=meta["latent_dim"], dropout=meta["dropout"])

        net_path = path / "engagement_net.pt"
        if net_path.exists():
            obj._net = _Autoencoder(
                n_input=meta["n_input"],
                latent_dim=meta["latent_dim"],
                dropout=meta["dropout"],
            )
            sd = torch.load(net_path, map_location=obj._device, weights_only=True)
            obj._net.load_state_dict(sd)
            obj._net.eval()
            obj._net.to(obj._device)

        mean_path = path / "engagement_mean.npy"
        if mean_path.exists():
            obj._mean = np.load(mean_path)
            obj._std  = np.load(path / "engagement_std.npy")

        obj.is_fitted = meta.get("is_fitted", False)
        log.info("EngagementModel loaded ← %s", path)
        return obj
