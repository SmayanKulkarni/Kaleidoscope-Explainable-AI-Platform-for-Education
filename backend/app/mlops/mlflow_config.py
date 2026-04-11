"""
Centralized MLflow Configuration
==================================
Single source of truth for MLflow tracking URI and experiment names.
All training, tuning, and retrain flows must call configure_mlflow()
instead of calling mlflow.set_tracking_uri() / mlflow.set_experiment() directly.

Environment variables
---------------------
MLFLOW_TRACKING_URI : Override tracking URI (e.g. "http://mlflow-server:5000"
                      for a remote server, or a local sqlite path).
                      Default: sqlite:///<project_root>/mlruns/mlflow.db

Usage
-----
    from backend.app.mlops.mlflow_config import configure_mlflow, TRACKING_URI

    configure_mlflow(experiment="xai-dropout-risk")
    with mlflow.start_run(...):
        ...
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_URI  = f"sqlite:///{_PROJECT_ROOT / 'mlruns' / 'mlflow.db'}"

TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_URI)

_EXPERIMENT_MAP: dict[str, str] = {
    "train":   "xai-dropout-risk",
    "tune":    "xai-dropout-risk",
    "retrain": "xai-dropout-risk-retrain",
}


def configure_mlflow(
    experiment: str | None = None,
    alias: str | None = None,
) -> str:
    """
    Set MLflow tracking URI and experiment from the canonical config.

    Parameters
    ----------
    experiment : str, optional
        Full experiment name (e.g. "xai-dropout-risk-retrain").
        If omitted and alias given, resolved via _EXPERIMENT_MAP.
    alias : str, optional
        Short alias: "train", "tune", or "retrain".

    Returns
    -------
    str : The experiment name that was set.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    if experiment is None and alias is not None:
        experiment = _EXPERIMENT_MAP.get(alias, "xai-dropout-risk")
    elif experiment is None:
        experiment = "xai-dropout-risk"

    mlflow.set_experiment(experiment)
    log.debug("MLflow configured  uri=%s  experiment=%s", TRACKING_URI, experiment)
    return experiment
