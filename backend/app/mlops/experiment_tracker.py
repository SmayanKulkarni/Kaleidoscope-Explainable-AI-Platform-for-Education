"""
MLflow Experiment Tracker
==========================
Centralised wrapper for MLflow experiment tracking.
All model training runs log through this interface.

Public API
----------
ExperimentTracker(experiment_name, tracking_uri)
    .log_training_run(model, model_name, metrics, params, artifacts)
    .log_shap_summary(shap_global, feature_names, run_id)
    .get_best_run(metric) -> dict
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np

from backend.app.mlops.mlflow_config import configure_mlflow, TRACKING_URI

log = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str = "xai-dropout-risk",
        tracking_uri: str | None = None,
    ):
        self.tracking_uri = tracking_uri or TRACKING_URI
        configure_mlflow(experiment=experiment_name)
        self.experiment_name = experiment_name
        log.info("ExperimentTracker initialised  experiment=%s  uri=%s",
                 experiment_name, self.tracking_uri)

    def log_training_run(
        self,
        model,
        model_name: str,
        metrics: dict,
        params: dict,
        artifacts: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Log a full training run. Returns the MLflow run_id.
        """
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if tags:
                mlflow.set_tags(tags)

            # Log model
            if model_name in ("lstm",):
                mlflow.pytorch.log_model(model, "model",
                                         registered_model_name=f"{model_name}-dropout-risk")
            else:
                mlflow.sklearn.log_model(model, "model",
                                         registered_model_name=f"{model_name}-dropout-risk")

            # Log additional artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

            return run.info.run_id

    def log_shap_summary(
        self,
        shap_global: dict[str, float],
        feature_names: list[str],
        run_id: str | None = None,
    ):
        """
        Log global SHAP importance as a JSON artifact.
        """
        sorted_feats = sorted(shap_global.items(), key=lambda x: abs(x[1]), reverse=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {"global_shap_importance": [{"feature": k, "mean_abs_shap": v}
                                            for k, v in sorted_feats]},
                f, indent=2,
            )
            f.flush()

            if run_id:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifact(f.name, "shap")
            else:
                mlflow.log_artifact(f.name, "shap")

    def get_best_run(self, metric: str = "test_auc_roc") -> dict | None:
        """
        Find the run with the best value of `metric` in the current experiment.
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if runs.empty:
            return None

        row = runs.iloc[0]
        return {
            "run_id":     row["run_id"],
            "run_name":   row.get("tags.mlflow.runName", ""),
            metric:       row.get(f"metrics.{metric}", None),
            "start_time": str(row.get("start_time", "")),
        }
