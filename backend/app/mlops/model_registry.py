"""
Model Registry + Promotion
============================
MLflow Model Registry helpers for versioning and stage promotion.

Public API
----------
ModelRegistry(tracking_uri)
    .promote(model_name, version, stage)
    .get_production_version(model_name) -> dict | None
    .get_latest_versions(model_name) -> list[dict]
    .rollback(model_name) -> bool
    .load_production_model(model_name) -> model | None
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MLRUNS_DIR   = PROJECT_ROOT / "mlruns"


class ModelRegistry:
    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = tracking_uri or f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        log.info("ModelRegistry initialised  uri=%s", self.tracking_uri)

    def promote(
        self, model_name: str, version: int, stage: str = "Production"
    ) -> dict:
        """
        Transition a model version to a new stage.
        Stages: None → Staging → Production → Archived
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        log.info("Promoted %s v%d → %s", model_name, version, stage)
        return {"model_name": model_name, "version": version, "stage": stage}

    def get_production_version(self, model_name: str) -> dict | None:
        """Get the current Production-stage version."""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                v = versions[0]
                return {
                    "model_name": model_name,
                    "version":    int(v.version),
                    "stage":      v.current_stage,
                    "run_id":     v.run_id,
                    "source":     v.source,
                    "created":    str(v.creation_timestamp),
                }
        except Exception as e:
            log.warning("Failed to get production version for %s: %s", model_name, e)
        return None

    def get_latest_versions(self, model_name: str) -> list[dict]:
        """List all versions across all stages."""
        try:
            results = self.client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": int(v.version),
                    "stage":   v.current_stage,
                    "run_id":  v.run_id,
                    "status":  v.status,
                    "created": str(v.creation_timestamp),
                }
                for v in results
            ]
        except Exception as e:
            log.warning("Failed to list versions for %s: %s", model_name, e)
            return []

    def rollback(self, model_name: str) -> bool:
        """
        Rollback to the previous version: archive current Production,
        promote the most recent Archived or Staging version.
        """
        try:
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            sorted_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)

            prod_versions = [v for v in sorted_versions if v.current_stage == "Production"]
            other_versions = [v for v in sorted_versions
                             if v.current_stage in ("Archived", "Staging", "None")]

            if not prod_versions or not other_versions:
                log.warning("Rollback not possible for %s — insufficient versions", model_name)
                return False

            # Archive current production
            for pv in prod_versions:
                self.client.transition_model_version_stage(
                    name=model_name, version=pv.version, stage="Archived"
                )

            # Promote most recent non-production version
            target = other_versions[0]
            self.client.transition_model_version_stage(
                name=model_name, version=target.version, stage="Production"
            )
            log.info("Rolled back %s: archived v%s, promoted v%s",
                     model_name, prod_versions[0].version, target.version)
            return True

        except Exception as e:
            log.error("Rollback failed for %s: %s", model_name, e)
            return False

    def load_production_model(self, model_name: str):
        """Load the Production-stage model via MLflow."""
        prod = self.get_production_version(model_name)
        if prod is None:
            return None
        try:
            model_uri = f"models:/{model_name}/Production"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            log.warning("Failed to load production model %s: %s", model_name, e)
            return None
