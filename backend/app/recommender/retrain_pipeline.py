from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from backend.app.model.implicit_aggregator import ImplicitAggregator
from backend.app.model.implicit_aggregator import ImplicitSignals
from backend.app.recommender.train_recommenders import train_recommenders

log = logging.getLogger(__name__)

@dataclass
class RecommendationRetrainResult:
    success: bool
    trigger: str
    n_student_rows: int = 0
    n_instructor_rows: int = 0
    n_implicit_learners: int = 0
    metrics: dict = field(default_factory=dict)
    message: str = ""
    rejection_reason: Optional[str] = None
    training_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "trigger": self.trigger,
            "n_student_rows": self.n_student_rows,
            "n_instructor_rows": self.n_instructor_rows,
            "n_implicit_learners": self.n_implicit_learners,
            "metrics": self.metrics,
            "message": self.message,
            "rejection_reason": self.rejection_reason,
            "training_time_sec": round(self.training_time_sec, 1),
        }


class RecommendationRetrainPipeline:
    """Retrains recommendation rankers enriched with learned implicit engagement signals."""

    def __init__(
        self,
        event_store,
        feedback_store,
        data_dir: Path,
        models_dir: Path,
    ):
        self._events = event_store
        self._feedback = feedback_store
        self._data_dir = Path(data_dir)
        self._models_dir = Path(models_dir)

    def run(self, trigger: str = "manual", seed: int = 42) -> RecommendationRetrainResult:
        t_start = time.time()
        try:
            student_path = self._data_dir / "synthetic" / "recommendations" / "student_recommendation_dataset.csv"
            instructor_path = self._data_dir / "synthetic" / "recommendations" / "instructor_recommendation_dataset.csv"
            if not student_path.exists() or not instructor_path.exists():
                return RecommendationRetrainResult(
                    success=False,
                    trigger=trigger,
                    message="Recommendation datasets not found.",
                    rejection_reason="synthetic recommendation datasets missing",
                )

            student_df = pd.read_csv(student_path)
            instructor_df = pd.read_csv(instructor_path)

            aggregator = ImplicitAggregator(event_store=self._events, feedback_store=self._feedback)
            signals = aggregator.aggregate_all()
            signal_df = self._signals_to_df(signals)

            enriched_student = self._attach_implicit_features(student_df, signal_df)
            enriched_instructor = self._attach_implicit_features(instructor_df, signal_df)

            summary = train_recommenders(
                student_df=enriched_student,
                instructor_df=enriched_instructor,
                seed=seed,
            )

            return RecommendationRetrainResult(
                success=True,
                trigger=trigger,
                n_student_rows=len(enriched_student),
                n_instructor_rows=len(enriched_instructor),
                n_implicit_learners=len(signal_df),
                metrics={
                    "student": summary.get("student", {}),
                    "instructor": summary.get("instructor", {}),
                },
                message="Recommendation retrain successful. Call /mlops/reload-all to activate.",
                training_time_sec=time.time() - t_start,
            )
        except Exception as exc:
            log.exception("Recommendation retrain failed")
            return RecommendationRetrainResult(
                success=False,
                trigger=trigger,
                message="Recommendation retrain failed.",
                rejection_reason=str(exc),
                training_time_sec=time.time() - t_start,
            )

    def _signals_to_df(self, signals: Mapping[str, ImplicitSignals]) -> pd.DataFrame:
        rows = []
        for learner_id, sig in signals.items():
            raw = sig.to_dict()
            raw["learner_id"] = learner_id
            rows.append(raw)

        if not rows:
            return pd.DataFrame(columns=["learner_id", "learned_implicit_available"])

        df = pd.DataFrame(rows)
        drop_cols = [c for c in ["cold_start"] if c in df.columns]
        df = df.drop(columns=drop_cols)

        value_cols = [c for c in df.columns if c != "learner_id"]
        df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Prefix features so model artifacts clearly indicate learned runtime implicit signals.
        renamed = {
            c: f"learned_implicit_{c}"
            for c in value_cols
        }
        df = df.rename(columns=renamed)
        df["learned_implicit_available"] = 1.0
        return df

    def _attach_implicit_features(self, df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
        if "learner_id" not in df.columns:
            out = df.copy()
            out["learned_implicit_available"] = 0.0
            return out

        merged = df.merge(signal_df, on="learner_id", how="left")
        learned_cols = [c for c in merged.columns if c.startswith("learned_implicit_")]
        if learned_cols:
            merged[learned_cols] = merged[learned_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        merged["learned_implicit_available"] = np.where(
            merged["learned_implicit_available"].isna(), 0.0, merged["learned_implicit_available"]
        )
        return merged
