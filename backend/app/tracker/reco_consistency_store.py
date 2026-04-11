"""
Recommendation Explanation Store (Feature 6 / Feature 13)
===========================================================
Persists per-learner recommendation explanation records to the database.
Mirrors consistency_store.py exactly but uses `score` (ranker output)
instead of `risk_score` (dropout probability).

Used by:
  - /recommend/student/explain  → save top-3 SHAP features + score
  - ExplanationDriftDetector    → check_learner_drift on recommendation history

Public API
----------
RecommendationExplanationStore(db_url="")
    .save(learner_id, score, shap_values, top3_features, ...) -> int
    .get_history(learner_id, n=10)      -> list[dict]
    .get_top3_timeline(learner_id)      -> list[dict]
    .get_latest(learner_id)             -> dict | None
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.db_config import make_engine

log = logging.getLogger(__name__)

Base = declarative_base()


class RecoExplanationRecord(Base):
    __tablename__ = "reco_explanation_records"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    learner_id    = Column(String(64), nullable=False, index=True)
    timestamp     = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    score         = Column(Float, nullable=False)          # ranker score (not dropout prob)
    trust_score   = Column(Float, nullable=True)
    model_used    = Column(String(32), default="student_ranker")
    top3_features = Column(Text, nullable=False)           # JSON list
    shap_values   = Column(Text, nullable=False)           # JSON dict
    anchor_rule   = Column(Text, nullable=True)
    extra         = Column(Text, nullable=True)            # JSON dict for extensibility

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "learner_id":    self.learner_id,
            "timestamp":     self.timestamp.isoformat() if self.timestamp else None,
            "score":         self.score,
            "trust_score":   self.trust_score,
            "model_used":    self.model_used,
            "top3_features": json.loads(self.top3_features) if self.top3_features else [],
            "shap_values":   json.loads(self.shap_values) if self.shap_values else {},
            "anchor_rule":   self.anchor_rule,
            "extra":         json.loads(self.extra) if self.extra else {},
        }


class RecommendationExplanationStore:
    def __init__(self, db_url: str = ""):
        if db_url:
            self.engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
                echo=False,
            )
        else:
            self.engine = make_engine("reco_explanations")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        log.info("RecommendationExplanationStore initialised")

    def save(
        self,
        learner_id: str,
        score: float,
        shap_values: dict,
        top3_features: list,
        trust_score: float | None = None,
        model_used: str = "student_ranker",
        anchor_rule: str | None = None,
        extra: dict | None = None,
    ) -> int:
        with self.Session() as session:
            record = RecoExplanationRecord(
                learner_id    = learner_id,
                score         = score,
                trust_score   = trust_score,
                model_used    = model_used,
                top3_features = json.dumps(top3_features),
                shap_values   = json.dumps(shap_values),
                anchor_rule   = anchor_rule,
                extra         = json.dumps(extra) if extra else None,
            )
            session.add(record)
            session.commit()
            record_id = record.id
        return record_id

    def get_history(self, learner_id: str, n: int = 10) -> list[dict]:
        with self.Session() as session:
            records = (
                session.query(RecoExplanationRecord)
                .filter(RecoExplanationRecord.learner_id == learner_id)
                .order_by(desc(RecoExplanationRecord.timestamp))
                .limit(n)
                .all()
            )
            return [r.to_dict() for r in records]

    def get_top3_timeline(self, learner_id: str) -> list[dict]:
        history = self.get_history(learner_id, n=50)
        return [
            {
                "timestamp":     h["timestamp"],
                "score":         h["score"],
                "top3_features": h["top3_features"],
            }
            for h in reversed(history)
        ]

    def get_latest(self, learner_id: str) -> dict | None:
        history = self.get_history(learner_id, n=1)
        return history[0] if history else None
