"""
Prediction Logger
==================
Logs every /predict call to a rolling table for monitoring and
drift analysis.  Auto-prunes to keep the most recent N rows.
Uses PostgreSQL in production (via DATABASE_URL) or SQLite locally.

Public API
----------
PredictionLogger(db_url, max_rows=10000)
    .log(features, risk_score, model_version, model_used)
    .get_recent(n=100) -> list[dict]
    .get_feature_matrix(n=500) -> pd.DataFrame  (for Evidently)
    .count() -> int
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    desc,
    func,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.app.db_config import make_engine

log = logging.getLogger(__name__)

Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "prediction_log"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    timestamp     = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    features_json = Column(Text, nullable=False)
    risk_score    = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=True)
    model_used    = Column(String(32), default="gbm")


class PredictionLogger:
    def __init__(
        self,
        db_url: str = "",
        max_rows: int = 10_000,
    ):
        if db_url:
            self.engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
                echo=False,
            )
        else:
            self.engine = make_engine("predictions")
        Base.metadata.create_all(self.engine)
        self.Session  = sessionmaker(bind=self.engine)
        self.max_rows = max_rows
        log.info("PredictionLogger initialised  max_rows=%d", max_rows)

    def log(
        self,
        features: dict,
        risk_score: float,
        model_version: str = "unknown",
        model_used: str = "gbm",
    ):
        with self.Session() as session:
            record = PredictionRecord(
                features_json=json.dumps(features),
                risk_score=risk_score,
                model_version=model_version,
                model_used=model_used,
            )
            session.add(record)
            session.commit()

        self._prune()

    def _prune(self):
        """Delete oldest rows if count exceeds max_rows."""
        with self.Session() as session:
            count = session.query(func.count(PredictionRecord.id)).scalar()
            if count > self.max_rows:
                cutoff_id = (
                    session.query(PredictionRecord.id)
                    .order_by(desc(PredictionRecord.id))
                    .offset(self.max_rows)
                    .limit(1)
                    .scalar()
                )
                if cutoff_id:
                    session.query(PredictionRecord).filter(
                        PredictionRecord.id <= cutoff_id
                    ).delete()
                    session.commit()

    def get_recent(self, n: int = 100) -> list[dict]:
        with self.Session() as session:
            records = (
                session.query(PredictionRecord)
                .order_by(desc(PredictionRecord.timestamp))
                .limit(n)
                .all()
            )
            return [
                {
                    "id":            r.id,
                    "timestamp":     r.timestamp.isoformat() if r.timestamp else None,
                    "features":      json.loads(r.features_json),
                    "risk_score":    r.risk_score,
                    "model_version": r.model_version,
                    "model_used":    r.model_used,
                }
                for r in records
            ]

    def get_feature_matrix(self, n: int = 500) -> pd.DataFrame:
        """Return recent predictions as a DataFrame for drift analysis."""
        recent = self.get_recent(n)
        if not recent:
            return pd.DataFrame()
        rows = [r["features"] for r in recent]
        return pd.DataFrame(rows)

    def count(self) -> int:
        with self.Session() as session:
            return session.query(func.count(PredictionRecord.id)).scalar()
