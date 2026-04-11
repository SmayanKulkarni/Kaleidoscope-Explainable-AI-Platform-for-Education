"""
Feedback Store — Human-in-the-Loop (Task 5.3)
==============================================
Persists user ratings, recommendation follow-rate, and feature corrections
to SQLite. Exposes aggregate stats used by GET /feedback/stats.

Tables
------
feedback_records   — per-explanation rating + follow toggle + correction
feedback_stats     — materialised aggregates refreshed on each write (fast reads)

Public API
----------
FeedbackStore(db_url)
    .record(feedback)               -> FeedbackRecord id
    .get_stats()                    -> FeedbackStats
    .get_learner_feedback(id)       -> list[dict]
    .get_feature_follow_rates()     -> dict[feature, float]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    Integer, String, Text, create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.db_config import make_engine

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# ORM models (own Base — separate from auth models)
# ──────────────────────────────────────────────────────────────────────────────

class _Base(DeclarativeBase):
    pass


class FeedbackRecord(_Base):
    """One feedback submission per /explain call."""
    __tablename__ = "feedback_records"

    id                     = Column(Integer, primary_key=True, autoincrement=True)
    learner_id             = Column(String(64), nullable=False, index=True)
    explanation_id         = Column(String(64), nullable=True, index=True)
    rating                 = Column(Integer, nullable=True)          # 1-5
    followed_recommendation = Column(Boolean, nullable=True)         # did they act?
    top_action_feature     = Column(String(64), nullable=True)       # which feature was #1
    correction_feature     = Column(String(64), nullable=True)       # feature they corrected
    correction_comment     = Column(Text, nullable=True)             # free-text correction
    trust_score_at_time    = Column(Float, nullable=True)            # trust score when rated
    risk_score_at_time     = Column(Float, nullable=True)            # risk score when rated
    audience               = Column(String(16), nullable=True)       # "learner" | "instructor"
    created_at             = Column(DateTime(timezone=True), default=_now, nullable=False)


# ──────────────────────────────────────────────────────────────────────────────
# Input / output dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackInput:
    learner_id:              str
    rating:                  Optional[int]   = None   # 1-5
    followed_recommendation: Optional[bool]  = None
    top_action_feature:      Optional[str]   = None
    correction_feature:      Optional[str]   = None
    correction_comment:      Optional[str]   = None
    trust_score_at_time:     Optional[float] = None
    risk_score_at_time:      Optional[float] = None
    explanation_id:          Optional[str]   = None
    audience:                Optional[str]   = None


@dataclass
class FeatureFollowRate:
    feature:     str
    follow_rate: float          # fraction of times learners followed this action
    n_shown:     int            # how many times shown
    avg_impact:  Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "feature":     self.feature,
            "follow_rate": round(self.follow_rate, 3),
            "n_shown":     self.n_shown,
        }


@dataclass
class FeedbackStats:
    total_feedback:          int
    avg_rating:              Optional[float]
    recommendation_follow_rate: Optional[float]   # fraction who followed
    high_trust_follow_rate:  Optional[float]       # follow rate when trust ≥ 0.7
    low_trust_follow_rate:   Optional[float]       # follow rate when trust < 0.7
    top_corrected_features:  list[str]             # most frequently corrected
    feature_follow_rates:    list[FeatureFollowRate]
    ratings_distribution:    dict[int, int]        # {1: count, 2: count, ...}

    def to_dict(self) -> dict:
        return {
            "total_feedback":             self.total_feedback,
            "avg_rating":                 round(self.avg_rating, 2) if self.avg_rating else None,
            "recommendation_follow_rate": round(self.recommendation_follow_rate, 3)
                                          if self.recommendation_follow_rate is not None else None,
            "high_trust_follow_rate":     round(self.high_trust_follow_rate, 3)
                                          if self.high_trust_follow_rate is not None else None,
            "low_trust_follow_rate":      round(self.low_trust_follow_rate, 3)
                                          if self.low_trust_follow_rate is not None else None,
            "top_corrected_features":     self.top_corrected_features,
            "feature_follow_rates":       [f.to_dict() for f in self.feature_follow_rates],
            "ratings_distribution":       {str(k): v for k, v in self.ratings_distribution.items()},
        }


# ──────────────────────────────────────────────────────────────────────────────
# FeedbackStore
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackStore:
    def __init__(self, db_url: str = ""):
        if db_url:
            self._engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
                echo=False,
            )
        else:
            self._engine = make_engine("feedback")
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        log.info("FeedbackStore initialised")

    # ── Write ──────────────────────────────────────────────────────────────────

    def record(self, feedback: FeedbackInput) -> int:
        """Persist a feedback submission. Returns the new record id."""
        if feedback.rating is not None and not (1 <= feedback.rating <= 5):
            raise ValueError(f"rating must be 1-5, got {feedback.rating}")

        with self._Session() as db:
            rec = FeedbackRecord(
                learner_id              = feedback.learner_id,
                explanation_id          = feedback.explanation_id,
                rating                  = feedback.rating,
                followed_recommendation = feedback.followed_recommendation,
                top_action_feature      = feedback.top_action_feature,
                correction_feature      = feedback.correction_feature,
                correction_comment      = feedback.correction_comment,
                trust_score_at_time     = feedback.trust_score_at_time,
                risk_score_at_time      = feedback.risk_score_at_time,
                audience                = feedback.audience,
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
            log.info(
                "FeedbackStore.record  learner=%s  rating=%s  followed=%s",
                feedback.learner_id, feedback.rating, feedback.followed_recommendation,
            )
            return rec.id

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_learner_feedback(self, learner_id: str) -> list[dict]:
        """All feedback records for a given learner, newest first."""
        with self._Session() as db:
            rows = (
                db.query(FeedbackRecord)
                .filter(FeedbackRecord.learner_id == learner_id)
                .order_by(FeedbackRecord.created_at.desc())
                .all()
            )
            return [
                {
                    "id":                      r.id,
                    "explanation_id":          r.explanation_id,
                    "rating":                  r.rating,
                    "followed_recommendation": r.followed_recommendation,
                    "correction_feature":      r.correction_feature,
                    "correction_comment":      r.correction_comment,
                    "trust_score_at_time":     r.trust_score_at_time,
                    "risk_score_at_time":      r.risk_score_at_time,
                    "created_at":              r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def get_stats(self) -> FeedbackStats:
        """Compute aggregate feedback statistics across all learners."""
        with self._Session() as db:
            total = db.query(func.count(FeedbackRecord.id)).scalar() or 0

            # Average rating
            avg_rating = db.query(func.avg(FeedbackRecord.rating)).scalar()

            # Overall follow rate
            follow_rows = (
                db.query(FeedbackRecord.followed_recommendation)
                .filter(FeedbackRecord.followed_recommendation.isnot(None))
                .all()
            )
            follow_rate = (
                sum(1 for r in follow_rows if r[0]) / len(follow_rows)
                if follow_rows else None
            )

            # High trust vs low trust follow rates
            high_trust_rows = (
                db.query(FeedbackRecord.followed_recommendation)
                .filter(
                    FeedbackRecord.trust_score_at_time >= 0.7,
                    FeedbackRecord.followed_recommendation.isnot(None),
                )
                .all()
            )
            low_trust_rows = (
                db.query(FeedbackRecord.followed_recommendation)
                .filter(
                    FeedbackRecord.trust_score_at_time < 0.7,
                    FeedbackRecord.followed_recommendation.isnot(None),
                )
                .all()
            )
            high_trust_follow = (
                sum(1 for r in high_trust_rows if r[0]) / len(high_trust_rows)
                if high_trust_rows else None
            )
            low_trust_follow = (
                sum(1 for r in low_trust_rows if r[0]) / len(low_trust_rows)
                if low_trust_rows else None
            )

            # Top corrected features
            correction_rows = (
                db.query(FeedbackRecord.correction_feature)
                .filter(FeedbackRecord.correction_feature.isnot(None))
                .all()
            )
            correction_counts: dict[str, int] = {}
            for (feat,) in correction_rows:
                correction_counts[feat] = correction_counts.get(feat, 0) + 1
            top_corrected = sorted(correction_counts, key=correction_counts.get, reverse=True)[:5]

            # Per-feature follow rates
            feature_rows = (
                db.query(
                    FeedbackRecord.top_action_feature,
                    FeedbackRecord.followed_recommendation,
                )
                .filter(
                    FeedbackRecord.top_action_feature.isnot(None),
                    FeedbackRecord.followed_recommendation.isnot(None),
                )
                .all()
            )
            feat_totals: dict[str, list[bool]] = {}
            for feat, followed in feature_rows:
                feat_totals.setdefault(feat, []).append(bool(followed))
            feature_follow_rates = [
                FeatureFollowRate(
                    feature=feat,
                    follow_rate=sum(vals) / len(vals),
                    n_shown=len(vals),
                )
                for feat, vals in sorted(feat_totals.items())
            ]

            # Ratings distribution
            rating_rows = (
                db.query(FeedbackRecord.rating)
                .filter(FeedbackRecord.rating.isnot(None))
                .all()
            )
            ratings_dist: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for (r,) in rating_rows:
                if r in ratings_dist:
                    ratings_dist[r] += 1

        return FeedbackStats(
            total_feedback=total,
            avg_rating=float(avg_rating) if avg_rating else None,
            recommendation_follow_rate=follow_rate,
            high_trust_follow_rate=high_trust_follow,
            low_trust_follow_rate=low_trust_follow,
            top_corrected_features=top_corrected,
            feature_follow_rates=feature_follow_rates,
            ratings_distribution=ratings_dist,
        )

    def get_feature_follow_rates(self) -> dict[str, float]:
        """Quick lookup: feature → follow rate. Used by ActionRanker future reweighting."""
        stats = self.get_stats()
        return {r.feature: r.follow_rate for r in stats.feature_follow_rates}
