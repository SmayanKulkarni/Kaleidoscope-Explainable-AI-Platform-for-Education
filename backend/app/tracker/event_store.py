"""
Event Store — Implicit Behavioral Telemetry (Step 1)
=====================================================
Persists frontend interaction events to SQLite for later aggregation
into implicit engagement signals used by the retraining pipeline.

Table: interaction_events
Public API
----------
EventStore(db_url)
    .record_batch(events)            -> int  (records inserted)
    .get_learner_events(learner_id)  -> list[dict]
    .get_events_since(ts)            -> list[dict]
    .get_all_learner_ids()           -> list[str]
    .count()                         -> int
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.db_config import make_engine

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Base(DeclarativeBase):
    pass


class InteractionEvent(_Base):
    """One frontend interaction event."""
    __tablename__ = "interaction_events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    learner_id   = Column(String(64), nullable=False, index=True)
    session_id   = Column(String(64), nullable=True,  index=True)
    event_type   = Column(String(32), nullable=False, index=True)
    event_target = Column(String(128), nullable=True)
    event_value  = Column(Float,  nullable=True)   # e.g. time_ms, scroll_pct, slider_value
    page         = Column(String(64),  nullable=True)
    extra        = Column(Text,        nullable=True)  # JSON blob for misc metadata
    client_ts    = Column(DateTime(timezone=True), nullable=True)   # browser timestamp
    server_ts    = Column(DateTime(timezone=True), default=_now, nullable=False)


# ──────────────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────────────

class EventStore:
    def __init__(self, db_url: str = ""):
        if db_url:
            self._engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
                echo=False,
            )
        else:
            self._engine = make_engine("events")
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        log.info("EventStore initialised")

    def record_batch(self, events: list[dict]) -> int:
        """
        Insert a batch of events. Each dict may contain:
            learner_id, session_id, event_type, event_target,
            event_value, page, extra, client_ts
        Returns number of records inserted.
        """
        if not events:
            return 0
        rows = []
        for e in events:
            client_ts = None
            if e.get("client_ts"):
                try:
                    client_ts = datetime.fromisoformat(str(e["client_ts"]).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            rows.append(InteractionEvent(
                learner_id   = str(e["learner_id"]),
                session_id   = e.get("session_id"),
                event_type   = str(e["event_type"]),
                event_target = e.get("event_target"),
                event_value  = float(e["event_value"]) if e.get("event_value") is not None else None,
                page         = e.get("page"),
                extra        = str(e["extra"]) if e.get("extra") else None,
                client_ts    = client_ts,
            ))
        with self._Session() as db:
            db.add_all(rows)
            db.commit()
        log.debug("EventStore.record_batch  inserted=%d", len(rows))
        return len(rows)

    def get_learner_events(
        self,
        learner_id: str,
        since: Optional[datetime] = None,
        event_types: Optional[list[str]] = None,
    ) -> list[dict]:
        with self._Session() as db:
            q = db.query(InteractionEvent).filter(
                InteractionEvent.learner_id == learner_id
            )
            if since:
                q = q.filter(InteractionEvent.server_ts >= since)
            if event_types:
                q = q.filter(InteractionEvent.event_type.in_(event_types))
            rows = q.order_by(InteractionEvent.server_ts.asc()).all()
        return [
            {
                "id":           r.id,
                "learner_id":   r.learner_id,
                "session_id":   r.session_id,
                "event_type":   r.event_type,
                "event_target": r.event_target,
                "event_value":  r.event_value,
                "page":         r.page,
                "extra":        r.extra,
                "server_ts":    r.server_ts.isoformat() if r.server_ts else None,
            }
            for r in rows
        ]

    def get_events_since(self, since: datetime) -> list[dict]:
        with self._Session() as db:
            rows = (
                db.query(InteractionEvent)
                .filter(InteractionEvent.server_ts >= since)
                .order_by(InteractionEvent.server_ts.asc())
                .all()
            )
        return [
            {
                "learner_id":   r.learner_id,
                "session_id":   r.session_id,
                "event_type":   r.event_type,
                "event_target": r.event_target,
                "event_value":  r.event_value,
                "page":         r.page,
                "server_ts":    r.server_ts.isoformat() if r.server_ts else None,
            }
            for r in rows
        ]

    def get_all_learner_ids(self) -> list[str]:
        with self._Session() as db:
            rows = (
                db.query(InteractionEvent.learner_id)
                .distinct()
                .all()
            )
        return [r[0] for r in rows]

    def count(self) -> int:
        with self._Session() as db:
            return db.query(func.count(InteractionEvent.id)).scalar() or 0
