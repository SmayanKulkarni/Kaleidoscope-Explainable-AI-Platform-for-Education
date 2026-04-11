"""
Auth Database — SQLAlchemy engine, session factory, and dependency.
Uses DATABASE_URL (PostgreSQL in production) or SQLite fallback in local dev.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from backend.app.auth.models import Base
from backend.app.db_config import make_engine, make_session_factory

_SessionFactory, engine = make_session_factory("auth")


def init_db() -> None:
    """Create all auth tables. Call once at startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a SQLAlchemy session."""
    db: Session = _SessionFactory()
    try:
        yield db
    finally:
        db.close()
