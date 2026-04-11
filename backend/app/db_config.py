"""
Central Database Configuration
================================
Reads DATABASE_URL from the environment and provides a shared SQLAlchemy
engine + session factory for all stores (auth, explanations, feedback, events, predictions).

In local dev without DATABASE_URL: falls back to per-store SQLite files
(backwards-compatible for running outside Docker).

In production (Docker / EC2): set DATABASE_URL to a PostgreSQL DSN:
    DATABASE_URL=postgresql://user:pass@host:5432/xai_db
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR     = _PROJECT_ROOT / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Connection URL ─────────────────────────────────────────────────────────────
# PostgreSQL in production, SQLite per-store in local dev (legacy fallback)
DATABASE_URL: str = os.getenv("DATABASE_URL", "")

_IS_POSTGRES = DATABASE_URL.startswith("postgresql")


def get_store_url(store_name: str) -> str:
    """
    Return the correct DB URL for a named store.

    If DATABASE_URL is set → use PostgreSQL for everything (production).
    Otherwise → use SQLite per-store file (local dev fallback).

    Parameters
    ----------
    store_name : "auth" | "explanations" | "feedback" | "events"
    """
    if _IS_POSTGRES:
        return DATABASE_URL
    return f"sqlite:///{_DATA_DIR / f'{store_name}.db'}"


def make_engine(store_name: str):
    """
    Create a SQLAlchemy engine for the given store.
    - PostgreSQL: connection pooling (pool_size=5, max_overflow=10)
    - SQLite:     check_same_thread=False
    """
    url = get_store_url(store_name)

    if url.startswith("postgresql"):
        return create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   # detect stale connections
            echo=False,
        )

    return create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False,
    )


def make_session_factory(store_name: str):
    """Return a sessionmaker bound to the given store's engine."""
    engine = make_engine(store_name)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine), engine
