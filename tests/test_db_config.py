"""
Tests for db_config — verifies SQLite fallback and engine creation.
These run without a live Postgres instance.
"""
import os
import pytest


def test_get_store_url_sqlite_fallback(tmp_path, monkeypatch):
    """With DATABASE_URL unset, each store gets a distinct SQLite path."""
    monkeypatch.setenv("DATABASE_URL", "")

    # Re-import after env change to pick up the new value
    import importlib
    import backend.app.db_config as cfg
    importlib.reload(cfg)

    url_auth     = cfg.get_store_url("auth")
    url_feedback = cfg.get_store_url("feedback")
    url_events   = cfg.get_store_url("events")

    assert url_auth.startswith("sqlite:///")
    assert "auth.db" in url_auth
    assert "feedback.db" in url_feedback
    assert "events.db" in url_events
    assert url_auth != url_feedback


def test_get_store_url_postgres(monkeypatch):
    """With DATABASE_URL set to postgres, all stores share the same URL."""
    pg_url = "postgresql://user:pass@localhost:5432/xai_db"
    monkeypatch.setenv("DATABASE_URL", pg_url)

    import importlib
    import backend.app.db_config as cfg
    importlib.reload(cfg)

    assert cfg.get_store_url("auth") == pg_url
    assert cfg.get_store_url("events") == pg_url
    assert cfg.get_store_url("feedback") == pg_url


def test_make_engine_sqlite(tmp_path, monkeypatch):
    """make_engine returns a working SQLite engine in fallback mode."""
    monkeypatch.setenv("DATABASE_URL", "")

    import importlib
    import backend.app.db_config as cfg
    importlib.reload(cfg)

    engine = cfg.make_engine("auth")
    # Should connect without error
    with engine.connect() as conn:
        result = conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        assert result.scalar() == 1
