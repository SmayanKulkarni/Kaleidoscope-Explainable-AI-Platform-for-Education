"""
Shared pytest fixtures.
Sets environment variables before any import of app modules so db_config
and auth use in-memory SQLite during tests (no Postgres required locally).
"""
import os
import pytest

# ── Override DB + auth before any app module is imported ─────────────────────
os.environ.setdefault("DATABASE_URL", "")          # empty → per-store SQLite fallback
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("GROQ_API_KEY",   "")        # narration disabled in tests
os.environ.setdefault("AWS_S3_BUCKET",  "")        # S3 disabled in tests


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
