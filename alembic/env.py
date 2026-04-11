"""
Alembic env.py — XAI Learning Recommendation System
======================================================
Reads DATABASE_URL from environment (via db_config) and discovers all ORM
models from auth, tracker, and event modules for autogenerate support.

Run migrations:
    alembic upgrade head

Generate a new revision after model changes:
    alembic revision --autogenerate -m "describe change"
"""
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy import engine_from_config, create_engine

from alembic import context

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Import ALL ORM models so autogenerate can see every table ─────────────────
from backend.app.auth.models import Base as AuthBase                    # users, roles
from backend.app.tracker.consistency_store import Base as ExplainBase   # explanation_records
from backend.app.tracker.feedback_store import _Base as FeedbackBase    # feedback_records
from backend.app.tracker.event_store import _Base as EventBase          # interaction_events

# Combine all metadata into a single target for autogenerate
from sqlalchemy import MetaData

target_metadata = MetaData()
for base in (AuthBase, ExplainBase, FeedbackBase, EventBase):
    for table in base.metadata.tables.values():
        table.tometadata(target_metadata)

# ── Resolve DB URL via db_config (picks up DATABASE_URL env var) ──────────────
from backend.app.db_config import get_store_url

_db_url = get_store_url("auth")   # all stores share the same URL in postgres mode


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = create_engine(_db_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,       # detect column type changes
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
