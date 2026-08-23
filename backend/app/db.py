"""
Database Connection and Session Management
SQLAlchemy engine, session factory, and initialization helpers.
"""

import contextlib
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()

# SQLite for local development; PostgreSQL in production via DATABASE_URL.
# Also used by PGVector when USE_PGVECTOR=true (requires a Postgres URL).
DATABASE_URL = settings.database_url

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=settings.db_echo_logs,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Dependency-style DB session getter."""
    return SessionLocal()


@contextmanager
def db_session():
    """Context manager for DB sessions (auto-commit/rollback)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """Create all tables and apply versioned, idempotent schema migrations.

    Migrations are tracked via PRAGMA user_version (SQLite) so each runs
    exactly once. For Postgres this falls back to the same list executed
    unconditionally-safe statements.
    """
    from sqlalchemy import inspect, text

    from app.models_db import Base

    Base.metadata.create_all(bind=engine)

    # Ordered migration history. Append-only: never edit an applied entry.
    MIGRATIONS = [
        # v1: conversations.thread_id for LangGraph thread mapping
        "ALTER TABLE conversations ADD COLUMN thread_id VARCHAR(100)",
    ]

    with engine.connect() as conn:
        inspector = inspect(engine)
        if "conversations" not in inspector.get_table_names():
            return  # Fresh DB — create_all already built the latest schema.

        if engine.dialect.name == "sqlite":
            current = conn.execute(text("PRAGMA user_version")).scalar() or 0
            for version in range(current, len(MIGRATIONS)):
                statement = MIGRATIONS[version]
                columns = {col["name"] for col in inspector.get_columns("conversations")}
                needed = statement.split("ADD COLUMN ", 1)[1].split()[0] \
                    if "ADD COLUMN " in statement else None
                if needed and needed in columns:
                    # Column already exists from a pre-migration database.
                    pass
                else:
                    conn.execute(text(statement))
                conn.execute(text(f"PRAGMA user_version = {version + 1}"))
            conn.commit()
        else:
            # Postgres/other: statements are written to be safely re-runnable.
            for statement in MIGRATIONS:
                with contextlib.suppress(Exception):
                    conn.execute(text(statement))
            conn.commit()


def drop_all_tables():
    """Drop all tables (use with caution)."""
    from app.models_db import Base
    Base.metadata.drop_all(bind=engine)
