"""
Database Connection and Session Management
SQLAlchemy engine, session factory, and initialization helpers.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from pathlib import Path
from app.config import get_settings

settings = get_settings()

# SQLite for local development; PostgreSQL in production via DATABASE_URL.
# Also used by PGVector when USE_PGVECTOR=true (requires a Postgres URL).
DATABASE_URL = settings.database_url

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=settings.is_production is False,
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
    """Create all tables and apply lightweight migrations for existing databases."""
    from app.models_db import Base
    from sqlalchemy import inspect, text

    Base.metadata.create_all(bind=engine)

    # Lightweight migration: add conversations.thread_id if missing
    with engine.connect() as conn:
        inspector = inspect(engine)
        if "conversations" in inspector.get_table_names():
            columns = {col["name"] for col in inspector.get_columns("conversations")}
            if "thread_id" not in columns:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN thread_id VARCHAR(100)"))
                conn.commit()


def drop_all_tables():
    """Drop all tables (use with caution)."""
    from app.models_db import Base
    Base.metadata.drop_all(bind=engine)
