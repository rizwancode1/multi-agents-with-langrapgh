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

# Use SQLite for local development; swap connection string for PostgreSQL/MySQL in production
DATABASE_URL = "sqlite:///./orders.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
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
    """Create all tables."""
    from app.models_db import Base
    Base.metadata.create_all(bind=engine)


def drop_all_tables():
    """Drop all tables (use with caution)."""
    from app.models_db import Base
    Base.metadata.drop_all(bind=engine)
