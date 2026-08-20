"""
Checkpoint Storage Factory
Provides LangGraph checkpoint savers based on environment configuration.

Supported backends:
- memory: MemorySaver (development, non-persistent)
- sqlite: SqliteSaver (local persistent storage)
"""

import sqlite3
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.config import get_settings

settings = get_settings()


def get_checkpointer() -> BaseCheckpointSaver:
    """
    Return a BaseCheckpointSaver instance based on CHECKPOINT_STORAGE config.
    """
    storage = settings.checkpoint_storage.lower()

    if storage == "memory":
        return MemorySaver()

    if storage == "sqlite":
        conn = sqlite3.connect(
            settings.checkpoint_path,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=conn)

    raise ValueError(
        f"Unsupported CHECKPOINT_STORAGE: {storage!r}. "
        "Use 'memory' or 'sqlite'."
    )
