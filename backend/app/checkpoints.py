"""
Checkpoint Storage Factory
Provides LangGraph checkpoint savers based on environment configuration.

Supported backends:
- memory: MemorySaver (development, non-persistent)
- sqlite: AsyncSqliteSaver (local persistent storage, async-compatible)
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.config import get_settings

settings = get_settings()


async def get_checkpointer() -> BaseCheckpointSaver:
    """
    Return a BaseCheckpointSaver instance based on CHECKPOINT_STORAGE config.

    Async-compatible so it works with ``astream_events`` / ``astream``.
    """
    storage = settings.checkpoint_storage.lower()

    if storage == "memory":
        return MemorySaver()

    if storage == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(settings.checkpoint_path)
        await conn.execute("PRAGMA journal_mode=WAL")
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        return saver

    raise ValueError(
        f"Unsupported CHECKPOINT_STORAGE: {storage!r}. "
        "Use 'memory' or 'sqlite'."
    )
