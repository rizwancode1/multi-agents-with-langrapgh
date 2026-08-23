"""
Response Caching Layer
In-memory cache with TTL for LLM response deduplication.
Production mode uses Redis (shared, persistent across restarts).
"""

import hashlib
import time

from app.config import get_settings
from app.monitoring import get_logger

logger = get_logger("cache")


class ResponseCache:
    """
    In-memory response cache with TTL (time-to-live).

    Used in development; production uses RedisResponseCache.
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        """Create a cache key from the normalized query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        """
        Get cached response if it exists and hasn't expired.
        Returns None on cache miss.
        """
        key = self._make_key(query)

        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self._hits += 1
                return entry["response"]
            else:
                del self._cache[key]

        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        """Cache a response."""
        key = self._make_key(query)
        self._cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "query": query,
        }

    @property
    def stats(self) -> dict:
        """Cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "backend": "memory",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }


class RedisResponseCache:
    """
    Redis-backed response cache with native TTL expiry.

    Shares cached responses across API instances and persists
    across restarts. Same interface as ResponseCache.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 300, prefix: str = "rag:resp:"):
        import redis as redis_lib

        self.ttl = ttl_seconds
        self._prefix = prefix
        self._client = redis_lib.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        normalized = query.lower().strip()
        return f"{self._prefix}{hashlib.sha256(normalized.encode()).hexdigest()}"

    def get(self, query: str) -> str | None:
        try:
            value = self._client.get(self._make_key(query))
        except Exception as e:
            logger.warning("redis_get_failed", extra={"extra_data": {"error": str(e)}})
            value = None

        if value is not None:
            self._hits += 1
            return value

        self._misses += 1
        return None

    def set(self, query: str, response: str) -> None:
        try:
            self._client.setex(self._make_key(query), self.ttl, response)
        except Exception as e:
            logger.warning("redis_set_failed", extra={"extra_data": {"error": str(e)}})

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        try:
            cached_entries = self._client.dbsize()
        except Exception:
            cached_entries = -1
        return {
            "backend": "redis",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": cached_entries,
        }


def create_cache(ttl_seconds: int = 300) -> ResponseCache | RedisResponseCache:
    """
    Cache factory following the checkpoints.py backend-switch pattern.

    CACHE_BACKEND:
      - "memory": always in-memory
      - "redis":  always Redis (hard requirement)
      - "auto":   Redis when APP_ENV=production, memory otherwise
    Falls back to in-memory with a warning if Redis is unreachable.
    """
    settings = get_settings()
    backend = settings.cache_backend.lower()

    use_redis = backend == "redis" or (backend == "auto" and settings.is_production)

    if use_redis:
        try:
            cache = RedisResponseCache(settings.redis_url, ttl_seconds=ttl_seconds)
            cache._client.ping()
            logger.info("cache_backend_selected", extra={"extra_data": {"backend": "redis"}})
            return cache
        except Exception as e:
            logger.warning(
                "redis_unreachable_fallback_memory",
                extra={"extra_data": {"redis_url": settings.redis_url, "error": str(e)}},
            )

    logger.info("cache_backend_selected", extra={"extra_data": {"backend": "memory"}})
    return ResponseCache(ttl_seconds=ttl_seconds)
