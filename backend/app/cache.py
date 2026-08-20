"""
Response Caching Layer
In-memory cache with TTL for LLM response deduplication.
"""

import hashlib
import time
from typing import Optional


class ResponseCache:
    """
    In-memory response cache with TTL (time-to-live).

    In production, replace this with Redis for:
    - Persistence across restarts
    - Shared cache across multiple instances
    - Built-in TTL management
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

    def get(self, query: str) -> Optional[str]:
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
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1%}",
            "cached_entries": len(self._cache),
        }
