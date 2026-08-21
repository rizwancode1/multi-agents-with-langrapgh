"""Unit tests for the response cache layer."""

import time

from app.cache import ResponseCache, create_cache


def test_response_cache_hit_and_miss():
    cache = ResponseCache(ttl_seconds=60)

    assert cache.get("hello") is None  # miss
    cache.set("hello", "world")
    assert cache.get("hello") == "world"  # hit


def test_response_cache_ttl_expiry():
    cache = ResponseCache(ttl_seconds=0)
    cache.set("q", "a")
    time.sleep(0.01)
    assert cache.get("q") is None  # expired


def test_response_cache_normalizes_query():
    cache = ResponseCache(ttl_seconds=60)
    cache.set("Hello World", "answer")
    assert cache.get("  hello world ") == "answer"


def test_stats_shape():
    cache = ResponseCache(ttl_seconds=60)
    cache.set("q", "a")
    cache.get("q")
    cache.get("missing")

    stats = cache.stats
    assert stats["backend"] == "memory"
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == "50.0%"
    assert stats["cached_entries"] == 1


def test_create_cache_memory_backend():
    cache = create_cache.__wrapped__ if hasattr(create_cache, "__wrapped__") else None
    # Explicit memory backend must never touch Redis
    from app.cache import ResponseCache as RC
    import app.cache as cache_module

    original = cache_module.get_settings

    class FakeSettings:
        cache_backend = "memory"
        is_production = True
        redis_url = "redis://localhost:6379/0"

    try:
        cache_module.get_settings = lambda: FakeSettings()
        c = cache_module.create_cache(ttl_seconds=30)
        assert isinstance(c, RC)
        assert c.stats["backend"] == "memory"
    finally:
        cache_module.get_settings = original
