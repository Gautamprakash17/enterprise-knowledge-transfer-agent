"""
Caching layer interface. Implement for Redis, Memcached, etc.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key. Returns None if missing or expired."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set key-value with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete key."""
        pass


class InMemoryCache(CacheBackend):
    """In-memory cache. Thread-safe for simple use. For scaling, use Redis."""

    def __init__(self, default_ttl: int = 300, max_size: int = 1000) -> None:
        self._store: dict[str, tuple[Any, Optional[float]]] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        val, expires = self._store[key]
        if expires is not None and expires < __import__("time").time():
            del self._store[key]
            return None
        return val

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        import time as _time
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires = _time.time() + ttl if ttl > 0 else None
        if len(self._store) >= self._max_size and key not in self._store:
            # Evict oldest (simplified: remove first)
            first = next(iter(self._store))
            del self._store[first]
        self._store[key] = (value, expires)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries (e.g. after FAISS reindex so retrieval is not stale)."""
        self._store.clear()


def cache_key(prefix: str, *parts: Any) -> str:
    """Generate cache key from parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    h = hashlib.sha256(raw.encode()).hexdigest()
    return f"{prefix}:{h}"


_cache: Optional[CacheBackend] = None


def get_cache() -> CacheBackend:
    """Get cache backend (from config or default)."""
    global _cache
    if _cache is None:
        try:
            settings = get_settings()
            ttl = getattr(settings, "cache_ttl_seconds", 300)
            size = getattr(settings, "cache_max_size", 1000)
            _cache = InMemoryCache(default_ttl=ttl, max_size=size)
        except Exception:
            _cache = InMemoryCache()
    return _cache


def set_cache(backend: CacheBackend) -> None:
    """Set cache backend (for DI/testing)."""
    global _cache
    _cache = backend


def invalidate_all_caches() -> None:
    """
    Clear in-memory caches after the vector index changes.

    Retrieval results are cached by query; after reindex/replace, those entries can
    still point at old chunk metadata (e.g. missing page_number).
    """
    global _cache
    if _cache is None:
        return
    clear = getattr(_cache, "clear", None)
    if callable(clear):
        clear()
