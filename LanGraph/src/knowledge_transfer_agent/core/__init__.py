"""
Core infrastructure: exceptions, caching, query logging.
"""

from knowledge_transfer_agent.core.cache import CacheBackend, InMemoryCache, cache_key, get_cache
from knowledge_transfer_agent.core.exceptions import (
    AgentError,
    LLMError,
    RetrievalError,
    VectorStoreError,
)
from knowledge_transfer_agent.core.query_logger import QueryLogger, get_query_logger

__all__ = [
    "AgentError",
    "RetrievalError",
    "VectorStoreError",
    "LLMError",
    "CacheBackend",
    "InMemoryCache",
    "cache_key",
    "get_cache",
    "QueryLogger",
    "get_query_logger",
]
