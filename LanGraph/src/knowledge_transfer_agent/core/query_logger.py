"""
Query logging service for audit and analytics.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QueryLogEntry:
    """Single query log entry."""

    question: str
    answer: str
    confidence_score: float
    reflection_status: str
    citations_count: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryLogger(ABC):
    """Abstract query logger. Implement for DB, file, or external service."""

    @abstractmethod
    def log(self, entry: QueryLogEntry) -> None:
        """Persist a query log entry."""
        pass


class InMemoryQueryLogger(QueryLogger):
    """In-memory query logger. For production, use DB-backed implementation."""

    def __init__(self, max_entries: int = 10000) -> None:
        self._entries: list[QueryLogEntry] = []
        self._max_entries = max_entries

    def log(self, entry: QueryLogEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        logger.debug(
            "Query logged: question_len=%d, confidence=%.2f, latency=%.0fms",
            len(entry.question),
            entry.confidence_score,
            entry.latency_ms,
        )

    def get_recent(self, limit: int = 100) -> list[QueryLogEntry]:
        """Get recent entries (for debugging/admin)."""
        return self._entries[-limit:]


_query_logger: Optional[QueryLogger] = None


def get_query_logger() -> QueryLogger:
    """Get the query logger instance."""
    global _query_logger
    if _query_logger is None:
        _query_logger = InMemoryQueryLogger()
    return _query_logger


def set_query_logger(impl: QueryLogger) -> None:
    """Set custom query logger (e.g., for DI/testing)."""
    global _query_logger
    _query_logger = impl
