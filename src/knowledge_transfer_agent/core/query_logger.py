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


class SQLiteQueryLogger(QueryLogger):
    """Persist audit entries to SQLite."""

    def log(self, entry: QueryLogEntry) -> None:
        if not get_settings().persist_to_database:
            return
        try:
            from knowledge_transfer_agent.core.database import insert_query_audit

            ws = entry.metadata.get("workspace_id")
            insert_query_audit(
                workspace_id=ws,
                thread_id=entry.thread_id,
                user_id=entry.metadata.get("user_id", "default"),
                question=entry.question,
                answer=entry.answer,
                confidence_score=entry.confidence_score,
                reflection_status=entry.reflection_status,
                citations_count=entry.citations_count,
                latency_ms=entry.latency_ms,
                success=entry.success,
                error=entry.error,
            )
        except Exception as e:
            logger.warning("SQLite query log failed: %s", e)
        logger.debug(
            "Query logged: question_len=%d, confidence=%.2f, latency=%.0fms",
            len(entry.question),
            entry.confidence_score,
            entry.latency_ms,
        )


class CompositeQueryLogger(QueryLogger):
    """Memory + SQLite for dev visibility and production audit."""

    def __init__(self) -> None:
        self._memory = InMemoryQueryLogger()

    def log(self, entry: QueryLogEntry) -> None:
        self._memory.log(entry)
        SQLiteQueryLogger().log(entry)

    def get_recent(self, limit: int = 100) -> list[QueryLogEntry]:
        return self._memory.get_recent(limit)


def get_query_logger() -> QueryLogger:
    """Get the query logger instance."""
    global _query_logger
    if _query_logger is None:
        if get_settings().persist_to_database:
            _query_logger = CompositeQueryLogger()
        else:
            _query_logger = InMemoryQueryLogger()
    return _query_logger


def set_query_logger(impl: QueryLogger) -> None:
    """Set custom query logger (e.g., for DI/testing)."""
    global _query_logger
    _query_logger = impl
