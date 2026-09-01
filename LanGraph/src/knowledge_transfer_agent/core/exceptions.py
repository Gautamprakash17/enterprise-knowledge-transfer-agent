"""
Custom exceptions for structured error handling.
"""

from typing import Any, Optional


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class VectorStoreError(AgentError):
    """Vector store not available or operation failed."""

    pass


class RetrievalError(AgentError):
    """Document retrieval failed."""

    pass


class LLMError(AgentError):
    """LLM call failed (timeout, rate limit, API error)."""

    pass


class IngestionError(AgentError):
    """Document ingestion failed."""

    pass


class ValidationError(AgentError):
    """Validation failed (e.g., reflection, citation check)."""

    pass
