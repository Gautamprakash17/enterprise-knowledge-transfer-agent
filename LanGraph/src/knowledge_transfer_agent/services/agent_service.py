"""
Agent service: orchestrates graph invocation with error handling and query logging.
Refactored for scalability and separation of concerns.
"""

import time
from typing import Any, Optional

from knowledge_transfer_agent.agent.context_formatter import extract_citations_from_documents
from knowledge_transfer_agent.agent.graph import create_knowledge_agent_graph
from knowledge_transfer_agent.core.exceptions import AgentError, LLMError, RetrievalError, VectorStoreError
from knowledge_transfer_agent.core.query_logger import QueryLogEntry, get_query_logger
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


class AgentService:
    """
    Service for invoking the knowledge transfer agent.
    Handles errors, logging, and response shaping.
    """

    def __init__(
        self,
        vector_store: Any = None,
        query_logger=None,
    ) -> None:
        self._vector_store = vector_store
        self._query_logger = query_logger or get_query_logger()

    def ask(
        self,
        question: str,
        thread_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run agent and return structured result.
        Raises AgentError (or subclass) on failure.
        """
        start = time.perf_counter()
        try:
            graph = create_knowledge_agent_graph(vector_store=self._vector_store)
            config = {"configurable": {"thread_id": thread_id or "ask"}}
            result = graph.invoke({"question": question}, config=config)
        except (LLMError, RetrievalError, VectorStoreError) as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self._log_query(question, "", 0.0, "error", 0, latency_ms, False, str(e), thread_id)
            raise
        except Exception as e:
            logger.exception("Agent invocation failed")
            latency_ms = (time.perf_counter() - start) * 1000
            self._log_query(question, "", 0.0, "error", 0, latency_ms, False, str(e), thread_id)
            raise AgentError(f"Agent failed: {e}", details={"error": str(e)}) from e

        latency_ms = (time.perf_counter() - start) * 1000
        answer = result.get("answer", "No response generated.")
        reflection = result.get("reflection", "unknown")
        confidence = result.get("confidence_score", 0.0)
        retrieved_docs = result.get("retrieved_docs", [])
        citations = extract_citations_from_documents(retrieved_docs)

        self._log_query(
            question, answer, confidence, reflection, len(citations),
            latency_ms, True, None, thread_id,
        )
        return {
            "answer": answer,
            "citations": citations,
            "reflection_status": reflection,
            "confidence_score": confidence,
            "retrieved_docs": retrieved_docs,
        }

    def _log_query(
        self,
        question: str,
        answer: str,
        confidence: float,
        reflection: str,
        citations_count: int,
        latency_ms: float,
        success: bool,
        error: Optional[str],
        thread_id: Optional[str],
    ) -> None:
        """Log query to query logger."""
        try:
            entry = QueryLogEntry(
                question=question,
                answer=answer,
                confidence_score=confidence,
                reflection_status=reflection,
                citations_count=citations_count,
                latency_ms=latency_ms,
                success=success,
                error=error,
                thread_id=thread_id,
            )
            self._query_logger.log(entry)
        except Exception as e:
            logger.warning("Query logging failed: %s", e)
