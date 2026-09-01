"""
Agent service: orchestrates graph invocation with error handling and query logging.
Refactored for scalability and separation of concerns.
"""

import time
from typing import Any, Iterator, Optional

from langgraph.checkpoint.memory import MemorySaver

from knowledge_transfer_agent.agent.context_formatter import extract_citations_from_documents
from knowledge_transfer_agent.agent.graph import create_knowledge_agent_graph
from knowledge_transfer_agent.core.exceptions import AgentError, LLMError, RetrievalError, VectorStoreError
from knowledge_transfer_agent.core.guardrails import apply_output_guardrails
from knowledge_transfer_agent.core.metrics import observe_ask
from knowledge_transfer_agent.core.query_logger import QueryLogEntry, get_query_logger
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

_CHECKPOINTER = MemorySaver()


class AgentService:
    """
    Service for invoking the knowledge transfer agent.
    Handles errors, logging, and response shaping.
    """

    def __init__(
        self,
        vector_store: Any = None,
        query_logger=None,
        checkpointer: MemorySaver | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._query_logger = query_logger or get_query_logger()
        self._checkpointer = checkpointer or _CHECKPOINTER

    def ask(
        self,
        question: str,
        thread_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run agent and return structured result.
        Raises AgentError (or subclass) on failure.
        """
        start = time.perf_counter()
        try:
            graph = create_knowledge_agent_graph(
                vector_store=self._vector_store,
                checkpointer=self._checkpointer,
            )
            config = {"configurable": {"thread_id": thread_id or "ask"}}
            # Reset per-ask orchestration fields so MemorySaver does not leak
            # prior agent_trace / docs into this run (conversation_history stays).
            result = graph.invoke(
                {
                    "question": question,
                    "workspace_id": workspace_id or "default",
                    "thread_id": thread_id or "ask",
                    "agent_trace": [],
                    "active_agent": "",
                    "next_agent": "",
                    "guardrails_blocked": False,
                    "guardrail_flags": [],
                    "retrieved_docs": [],
                    "context_docs": [],
                    "answer": "",
                    "is_valid": True,
                    "validation_issues": [],
                    "sub_queries": [],
                    "hop_index": 0,
                    "reflection_retries": 0,
                    "shared_memory_context": "",
                    "shared_memories": [],
                    "shared_memory_saved_id": "",
                    "agent_timings_ms": {},
                },
                config=config,
            )
        except (LLMError, RetrievalError, VectorStoreError) as e:
            latency_s = time.perf_counter() - start
            latency_ms = latency_s * 1000
            observe_ask(
                latency_seconds=latency_s,
                success=False,
                error_type=type(e).__name__,
            )
            self._log_query(
                question, "", 0.0, "error", 0, latency_ms, False, str(e), thread_id, workspace_id
            )
            raise
        except Exception as e:
            logger.exception("Agent invocation failed")
            latency_s = time.perf_counter() - start
            latency_ms = latency_s * 1000
            observe_ask(
                latency_seconds=latency_s,
                success=False,
                error_type=type(e).__name__,
            )
            self._log_query(
                question, "", 0.0, "error", 0, latency_ms, False, str(e), thread_id, workspace_id
            )
            raise AgentError(f"Agent failed: {e}", details={"error": str(e)}) from e

        latency_s = time.perf_counter() - start
        latency_ms = latency_s * 1000
        answer = result.get("answer", "No response generated.")
        out = apply_output_guardrails(answer)
        answer = out.text
        reflection = result.get("reflection", "unknown")
        confidence = result.get("confidence_score", 0.0)
        retrieved_docs = result.get("retrieved_docs", [])
        citations = extract_citations_from_documents(retrieved_docs)
        guardrail_flags = list(result.get("guardrail_flags") or []) + list(out.flags)
        blocked = bool(result.get("guardrails_blocked"))
        observe_ask(
            latency_seconds=latency_s,
            success=True,
            blocked=blocked,
        )

        self._log_query(
            question, answer, confidence, reflection, len(citations),
            latency_ms, True, None, thread_id, workspace_id,
        )
        return {
            "answer": answer,
            "citations": citations,
            "reflection_status": reflection,
            "confidence_score": confidence,
            "retrieved_docs": retrieved_docs,
            "agent_trace": list(result.get("agent_trace") or []),
            "guardrail_flags": guardrail_flags,
            "guardrails_blocked": blocked,
            "shared_memories_used": len(result.get("shared_memories") or []),
            "shared_memory_saved_id": result.get("shared_memory_saved_id"),
            "agent_timings_ms": dict(result.get("agent_timings_ms") or {}),
            "latency_ms": round(latency_ms, 2),
        }

    def stream_ask(
        self,
        question: str,
        thread_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Stream the answer token-by-token for lower perceived latency.

        Yields event dicts:
          {"type": "token", "text": "..."}  -> partial answer text
          {"type": "done", "citations": [...]}  -> final metadata
          {"type": "error", "error": "..."}  -> on failure

        Note: this path runs planner + retrieval, then streams ONLY the generation
        step. Reflection/confidence are skipped here (they run after the full answer
        in the non-streaming endpoint), trading that check for speed.
        """
        # Local imports to avoid circular imports at module load.
        from knowledge_transfer_agent.agent.context_formatter import (
            format_documents_for_context,
        )
        from knowledge_transfer_agent.agent.llm_client import get_llm_with_retry
        from knowledge_transfer_agent.agent.nodes import create_retrieve_node, planner_node
        from knowledge_transfer_agent.agent.prompts import GENERATE_PROMPT
        from knowledge_transfer_agent.core.guardrails import (
            apply_input_guardrails_soft,
            apply_output_guardrails,
        )

        start = time.perf_counter()
        try:
            gated = apply_input_guardrails_soft(question)
            if gated.blocked:
                yield {
                    "type": "token",
                    "text": (
                        "I can't process this request because it violates safety guardrails."
                    ),
                }
                yield {
                    "type": "done",
                    "citations": [],
                    "guardrail_flags": gated.flags,
                    "guardrails_blocked": True,
                }
                return

            state: dict[str, Any] = {"question": gated.text}
            state.update(planner_node(state))
            retrieve_node = create_retrieve_node(vector_store=self._vector_store)
            state.update(retrieve_node(state))

            docs = state.get("retrieved_docs", [])
            if not docs:
                yield {"type": "token", "text": "No sufficient data"}
                yield {"type": "done", "citations": []}
                return

            context = format_documents_for_context(docs)
            llm = get_llm_with_retry()
            chain = GENERATE_PROMPT | llm
            inputs = {
                "context": context,
                "question": gated.text,
                "history": "",
                "shared_memory": "(none)",
            }

            full_text = ""
            for chunk in chain.stream(inputs):
                text = getattr(chunk, "content", "") or ""
                if text:
                    full_text += text
                    yield {"type": "token", "text": text}

            safe = apply_output_guardrails(full_text)
            citations = extract_citations_from_documents(docs)
            latency_ms = (time.perf_counter() - start) * 1000
            self._log_query(
                question, safe.text, 0.0, "streamed", len(citations),
                latency_ms, True, None, thread_id, workspace_id,
            )
            done: dict[str, Any] = {
                "type": "done",
                "citations": citations,
                "guardrail_flags": list(gated.flags) + list(safe.flags),
            }
            if safe.text != full_text:
                done["sanitized_answer"] = safe.text
            yield done
        except Exception as e:
            logger.exception("Streaming agent failed")
            yield {"type": "error", "error": str(e)}

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
        workspace_id: Optional[str] = None,
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
                metadata={"workspace_id": workspace_id or "", "user_id": "default"},
            )
            self._query_logger.log(entry)
        except Exception as e:
            logger.warning("Query logging failed: %s", e)
