"""
LangGraph nodes: retrieve, generate, reflection, confidence.
Clear separation of concerns. Uses retry, caching, structured output.
"""

import re
from typing import Any

from langchain_core.documents import Document

from knowledge_transfer_agent.agent.context_formatter import format_documents_for_context
from knowledge_transfer_agent.agent.llm_client import get_llm_with_retry, invoke_cached, invoke_with_retry
from knowledge_transfer_agent.agent.output_schemas import ReflectionVerdict
from knowledge_transfer_agent.agent.prompts import (
    GENERATE_PROMPT,
    REFLECTION_PROMPT,
    REFLECTION_STRUCTURED_PROMPT,
)
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.cache import cache_key, get_cache
from knowledge_transfer_agent.core.exceptions import RetrievalError
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.retrieval.retriever_factory import get_hybrid_retriever

logger = get_logger(__name__)

_ABSTAIN_EXACT = "no sufficient data"


def _is_abstain_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    t = answer.strip().lower().rstrip('.!?"\'').strip()
    return t == _ABSTAIN_EXACT


def create_retrieve_node(vector_store=None, cache=None):
    """Factory: returns retrieve_node that uses the given vector_store (for DI)."""

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        """Retrieve top-k documents from vector store for the question."""
        question = state.get("question", "")
        if not question:
            return {"retrieved_docs": []}

        settings = get_settings()
        top_k = settings.top_k_semantic + settings.top_k_keyword
        cache_backend = cache or (get_cache() if settings.cache_enabled else None)

        if cache_backend and settings.cache_enabled:
            key = cache_key("retrieve", question, top_k)
            cached = cache_backend.get(key)
            if cached is not None:
                logger.debug("Retrieve cache hit")
                return {"retrieved_docs": cached}
        try:
            retriever = get_hybrid_retriever(vector_store=vector_store)
            docs = retriever.invoke(question)[:top_k]
        except Exception as e:
            logger.exception("Retrieval failed: %s", e)
            raise RetrievalError(f"Retrieval failed: {e}", details={"question": question[:100]}) from e

        if cache_backend and settings.cache_enabled:
            cache_backend.set(key, docs, ttl_seconds=settings.cache_ttl_seconds)

        logger.debug("Retrieved %d docs for question", len(docs))
        return {"retrieved_docs": docs}

    return retrieve_node


def retrieve_node(state: AgentState) -> dict[str, Any]:
    """Retrieve node (uses default vector store when no DI)."""
    return create_retrieve_node()(state)


def generate_node(state: AgentState) -> dict[str, Any]:
    """
    Generate grounded answer. Must cite sources using [N].
    Uses retry and optional caching.
    """
    question = state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            "answer": "I couldn't find relevant documentation for your question. "
            "Please try rephrasing or ensure the knowledge base has been populated.",
        }

    context = format_documents_for_context(retrieved_docs)
    llm = get_llm_with_retry()
    inputs = {"context": context, "question": question}
    answer = invoke_cached(llm, GENERATE_PROMPT, inputs, cache_prefix="generate")
    if _is_abstain_answer(answer):
        answer = "No sufficient data"
    return {"answer": answer}


def reflection_node(state: AgentState) -> dict[str, Any]:
    """
    Check for hallucination. If hallucinated, mark invalid.
    """
    answer = state.get("answer", "")
    retrieved_docs = state.get("retrieved_docs", [])

    validation_issues: list[str] = []
    is_valid = True

    # No docs: answer should be fallback message
    if not retrieved_docs:
        if "couldn't find" in answer.lower():
            return {"reflection": "No docs; fallback message", "is_valid": True}
        validation_issues.append("Answer generated without retrieved context")
        return {
            "reflection": "; ".join(validation_issues),
            "is_valid": False,
            "validation_issues": validation_issues,
        }

    context = format_documents_for_context(retrieved_docs)
    max_idx = len(retrieved_docs)

    # Citation check: factual claims must have [N]
    citation_refs = re.findall(r"\[(\d+)\]", answer)
    if answer and not citation_refs:
        if not _is_abstain_answer(answer):
            validation_issues.append("Response contains factual claims but no citation markers [N]")
            is_valid = False

    for ref in citation_refs:
        try:
            n = int(ref)
            if n < 1 or n > max_idx:
                validation_issues.append(
                    f"Citation [{n}] references non-existent source (max {max_idx})"
                )
                is_valid = False
        except ValueError:
            validation_issues.append(f"Invalid citation format: [{ref}]")

    # LLM-based hallucination check (structured output when supported)
    if is_valid and answer:
        llm = get_llm_with_retry()
        inputs = {"context": context, "answer": answer}
        try:
            structured_llm = llm.with_structured_output(ReflectionVerdict)
            chain = REFLECTION_STRUCTURED_PROMPT | structured_llm
            verdict_obj = chain.invoke(inputs)
            if not verdict_obj.grounded:
                validation_issues.append(
                    verdict_obj.reason or "Answer may contain information not in context"
                )
                is_valid = False
        except Exception:
            # Fallback to non-structured
            result = invoke_with_retry(llm, REFLECTION_PROMPT, inputs)
            verdict = (result or "").strip().upper()[:10]
            if "NO" in verdict:
                validation_issues.append("Answer may contain information not in context")
                is_valid = False

    reflection = "Validation passed" if is_valid else "; ".join(validation_issues)
    return {
        "reflection": reflection,
        "is_valid": is_valid,
        "validation_issues": validation_issues,
    }


def confidence_node(state: AgentState) -> dict[str, Any]:
    """
    Assign confidence score based on reflection.
    """
    is_valid = state.get("is_valid", True)
    validation_issues = state.get("validation_issues", [])
    retrieved_docs = state.get("retrieved_docs", [])

    settings = get_settings()

    # Base score from validity (from config)
    if not is_valid:
        confidence = settings.confidence_invalid
    elif not retrieved_docs:
        confidence = settings.confidence_no_docs
    else:
        confidence = settings.confidence_valid

    # Reduce for each validation issue
    for _ in validation_issues:
        confidence = max(0.0, confidence - settings.confidence_penalty_per_issue)

    return {"confidence_score": round(min(1.0, confidence), 2)}
