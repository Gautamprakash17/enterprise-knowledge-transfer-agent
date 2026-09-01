"""
LangGraph nodes: retrieve, generate, reflection, confidence.
Clear separation of concerns. Uses retry, caching, structured output.
"""

import re
from typing import Any

from knowledge_transfer_agent.agent.context_formatter import format_documents_for_context
from knowledge_transfer_agent.agent.llm_client import get_llm_with_retry, invoke_cached, invoke_with_retry
from knowledge_transfer_agent.agent.output_schemas import PlannerOutput, ReflectionVerdict, ToolSelection
from knowledge_transfer_agent.agent.prompts import (
    CONTEXT_COMPRESS_PROMPT,
    GENERATE_PROMPT,
    PLANNER_PROMPT,
    REFLECTION_PROMPT,
    REFLECTION_STRUCTURED_PROMPT,
    TOOL_SELECTION_PROMPT,
)
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.cache import cache_key, get_cache
from knowledge_transfer_agent.core.exceptions import RetrievalError
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.retrieval.reranker import maybe_rerank
from knowledge_transfer_agent.retrieval.retriever_factory import get_hybrid_retriever

logger = get_logger(__name__)

_ABSTAIN_EXACT = "no sufficient data"


def _is_abstain_answer(answer: str) -> bool:
    """True if the model abstains (no grounded answer). Allows trailing punctuation / quotes."""
    if not answer or not answer.strip():
        return False
    t = answer.strip().lower().rstrip('.!?"\'').strip()
    return t == _ABSTAIN_EXACT


def _format_history(history: list[str] | None, max_items: int = 4) -> str:
    if not history:
        return ""
    trimmed = history[-max_items:]
    return "\n".join(f"- {h}" for h in trimmed)


def _split_multi_topic_question(question: str, *, max_topics: int = 4) -> list[str]:
    """
    Deterministic fallback for long multi-topic user inputs.

    If the prompt is long and/or formatted like a list, extract up to `max_topics`
    focused items to use as sub-queries for retrieval.
    """
    q = (question or "").strip()
    if not q:
        return []

    looks_like_list = any(
        re.match(r"^\s*(?:[-*]|(?:\d+[\).])|•)\s+\S+", line)
        for line in q.splitlines()
        if line.strip()
    )
    if len(q) < 800 and not looks_like_list:
        return []

    topics: list[str] = []

    # Prefer explicit bullet/numbered items: they're usually separate topics.
    for line in q.splitlines():
        if len(topics) >= max_topics:
            break
        m = re.match(r"^\s*(?:[-*]|(?:\d+[\).])|•)\s+(.*)$", line.strip())
        if not m:
            continue
        item = (m.group(1) or "").strip()
        if len(item) < 8:
            continue
        topics.append(item[:300].strip())

    # Fallback: split by question marks / coarse sentence boundaries.
    if not topics:
        parts = re.split(r"[?]+|\n{2,}|(?:\.\s+)", q)
        for p in parts:
            if len(topics) >= max_topics:
                break
            p = " ".join(p.split()).strip()
            if len(p) < 12:
                continue
            topics.append(p[:300])

    # De-dup (case-insensitive) while preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for t in topics:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out[:max_topics]


def planner_node(state: AgentState) -> dict[str, Any]:
    """Plan sub-queries for multi-hop retrieval."""
    question = state.get("question", "")
    if not question:
        return {"sub_queries": []}

    # Deterministic multi-topic split for very long/bulleted prompts.
    # This makes "case 2" (multi-topic prompts) more reliable.
    heuristic_subs = _split_multi_topic_question(question, max_topics=4)
    if len(heuristic_subs) >= 2:
        settings = get_settings()
        return {
            "sub_queries": heuristic_subs,
            "hop_index": 0,
            "max_hops": settings.max_hops,
            "reflection_retries": 0,
            "max_reflection_retries": settings.max_reflection_retries,
        }

    llm = get_llm_with_retry()
    try:
        structured_llm = llm.with_structured_output(PlannerOutput)
        chain = PLANNER_PROMPT | structured_llm
        plan = chain.invoke({"question": question})
        sub_queries = plan.sub_queries[:4]
    except Exception:
        text = invoke_with_retry(llm, PLANNER_PROMPT, {"question": question})
        sub_queries = [q.strip() for q in text.split("\n") if q.strip()][:4]

    if not sub_queries:
        sub_queries = [question]

    settings = get_settings()
    return {
        "sub_queries": sub_queries,
        "hop_index": 0,
        "max_hops": settings.max_hops,
        "reflection_retries": 0,
        "max_reflection_retries": settings.max_reflection_retries,
    }


def tool_selection_node(state: AgentState) -> dict[str, Any]:
    """Select tool dynamically based on plan."""
    question = state.get("question", "")
    sub_queries = state.get("sub_queries", [])
    llm = get_llm_with_retry()
    try:
        structured_llm = llm.with_structured_output(ToolSelection)
        chain = TOOL_SELECTION_PROMPT | structured_llm
        selection = chain.invoke({"question": question, "sub_queries": sub_queries})
        tool_name = selection.tool_name
    except Exception:
        tool_name = "retrieve"

    return {"tool_choice": tool_name}


def create_retrieve_node(vector_store=None, cache=None):
    """Factory: returns retrieve_node that uses the given vector_store (for DI)."""

    def _retrieve_one(query: str, top_k: int, cache_backend, settings) -> list:
        """Cache-aware single-query retrieval (hybrid + optional rerank)."""
        if cache_backend and settings.cache_enabled:
            key = cache_key(
                "retrieve",
                query,
                top_k,
                settings.rerank_enabled,
                settings.rerank_top_n,
                settings.rerank_model,
            )
            cached = cache_backend.get(key)
            if cached is not None:
                logger.debug("Retrieve cache hit")
                return cached

        retriever = get_hybrid_retriever(vector_store=vector_store)
        docs = retriever.invoke(query)[:top_k]
        docs = maybe_rerank(query, docs)

        if cache_backend and settings.cache_enabled:
            cache_backend.set(key, docs, ttl_seconds=settings.cache_ttl_seconds)
        return docs

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        """
        Retrieve top-k documents. Runs all sub-queries (multi-hop) concurrently
        in a single pass to reduce latency, then merges + dedupes.
        """
        question = state.get("question", "")
        if not question:
            return {"retrieved_docs": []}

        settings = get_settings()
        top_k = settings.top_k_semantic + settings.top_k_keyword
        cache_backend = cache or (get_cache() if settings.cache_enabled else None)

        sub_queries = state.get("sub_queries", [])
        max_hops = state.get("max_hops", settings.max_hops)
        queries = sub_queries[:max_hops] if sub_queries else [question]

        try:
            if len(queries) == 1:
                results = [_retrieve_one(queries[0], top_k, cache_backend, settings)]
            else:
                # Parallel multi-hop retrieval: fetch all sub-queries at once.
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as ex:
                    results = list(
                        ex.map(
                            lambda q: _retrieve_one(q, top_k, cache_backend, settings),
                            queries,
                        )
                    )
        except Exception as e:
            logger.exception("Retrieval failed: %s", e)
            raise RetrievalError(
                f"Retrieval failed: {e}",
                details={"question": question[:100], "queries": str(queries)[:100]},
            ) from e

        # Merge all hops + previous docs, dedupe by content.
        previous = state.get("retrieved_docs", [])
        combined: list = []
        seen: set = set()
        for bucket in [previous, *results]:
            for d in bucket:
                content = getattr(d, "page_content", "")
                if content and content not in seen:
                    seen.add(content)
                    combined.append(d)

        logger.debug("Retrieved %d docs across %d queries", len(combined), len(queries))
        # Mark all hops done so the graph moves on (no sequential loop).
        return {
            "retrieved_docs": combined,
            "hop_index": max_hops,
        }

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
    retrieved_docs = state.get("context_docs") or state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {
            "answer": "No sufficient data",
        }

    context = format_documents_for_context(retrieved_docs)
    history = _format_history(state.get("conversation_history"))
    shared_memory = (state.get("shared_memory_context") or "").strip() or "(none)"
    llm = get_llm_with_retry()
    inputs = {
        "context": context,
        "question": question,
        "history": history,
        "shared_memory": shared_memory,
    }
    answer = invoke_cached(llm, GENERATE_PROMPT, inputs, cache_prefix="generate")
    if _is_abstain_answer(answer):
        answer = "No sufficient data"
    return {"answer": answer}


def compress_context_node(state: AgentState) -> dict[str, Any]:
    """
    Optional: compress/summarize retrieved docs to reduce prompt size.
    Keeps the same number of sources so citation numbering [N] remains valid.
    """
    settings = get_settings()
    retrieved_docs = state.get("retrieved_docs", [])
    if not settings.context_compression_enabled or not retrieved_docs:
        return {"context_docs": retrieved_docs}

    # If chunks are already small enough, skip compression.
    max_chars = settings.context_compression_max_chars_per_chunk
    if max_chars > 0:
        too_big = any(len((d.page_content or "").strip()) > max_chars for d in retrieved_docs)
        if not too_big:
            return {"context_docs": retrieved_docs}

    context = format_documents_for_context(retrieved_docs)
    llm = get_llm_with_retry()
    compressed = invoke_cached(
        llm,
        CONTEXT_COMPRESS_PROMPT,
        {"context": context},
        cache_prefix="compress_context",
    )

    # Parse summaries: expect blocks like "[N] ...".
    summaries: dict[int, str] = {}
    current_n: int | None = None
    buf: list[str] = []
    for line in (compressed or "").splitlines():
        m = re.match(r"^\s*\[(\d+)\]\s*(.*)$", line)
        if m:
            if current_n is not None:
                summaries[current_n] = " ".join(" ".join(buf).split()).strip()
            current_n = int(m.group(1))
            buf = [m.group(2).strip()]
        else:
            if current_n is not None:
                buf.append(line.strip())
    if current_n is not None:
        summaries[current_n] = " ".join(" ".join(buf).split()).strip()

    # Build context_docs with same metadata, but compressed text (fallback to original if missing).
    context_docs = []
    for i, d in enumerate(retrieved_docs, 1):
        text = summaries.get(i) or (d.page_content or "")
        if max_chars > 0:
            text = text.strip()[:max_chars]
        context_docs.append(d.__class__(page_content=text, metadata=dict(d.metadata)))

    return {"context_docs": context_docs}


def reflection_node(state: AgentState) -> dict[str, Any]:
    """
    Check for hallucination. If hallucinated, mark invalid.
    """
    settings = get_settings()
    if not settings.reflection_enabled:
        return {
            "reflection": "Skipped (reflection disabled)",
            "is_valid": True,
            "validation_issues": [],
        }

    answer = state.get("answer", "")
    retrieved_docs = state.get("context_docs") or state.get("retrieved_docs", [])

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
        # Allow explicit abstain responses without citations (exact or with trailing . ! ?)
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
    reflection_retries = state.get("reflection_retries", 0)
    if not is_valid:
        reflection_retries += 1
    return {
        "reflection": reflection,
        "is_valid": is_valid,
        "validation_issues": validation_issues,
        "reflection_retries": reflection_retries,
    }


def memory_update_node(state: AgentState) -> dict[str, Any]:
    """Update conversation history in state."""
    question = state.get("question", "")
    answer = state.get("answer", "")
    history = list(state.get("conversation_history", []))
    if question:
        history.append(f"User: {question}")
    if answer:
        history.append(f"Assistant: {answer}")
    return {"conversation_history": history}


def confidence_node(state: AgentState) -> dict[str, Any]:
    """
    Assign confidence score based on reflection.
    """
    if state.get("guardrails_blocked"):
        return {"confidence_score": 0.0}

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
