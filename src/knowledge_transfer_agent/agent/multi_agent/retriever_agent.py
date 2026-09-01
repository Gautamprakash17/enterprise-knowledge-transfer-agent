"""
Retriever agent: plan (once) + hybrid multi-hop retrieve + optional compress.
"""

from __future__ import annotations

from typing import Any

from knowledge_transfer_agent.agent.multi_agent.trace import stamp_agent
from knowledge_transfer_agent.agent.nodes import (
    compress_context_node,
    create_retrieve_node,
    planner_node,
)
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.cache import get_cache


def create_retriever_agent(vector_store: Any = None, cache: Any = None):
    """Factory: Retriever agent node bound to a vector store."""

    settings = get_settings()
    cache_backend = cache if cache is not None else (
        get_cache() if settings.cache_enabled else None
    )
    retrieve_node = create_retrieve_node(vector_store=vector_store, cache=cache_backend)

    def retriever_agent(state: AgentState) -> dict[str, Any]:
        updates: dict[str, Any] = {}

        # Plan once; on critic-driven retry keep existing sub_queries / retry counters.
        if not state.get("sub_queries"):
            updates.update(planner_node(state))
            # Merge into a working view for retrieve
            working: AgentState = {**state, **updates}  # type: ignore[misc]
        else:
            working = state

        updates.update(retrieve_node(working))
        working = {**working, **updates}  # type: ignore[misc]
        updates.update(compress_context_node(working))

        return stamp_agent(state, "retriever", updates)

    return retriever_agent
