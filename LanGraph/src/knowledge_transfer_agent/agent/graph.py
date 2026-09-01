"""
LangGraph workflow: retrieve -> generate -> reflection -> confidence.
"""

from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from knowledge_transfer_agent.agent.nodes import (
    confidence_node,
    create_retrieve_node,
    generate_node,
    reflection_node,
)
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


def create_knowledge_agent_graph(
    vector_store: Any = None,
    checkpointer=None,
):
    """
    Create the LangGraph workflow for knowledge transfer.

    Flow: retrieve -> generate -> reflection -> confidence -> END

    Args:
        vector_store: Optional vector store for DI (uses default when None)
        checkpointer: Optional checkpointer for conversation memory

    Returns:
        Compiled StateGraph
    """
    builder = StateGraph(AgentState)

    retrieve_node = create_retrieve_node(vector_store=vector_store)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("confidence", confidence_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "reflection")
    builder.add_edge("reflection", "confidence")
    builder.add_edge("confidence", END)

    memory = checkpointer or MemorySaver()
    return builder.compile(checkpointer=memory)
