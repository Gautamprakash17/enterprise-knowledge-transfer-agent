"""
LangGraph multi-agent workflow:
Guardrails(in) → SharedMemory → Supervisor → Retriever → Writer → Critic
→ Memory → Confidence → Guardrails(out) → SaveMemory.
"""

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from knowledge_transfer_agent.agent.multi_agent import (
    create_retriever_agent,
    critic_agent,
    guardrails_agent,
    output_guardrails_agent,
    route_from_supervisor,
    supervisor_node,
    writer_agent,
)
from knowledge_transfer_agent.agent.multi_agent.shared_memory_agent import (
    shared_memory_load_agent,
    shared_memory_save_agent,
)
from knowledge_transfer_agent.agent.nodes import confidence_node, memory_update_node
from knowledge_transfer_agent.agent.state import AgentState
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.cache import get_cache
from knowledge_transfer_agent.core.metrics import timed_node
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


def create_knowledge_agent_graph(
    vector_store: Any = None,
    checkpointer=None,
):
    """
    Create the multi-agent LangGraph workflow for knowledge transfer.

    Agents:
      - guardrails: input safety (injection block, PII redact)
      - shared_memory: load workspace long-term memory
      - supervisor: routes to retriever / writer / critic / finish
      - retriever: plan + hybrid retrieve + compress
      - writer: grounded generation with citations
      - critic: reflection / groundedness check

    Then conversation memory → confidence → output_guardrails → shared_memory_save → END.

    Each node is wrapped with Prometheus timing (kta_agent_duration_seconds).
    """
    builder = StateGraph(AgentState)

    settings = get_settings()
    cache = get_cache() if settings.cache_enabled else None
    retriever = create_retriever_agent(vector_store=vector_store, cache=cache)

    builder.add_node("guardrails", timed_node("guardrails", guardrails_agent))
    builder.add_node("shared_memory", timed_node("shared_memory", shared_memory_load_agent))
    builder.add_node("supervisor", timed_node("supervisor", supervisor_node))
    builder.add_node("retriever", timed_node("retriever", retriever))
    builder.add_node("writer", timed_node("writer", writer_agent))
    builder.add_node("critic", timed_node("critic", critic_agent))
    builder.add_node("memory", timed_node("memory", memory_update_node))
    builder.add_node("confidence", timed_node("confidence", confidence_node))
    builder.add_node(
        "output_guardrails",
        timed_node("output_guardrails", output_guardrails_agent),
    )
    builder.add_node(
        "shared_memory_save",
        timed_node("shared_memory_save", shared_memory_save_agent),
    )

    builder.set_entry_point("guardrails")
    builder.add_edge("guardrails", "shared_memory")
    builder.add_edge("shared_memory", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "retriever": "retriever",
            "writer": "writer",
            "critic": "critic",
            "memory": "memory",
        },
    )

    builder.add_edge("retriever", "supervisor")
    builder.add_edge("writer", "supervisor")
    builder.add_edge("critic", "supervisor")

    builder.add_edge("memory", "confidence")
    builder.add_edge("confidence", "output_guardrails")
    builder.add_edge("output_guardrails", "shared_memory_save")
    builder.add_edge("shared_memory_save", END)

    memory = checkpointer or MemorySaver()
    logger.debug("Compiled multi-agent graph with metrics instrumentation")
    return builder.compile(checkpointer=memory)
