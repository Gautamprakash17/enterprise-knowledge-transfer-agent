"""
TypedDict state for the multi-agent LangGraph workflow.
"""

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """State schema for the knowledge transfer multi-agent graph."""

    # Input
    question: str
    conversation_history: list[str]

    # Multi-agent orchestration
    active_agent: str
    next_agent: str
    agent_trace: list[str]

    # Guardrails
    guardrails_blocked: bool
    guardrail_flags: list[str]

    # Shared long-term memory (workspace-scoped)
    workspace_id: str
    thread_id: str
    shared_memory_context: str
    shared_memories: list
    shared_memory_saved_id: str

    # Monitoring
    agent_timings_ms: dict

    # Planner output
    sub_queries: list[str]
    tool_choice: str
    hop_index: int
    max_hops: int
    reflection_retries: int
    max_reflection_retries: int

    # Retrieval output
    retrieved_docs: list[Document]
    context_docs: list[Document]

    # Generation output
    answer: str

    # Reflection output
    reflection: str
    is_valid: bool
    validation_issues: list[str]

    # Confidence output
    confidence_score: float
