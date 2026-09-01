"""
TypedDict state for the LangGraph agent.
"""

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    """State schema for the knowledge transfer agent graph."""

    # Input
    question: str

    # Retrieval output
    retrieved_docs: list[Document]

    # Generation output
    answer: str

    # Reflection output
    reflection: str
    is_valid: bool
    validation_issues: list[str]

    # Confidence output
    confidence_score: float
