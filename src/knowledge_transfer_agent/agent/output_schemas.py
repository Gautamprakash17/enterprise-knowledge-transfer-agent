"""
Pydantic schemas for structured LLM output parsing.
"""

from pydantic import BaseModel, Field


class ReflectionVerdict(BaseModel):
    """Structured output for hallucination check."""

    grounded: bool = Field(description="True if answer draws only from context")
    reason: str = Field(default="", description="Brief reason if not grounded")


class PlannerOutput(BaseModel):
    """Structured output for planner."""

    sub_queries: list[str] = Field(
        default_factory=list,
        description="Decomposed sub-queries for multi-hop retrieval",
    )


class ToolSelection(BaseModel):
    """Structured output for tool selection."""

    tool_name: str = Field(
        description="Selected tool name for next step, e.g. 'retrieve'",
    )


class CitationCheck(BaseModel):
    """Structured citation validation result."""

    has_citations: bool = Field(description="True if [N] markers present")
    valid_citations: bool = Field(description="True if all citations reference valid sources")
    issues: list[str] = Field(default_factory=list, description="Validation issues")
