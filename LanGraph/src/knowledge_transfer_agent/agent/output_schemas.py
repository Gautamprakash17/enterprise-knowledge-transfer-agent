"""
Pydantic schemas for structured LLM output parsing.
"""

from pydantic import BaseModel, Field


class ReflectionVerdict(BaseModel):
    """Structured output for hallucination check."""

    grounded: bool = Field(description="True if answer draws only from context")
    reason: str = Field(default="", description="Brief reason if not grounded")


class CitationCheck(BaseModel):
    """Structured citation validation result."""

    has_citations: bool = Field(description="True if [N] markers present")
    valid_citations: bool = Field(description="True if all citations reference valid sources")
    issues: list[str] = Field(default_factory=list, description="Validation issues")
