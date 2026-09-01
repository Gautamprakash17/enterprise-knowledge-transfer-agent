"""
Pydantic schemas for API request/response validation.
"""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for knowledge query."""

    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, description="Optional thread ID for conversation")


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, max_length=2000)


class CitationSchema(BaseModel):
    """Citation metadata."""

    source: str
    source_type: str
    doc_id: str
    page_number: int | None = Field(
        default=None,
        description="1-based page index when the chunk came from a PDF page",
    )


class QueryResponse(BaseModel):
    """Response from knowledge query."""

    response: str
    citations: list[CitationSchema] = Field(default_factory=list)
    confidence_score: float | None = Field(default=None, description="Agent confidence 0-1")
    thread_id: str | None = None


class AskResponse(BaseModel):
    """Structured response from POST /ask."""

    answer: str
    citations: list[CitationSchema] = Field(default_factory=list)
    reflection_status: str
    confidence_score: float


class FeedbackRequest(BaseModel):
    """Feedback for query logging / improvement loop."""

    thread_id: str
    query: str
    response_id: str | None = None
    was_helpful: bool
    feedback_text: str | None = None


class IngestionRequest(BaseModel):
    """Request to run ingestion on paths."""

    paths: list[str] = Field(default_factory=list)
    replace_index: bool = Field(
        default=False,
        description=(
            "If true, rebuild the FAISS index from scratch from this run only (no merge with old vectors). "
            "Use after enabling per-page PDF metadata or to drop stale chunks with missing page_number."
        ),
    )


class IngestionResponse(BaseModel):
    """Result of ingestion run."""

    success: bool
    results: dict[str, Any]
    message: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    vector_store_loaded: bool
