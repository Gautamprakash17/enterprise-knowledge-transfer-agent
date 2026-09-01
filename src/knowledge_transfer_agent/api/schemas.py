"""
Pydantic schemas for API request/response validation.
"""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for knowledge query."""

    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, description="Optional thread ID for conversation")
    workspace_id: str | None = Field(default=None, description="Project id for scoped retrieval")


class IndexResetRequest(BaseModel):
    """Reset (clear) vector index for a project or all projects."""

    workspace_id: str | None = Field(
        default=None,
        description="Project to clear; ignored when all_workspaces is true",
    )
    all_workspaces: bool = Field(
        default=False,
        description="If true, clear every project's FAISS index",
    )


class IndexResetResponse(BaseModel):
    """Result of index reset."""

    success: bool
    message: str
    cleared: list[dict[str, Any]] = Field(default_factory=list)


class WorkspaceCreateRequest(BaseModel):
    """Create an isolated project workspace."""

    name: str = Field(..., min_length=1, max_length=120)


class WorkspaceSchema(BaseModel):
    """Workspace (project) metadata."""

    id: str
    name: str
    created_at: str


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(..., min_length=1, max_length=2000)
    workspace_id: str | None = Field(
        default=None,
        description="Project id — search only this workspace's indexed documents",
    )
    include_followups: bool = Field(
        default=False,
        description="If true, include suggested follow-up questions in the response",
    )


class CitationSchema(BaseModel):
    """Citation metadata."""

    source: str
    source_type: str
    doc_id: str
    page_number: int | None = Field(
        default=None,
        description="1-based page index when the chunk came from a PDF page",
    )
    snippet: str | None = Field(default=None, description="Short excerpt from the source chunk")


class IngestJobStartedResponse(BaseModel):
    """Async ingest job queued."""

    job_id: str | None = None
    status: str
    message: str
    batch_id: str | None = None
    files_received: int | None = None


class LocalPathIngestRequest(BaseModel):
    """Ingest a directory already on the API server (local dev)."""

    path: str = Field(..., min_length=1, description="Absolute or ~ path to project root")
    workspace_id: str | None = None
    replace_index: bool = False


class GitCloneIngestRequest(BaseModel):
    """Clone a remote Git repository on the API server, then index it."""

    repo_url: str = Field(
        ...,
        min_length=8,
        description="Remote Git URL (https://, http://, git@, or ssh://)",
    )
    branch: str | None = Field(
        default=None,
        max_length=120,
        description="Optional branch name (overrides GITHUB_CLONE_BRANCH from env)",
    )
    workspace_id: str | None = None
    replace_index: bool = False


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
    follow_up_suggestions: list[str] = Field(default_factory=list)
    agent_trace: list[str] = Field(
        default_factory=list,
        description="Ordered multi-agent visits (guardrails, …, output_guardrails, shared_memory_save)",
    )
    guardrail_flags: list[str] = Field(
        default_factory=list,
        description="Guardrail events (e.g. keyword:exfiltrate, pii_email, output:pii_email, injection:...)",
    )
    guardrails_blocked: bool = Field(
        default=False,
        description="True when input was blocked by safety guardrails",
    )
    shared_memories_used: int = Field(
        default=0,
        description="How many shared-memory entries were loaded for this answer",
    )
    shared_memory_saved_id: str | None = Field(
        default=None,
        description="Id of episodic memory saved after this answer, if any",
    )
    agent_timings_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Per-agent wall time in milliseconds for this request",
    )
    latency_ms: float | None = Field(
        default=None,
        description="End-to-end /ask latency in milliseconds",
    )


class FeedbackRequest(BaseModel):
    """Feedback for query logging / improvement loop."""

    thread_id: str
    query: str
    response_id: str | None = None
    was_helpful: bool
    feedback_text: str | None = None
    workspace_id: str | None = None


class IngestionRequest(BaseModel):
    """Request to run ingestion on paths."""

    paths: list[str] = Field(default_factory=list)
    workspace_id: str | None = Field(default=None, description="Target project workspace")
    include_configured_sources: bool = Field(
        default=False,
        description="Also ingest Confluence/GitHub from .env into this workspace",
    )
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


class IndexedSourceSchema(BaseModel):
    """One unique source in the project FAISS index."""

    source: str
    source_type: str
    file_name: str
    chunk_count: int
    doc_ids: list[str] = Field(default_factory=list)


class DocumentsListResponse(BaseModel):
    """Indexed sources for a workspace (document library)."""

    workspace_id: str
    total_chunks: int
    source_count: int
    sources: list[IndexedSourceSchema] = Field(default_factory=list)
