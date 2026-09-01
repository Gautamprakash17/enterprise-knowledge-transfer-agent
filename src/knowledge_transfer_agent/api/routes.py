"""
API route definitions.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from knowledge_transfer_agent.core.exceptions import AgentError
from knowledge_transfer_agent.api.dependencies import (
    check_vector_store,
    get_agent_service,
    get_current_user,
    get_workspace_id_header,
    resolve_workspace_id,
)
from knowledge_transfer_agent.ingestion.document_loader import SUPPORTED_EXTENSIONS
from knowledge_transfer_agent.api.schemas import (
    AskRequest,
    AskResponse,
    CitationSchema,
    DocumentsListResponse,
    FeedbackRequest,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
    IndexResetRequest,
    IndexResetResponse,
    IngestJobStartedResponse,
    LocalPathIngestRequest,
    GitCloneIngestRequest,
    WorkspaceCreateRequest,
    WorkspaceSchema,
)
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.exceptions import VectorStoreError
from knowledge_transfer_agent.core.database import delete_workspace_data
from knowledge_transfer_agent.core.workspaces import (
    clear_all_workspace_indices,
    clear_workspace_index,
    create_workspace,
    delete_workspace,
    list_workspaces,
)
from knowledge_transfer_agent.retrieval.vector_store import get_vector_store, list_index_sources
from knowledge_transfer_agent.services.agent_service import AgentService
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

_WEB_UPLOAD_ROOT = Path(__file__).resolve().parents[3] / "data" / "web_uploads"
_ALLOWED_UPLOAD_SUFFIXES = SUPPORTED_EXTENSIONS
_SKIP_UPLOAD_DIR_NAMES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "target",
    ".next",
    ".nuxt",
    "coverage",
    ".eggs",
}


def _should_skip_upload_relpath(rel: str) -> bool:
    parts = Path(rel.replace("\\", "/")).parts
    lowered = {p.lower() for p in parts}
    if lowered & _SKIP_UPLOAD_DIR_NAMES:
        return True
    if any(p.endswith(".egg-info") for p in parts):
        return True
    return False


def _safe_upload_relative_path(name: str) -> Path | None:
    """Preserve folder structure from browser folder upload (webkitRelativePath)."""
    normalized = name.replace("\\", "/").strip().lstrip("/")
    if not normalized or ".." in normalized.split("/"):
        return None
    if _should_skip_upload_relpath(normalized):
        return None
    parts: list[str] = []
    for segment in normalized.split("/"):
        if not segment or segment in (".", ".."):
            continue
        safe = re.sub(r"[^\w.\-]", "_", segment)[:160]
        if safe:
            parts.append(safe)
    if not parts:
        return None
    rel = Path(*parts)
    if rel.suffix.lower() not in _ALLOWED_UPLOAD_SUFFIXES:
        return None
    return rel


def _scoped_thread_id(workspace_id: str, thread_id: str | None) -> str:
    return f"{workspace_id}:{thread_id or 'default'}"


def _citation_schemas(citations: list[dict]) -> list[CitationSchema]:
    return [
        CitationSchema(
            source=c.get("source", ""),
            source_type=c.get("source_type", "generic"),
            doc_id=str(c.get("doc_id", "")),
            page_number=c.get("page_number"),
            snippet=c.get("snippet"),
        )
        for c in citations
    ]


def _agent_for_workspace(workspace_id: str) -> AgentService:
    try:
        store = get_vector_store(workspace_id=workspace_id)
    except FileNotFoundError as e:
        raise VectorStoreError(
            f"No index for project '{workspace_id}'. Add documents to this project first.",
            details={"workspace_id": workspace_id},
        ) from e
    return AgentService(vector_store=store)


@router.get("/meta")
async def api_meta() -> dict[str, Any]:
    """Lightweight capability probe for the web UI (detect stale API servers)."""
    from knowledge_transfer_agent import __version__

    return {
        "version": __version__,
        "features": {
            "workspaces": True,
            "ingest_upload": True,
            "ask_stream": True,
            "index_reset": True,
            "chat_threads": True,
            "ingest_jobs": True,
            "ingest_folder": True,
            "ingest_local_path": get_settings().allow_local_path_ingest,
            "ingest_git_clone": get_settings().allow_git_clone_ingest,
            "followups": True,
            "audit_log": True,
            "metrics": get_settings().metrics_enabled,
            "shared_memory": get_settings().shared_memory_enabled,
            "guardrails": get_settings().guardrails_enabled,
            "document_library": True,
        },
    }


@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents(
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
    workspace_id: str | None = None,
) -> DocumentsListResponse:
    """List unique indexed sources for the active project (document library)."""
    ws = resolve_workspace_id(workspace_id, header_workspace)
    try:
        data = await run_in_threadpool(list_index_sources, workspace_id=ws)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        ) from e
    return DocumentsListResponse(**data)


@router.get("/workspaces", response_model=list[WorkspaceSchema])
async def get_workspaces(
    _: Annotated[str, Depends(get_current_user)],
) -> list[WorkspaceSchema]:
    """List project workspaces (each has its own index and chat scope)."""
    return [WorkspaceSchema(**ws) for ws in list_workspaces()]


@router.post("/index/reset", response_model=IndexResetResponse)
async def reset_index(
    request: IndexResetRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> IndexResetResponse:
    """
    Delete FAISS index + manifest for the current project (or all projects).
    Uploaded files on disk are kept; only search index is removed.
    """
    try:
        if request.all_workspaces:
            cleared = await run_in_threadpool(clear_all_workspace_indices)
            return IndexResetResponse(
                success=True,
                message=f"Cleared indexed data for {len(cleared)} project(s).",
                cleared=cleared,
            )
        ws = resolve_workspace_id(request.workspace_id, header_workspace)
        result = await run_in_threadpool(clear_workspace_index, ws)
        removed = result.get("removed") or []
        if not removed:
            return IndexResetResponse(
                success=True,
                message=f"No index found for project '{ws}' (already empty).",
                cleared=[result],
            )
        return IndexResetResponse(
            success=True,
            message=f"Cleared index for project '{ws}' ({', '.join(removed)}).",
            cleared=[result],
        )
    except Exception as e:
        logger.exception("Index reset failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/workspaces", response_model=WorkspaceSchema)
async def post_workspace(
    request: WorkspaceCreateRequest,
    _: Annotated[str, Depends(get_current_user)],
) -> WorkspaceSchema:
    """Create a new isolated project workspace."""
    ws = create_workspace(request.name)
    return WorkspaceSchema(**ws)


@router.delete("/workspaces/{workspace_id}")
async def remove_workspace(
    workspace_id: str,
    _: Annotated[str, Depends(get_current_user)],
) -> dict[str, Any]:
    """Delete a project (not allowed for the default project)."""
    try:
        result = await run_in_threadpool(delete_workspace, workspace_id)
        await run_in_threadpool(delete_workspace_data, result["workspace_id"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        logger.exception("Workspace delete failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> AskResponse:
    """
    Submit a question and get structured response with answer, citations,
    reflection status, and confidence score.
    """
    workspace_id = resolve_workspace_id(request.workspace_id, header_workspace)
    agent_service = _agent_for_workspace(workspace_id)
    thread_id = _scoped_thread_id(workspace_id, "ask")

    try:
        result = await run_in_threadpool(
            agent_service.ask,
            question=request.question,
            thread_id=thread_id,
            workspace_id=workspace_id,
        )
    except (AgentError, VectorStoreError):
        raise
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    followups: list[str] = []
    if request.include_followups:
        from knowledge_transfer_agent.services.followups import suggest_followups

        followups = await run_in_threadpool(
            suggest_followups, request.question, result["answer"]
        )

    return AskResponse(
        answer=result["answer"],
        citations=_citation_schemas(result["citations"]),
        reflection_status=result["reflection_status"],
        confidence_score=result["confidence_score"],
        follow_up_suggestions=followups,
        agent_trace=list(result.get("agent_trace") or []),
        guardrail_flags=list(result.get("guardrail_flags") or []),
        guardrails_blocked=bool(result.get("guardrails_blocked")),
        shared_memories_used=int(result.get("shared_memories_used") or 0),
        shared_memory_saved_id=result.get("shared_memory_saved_id"),
        agent_timings_ms=dict(result.get("agent_timings_ms") or {}),
        latency_ms=result.get("latency_ms"),
    )


@router.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> StreamingResponse:
    """
    Stream the answer as Server-Sent Events (SSE) for lower perceived latency.

    Emits lines like:
      data: {"type": "token", "text": "..."}
      data: {"type": "done", "citations": [...]}
    """

    workspace_id = resolve_workspace_id(request.workspace_id, header_workspace)
    try:
        agent_service = _agent_for_workspace(workspace_id)
    except VectorStoreError as e:
        raise HTTPException(status_code=status.HTTP_503_INTERNAL_SERVER_ERROR, detail=e.message) from e

    thread_id = _scoped_thread_id(workspace_id, "ask-stream")

    def event_gen():
        try:
            for event in agent_service.stream_ask(
                question=request.question,
                thread_id=thread_id,
                workspace_id=workspace_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Stream endpoint failed")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    vector_store_loaded: Annotated[bool, Depends(check_vector_store)],
) -> HealthResponse:
    """Health check endpoint for liveness/readiness probes."""
    from knowledge_transfer_agent import __version__
    return HealthResponse(
        status="ok" if vector_store_loaded else "degraded",
        version=__version__,
        vector_store_loaded=vector_store_loaded,
    )


@router.get("/metrics")
async def api_metrics():
    """Prometheus metrics (same payload as GET /metrics)."""
    from fastapi.responses import Response

    from knowledge_transfer_agent.core.metrics import render_metrics
    from knowledge_transfer_agent.config import get_settings as _gs

    if not _gs().metrics_enabled:
        return Response(content=b"# metrics disabled\n", media_type="text/plain")
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> QueryResponse:
    """
    Submit a question to the knowledge transfer agent.
    Returns a citation-backed response.
    """
    workspace_id = resolve_workspace_id(request.workspace_id, header_workspace)
    agent_service = _agent_for_workspace(workspace_id)
    thread_id = _scoped_thread_id(workspace_id, request.thread_id or str(uuid.uuid4()))
    try:
        result = await run_in_threadpool(
            agent_service.ask,
            question=request.query,
            thread_id=thread_id,
            workspace_id=workspace_id,
        )
    except (AgentError, VectorStoreError):
        raise
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    return QueryResponse(
        response=result["answer"],
        citations=_citation_schemas(result["citations"]),
        confidence_score=result["confidence_score"],
        thread_id=thread_id,
    )


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> dict[str, str]:
    """Submit feedback (persisted to SQLite when enabled)."""
    from knowledge_transfer_agent.core.database import insert_feedback

    ws = resolve_workspace_id(request.workspace_id, header_workspace)
    await run_in_threadpool(
        insert_feedback,
        workspace_id=ws,
        thread_id=request.thread_id,
        query=request.query,
        was_helpful=request.was_helpful,
        feedback_text=request.feedback_text,
    )
    logger.info("Feedback received: helpful=%s", request.was_helpful)
    return {"status": "ok", "message": "Feedback recorded"}


@router.post("/ingest/upload", response_model=IngestJobStartedResponse)
async def ingest_uploaded_files(
    files: Annotated[
        list[UploadFile],
        File(description="Docs and source files (folder upload supported)"),
    ],
    replace_index: Annotated[bool, Form()] = False,
    workspace_id: Annotated[str | None, Form()] = None,
    upload_batch_id: Annotated[str | None, Form()] = None,
    start_ingest: Annotated[bool, Form()] = True,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)] = "default",
    _: Annotated[str, Depends(get_current_user)] = "default",
) -> IngestJobStartedResponse:
    """
    Upload files or folder chunks. Use upload_batch_id + start_ingest=false for multi-part
    folder uploads, then a final chunk with start_ingest=true to queue indexing.
    """
    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    max_files = settings.max_upload_files_per_batch
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Select at least one file to upload",
        )

    ws = resolve_workspace_id(workspace_id, header_workspace)
    batch_token = (upload_batch_id or "").strip() or f"batch_{uuid.uuid4().hex[:12]}"
    if not re.match(r"^batch_[a-z0-9]{8,24}$", batch_token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload_batch_id",
        )
    batch_dir = _WEB_UPLOAD_ROOT / ws / batch_token
    batch_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    skipped = 0

    for upload in files:
        if not upload.filename:
            skipped += 1
            continue
        rel = _safe_upload_relative_path(upload.filename)
        if rel is None:
            skipped += 1
            continue
        dest = batch_dir / rel
        content = await upload.read()
        if not content:
            skipped += 1
            continue
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File {rel} exceeds {settings.max_upload_size_mb} MB limit",
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        saved_count += 1

    existing = sum(1 for _ in batch_dir.rglob("*") if _.is_file())
    if existing > max_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch exceeds {max_files} files limit",
        )

    if saved_count == 0 and not start_ingest:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No supported files in this chunk (check extensions and skipped folders)",
        )

    if not start_ingest:
        return IngestJobStartedResponse(
            job_id=None,
            status="uploading",
            batch_id=batch_token,
            files_received=existing,
            message=f"Received {saved_count} file(s) in this chunk ({existing} total). Send final chunk with start_ingest=true.",
        )

    if existing == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files in batch to index",
        )

    from knowledge_transfer_agent.services.ingest_jobs import submit_file_ingest_job

    try:
        job = await run_in_threadpool(
            submit_file_ingest_job, ws, batch_dir, replace_index=replace_index
        )
        msg = f"Queued indexing for {existing} file(s)"
        if skipped:
            msg += f" ({skipped} skipped in last chunk)"
        return IngestJobStartedResponse(
            job_id=job["id"],
            status=job.get("status", "pending"),
            batch_id=batch_token,
            files_received=existing,
            message=f"{msg}. Poll /api/v1/ingest/jobs/{job['id']}",
        )
    except Exception as e:
        logger.exception("Upload ingestion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/ingest/local-path", response_model=IngestJobStartedResponse)
async def ingest_local_directory(
    request: LocalPathIngestRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> IngestJobStartedResponse:
    """
    Index a directory on the machine running the API (local dev).
    Enable with ALLOW_LOCAL_PATH_INGEST=true in .env.
    """
    settings = get_settings()
    if not settings.allow_local_path_ingest:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local path ingest is disabled. Set ALLOW_LOCAL_PATH_INGEST=true in .env and restart the API.",
        )

    ws = resolve_workspace_id(request.workspace_id, header_workspace)
    root = Path(request.path).expanduser().resolve()
    if not root.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path not found: {root}",
        )
    if not root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path must be a directory",
        )

    from knowledge_transfer_agent.services.ingest_jobs import submit_path_ingest_job

    try:
        job = await run_in_threadpool(
            submit_path_ingest_job,
            ws,
            root,
            replace_index=request.replace_index,
        )
        return IngestJobStartedResponse(
            job_id=job["id"],
            status=job.get("status", "pending"),
            message=f"Queued codebase ingest for {root}. Poll /api/v1/ingest/jobs/{job['id']}",
        )
    except Exception as e:
        logger.exception("Local path ingest failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/ingest/git-clone", response_model=IngestJobStartedResponse)
async def ingest_git_clone(
    request: GitCloneIngestRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> IngestJobStartedResponse:
    """
    Clone a remote Git repository on the API server, then index it into the workspace.
    Requires git on the server PATH. Private GitHub HTTPS repos use GITHUB_TOKEN in .env.
    """
    from knowledge_transfer_agent.ingestion.github import _is_remote_git_url

    settings = get_settings()
    if not settings.allow_git_clone_ingest:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Git clone ingest is disabled. Set ALLOW_GIT_CLONE_INGEST=true in .env and restart the API.",
        )

    repo_url = request.repo_url.strip()
    if not _is_remote_git_url(repo_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="repo_url must be a remote Git URL (https://, http://, git@, or ssh://)",
        )

    ws = resolve_workspace_id(request.workspace_id, header_workspace)
    from knowledge_transfer_agent.services.ingest_jobs import submit_git_clone_ingest_job

    try:
        job = await run_in_threadpool(
            submit_git_clone_ingest_job,
            ws,
            repo_url,
            replace_index=request.replace_index,
            branch=request.branch.strip() if request.branch else None,
        )
        short = repo_url.rstrip("/").split("/")[-1].replace(".git", "") or "repository"
        return IngestJobStartedResponse(
            job_id=job["id"],
            status=job.get("status", "pending"),
            message=f"Cloning {short}. Poll /api/v1/ingest/jobs/{job['id']}",
        )
    except Exception as e:
        logger.exception("Git clone ingest failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/ingest", response_model=IngestionResponse)
async def run_ingestion(
    request: IngestionRequest,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> IngestionResponse:
    """Run the ingestion pipeline on optional paths (scoped to one project)."""
    from knowledge_transfer_agent.ingestion.pipeline import IngestionPipeline

    ws = resolve_workspace_id(request.workspace_id, header_workspace)
    pipeline = IngestionPipeline()
    paths = [Path(p) for p in request.paths] if request.paths else []
    try:
        results = pipeline.run(
            additional_paths=paths or None,
            persist=True,
            replace_index=request.replace_index,
            workspace_id=ws,
            include_configured_sources=request.include_configured_sources,
        )
        success = any(r.success for r in results.values())
        return IngestionResponse(
            success=success,
            results={
                k: {
                    "success": v.success,
                    "documents_processed": v.documents_processed,
                    "documents_failed": v.documents_failed,
                    "errors": v.errors[:5],
                }
                for k, v in results.items()
            },
            message="Ingestion completed" if success else "Ingestion had errors",
        )
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
