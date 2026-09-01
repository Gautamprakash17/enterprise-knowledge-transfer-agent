"""
API route definitions.
"""

import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from langgraph.checkpoint.memory import MemorySaver

from knowledge_transfer_agent.core.exceptions import AgentError
from knowledge_transfer_agent.api.dependencies import (
    check_vector_store,
    get_agent_service,
    get_current_user,
)
from knowledge_transfer_agent.api.schemas import (
    AskRequest,
    AskResponse,
    CitationSchema,
    FeedbackRequest,
    HealthResponse,
    IngestionRequest,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# In-memory thread store for demo; use Redis/DB in production
_thread_store: dict[str, MemorySaver] = {}
_feedback_log: list[dict[str, Any]] = []


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    agent_service: Annotated[Any, Depends(get_agent_service)],
    _: Annotated[str, Depends(get_current_user)],
) -> AskResponse:
    """
    Submit a question and get structured response with answer, citations,
    reflection status, and confidence score.
    """
    try:
        result = agent_service.ask(question=request.question, thread_id="ask")
    except AgentError:
        raise
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    citations = [
        CitationSchema(
            source=c.get("source", ""),
            source_type=c.get("source_type", "generic"),
            doc_id=str(c.get("doc_id", "")),
            page_number=c.get("page_number"),
        )
        for c in result["citations"]
    ]
    return AskResponse(
        answer=result["answer"],
        citations=citations,
        reflection_status=result["reflection_status"],
        confidence_score=result["confidence_score"],
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


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    agent_service: Annotated[Any, Depends(get_agent_service)],
    _: Annotated[str, Depends(get_current_user)],
) -> QueryResponse:
    """
    Submit a question to the knowledge transfer agent.
    Returns a citation-backed response.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        result = agent_service.ask(question=request.query, thread_id=thread_id)
    except AgentError:
        raise
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e

    citations = [
        CitationSchema(
            source=c.get("source", ""),
            source_type=c.get("source_type", "generic"),
            doc_id=str(c.get("doc_id", "")),
            page_number=c.get("page_number"),
        )
        for c in result["citations"]
    ]
    return QueryResponse(
        response=result["answer"],
        citations=citations,
        confidence_score=result["confidence_score"],
        thread_id=thread_id,
    )


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    _: Annotated[str, Depends(get_current_user)],
) -> dict[str, str]:
    """Submit feedback for the query logging / improvement loop."""
    _feedback_log.append({
        "thread_id": request.thread_id,
        "query": request.query,
        "response_id": request.response_id,
        "was_helpful": request.was_helpful,
        "feedback_text": request.feedback_text,
    })
    logger.info("Feedback received: helpful=%s", request.was_helpful)
    return {"status": "ok", "message": "Feedback recorded"}


@router.post("/ingest", response_model=IngestionResponse)
async def run_ingestion(
    request: IngestionRequest,
    _: Annotated[str, Depends(get_current_user)],
) -> IngestionResponse:
    """Run the ingestion pipeline on optional paths."""
    from knowledge_transfer_agent.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    paths = [Path(p) for p in request.paths] if request.paths else []
    try:
        results = pipeline.run(
            additional_paths=paths or None,
            persist=True,
            replace_index=request.replace_index,
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
