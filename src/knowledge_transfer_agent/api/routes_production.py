"""Production endpoints: chat threads, audit, feedback DB, ingest jobs, follow-ups."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from knowledge_transfer_agent.api.dependencies import (
    get_current_user,
    get_workspace_id_header,
    resolve_workspace_id,
)
from knowledge_transfer_agent.core.database import (
    append_chat_message,
    create_chat_thread,
    delete_chat_thread,
    get_chat_thread,
    update_chat_thread_title,
    get_chat_messages,
    get_recent_audit,
    list_chat_threads,
)
from knowledge_transfer_agent.logging_config import get_logger
from knowledge_transfer_agent.services.followups import suggest_followups

logger = get_logger(__name__)

router = APIRouter()


class ChatThreadCreate(BaseModel):
    workspace_id: str | None = None
    title: str | None = Field(default=None, max_length=200)


class ChatThreadUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ChatThreadSchema(BaseModel):
    id: str
    workspace_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int | None = None


class ChatMessageSchema(BaseModel):
    id: int | None = None
    thread_id: str
    role: str
    content: str
    created_at: str | None = None


class ChatMessageCreate(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=50000)


class FollowupRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=20000)


class FollowupResponse(BaseModel):
    suggestions: list[str]


@router.get("/chats", response_model=list[ChatThreadSchema])
async def list_chats(
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
    workspace_id: str | None = None,
) -> list[ChatThreadSchema]:
    ws = resolve_workspace_id(workspace_id, header_workspace)
    threads = await run_in_threadpool(list_chat_threads, ws)
    return [
        ChatThreadSchema(
            id=t["id"],
            workspace_id=t["workspace_id"],
            title=t["title"],
            created_at=t["created_at"],
            updated_at=t["updated_at"],
            message_count=t.get("message_count"),
        )
        for t in threads
    ]


@router.post("/chats", response_model=ChatThreadSchema)
async def create_chat(
    request: ChatThreadCreate,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
) -> ChatThreadSchema:
    ws = resolve_workspace_id(request.workspace_id, header_workspace)
    t = await run_in_threadpool(create_chat_thread, ws, request.title)
    return ChatThreadSchema(**t, message_count=0)


@router.get("/chats/{thread_id}/messages", response_model=list[ChatMessageSchema])
async def get_messages(
    thread_id: str,
    _: Annotated[str, Depends(get_current_user)],
) -> list[ChatMessageSchema]:
    msgs = await run_in_threadpool(get_chat_messages, thread_id)
    return [ChatMessageSchema(**m) for m in msgs]


@router.post("/chats/{thread_id}/messages", response_model=ChatMessageSchema)
async def post_message(
    thread_id: str,
    request: ChatMessageCreate,
    _: Annotated[str, Depends(get_current_user)],
) -> ChatMessageSchema:
    try:
        m = await run_in_threadpool(
            append_chat_message, thread_id, request.role, request.content
        )
        return ChatMessageSchema(**m)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.patch("/chats/{thread_id}", response_model=ChatThreadSchema)
async def patch_chat(
    thread_id: str,
    request: ChatThreadUpdate,
    _: Annotated[str, Depends(get_current_user)],
) -> ChatThreadSchema:
    ok = await run_in_threadpool(update_chat_thread_title, thread_id, request.title)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    t = await run_in_threadpool(get_chat_thread, thread_id)
    if not t:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    return ChatThreadSchema(**t, message_count=t.get("message_count"))


@router.delete("/chats/{thread_id}")
async def remove_chat(
    thread_id: str,
    _: Annotated[str, Depends(get_current_user)],
) -> dict[str, str]:
    ok = await run_in_threadpool(delete_chat_thread, thread_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    return {"status": "ok"}


@router.get("/audit/recent")
async def audit_recent(
    _: Annotated[str, Depends(get_current_user)],
    limit: int = Query(default=50, ge=1, le=200),
    workspace_id: str | None = None,
) -> list[dict[str, Any]]:
    return await run_in_threadpool(get_recent_audit, limit, workspace_id)


@router.post("/suggest-followups", response_model=FollowupResponse)
async def api_suggest_followups(
    request: FollowupRequest,
    _: Annotated[str, Depends(get_current_user)],
) -> FollowupResponse:
    suggestions = await run_in_threadpool(
        suggest_followups, request.question, request.answer
    )
    return FollowupResponse(suggestions=suggestions)


class SharedMemorySchema(BaseModel):
    id: str
    workspace_id: str
    thread_id: str | None = None
    memory_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class SharedMemoryListResponse(BaseModel):
    workspace_id: str
    items: list[SharedMemorySchema]
    count: int


@router.get("/memory", response_model=SharedMemoryListResponse)
async def list_shared_memory(
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
    workspace_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> SharedMemoryListResponse:
    """List workspace-scoped shared long-term memories."""
    from knowledge_transfer_agent.core.shared_memory import list_workspace_memories

    ws = resolve_workspace_id(workspace_id, header_workspace)
    items = await run_in_threadpool(list_workspace_memories, ws, limit)
    return SharedMemoryListResponse(
        workspace_id=ws,
        items=[SharedMemorySchema(**i) for i in items],
        count=len(items),
    )


@router.delete("/memory")
async def clear_shared_memory(
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Clear all shared memories for a workspace."""
    from knowledge_transfer_agent.core.shared_memory import clear_workspace_memories

    ws = resolve_workspace_id(workspace_id, header_workspace)
    deleted = await run_in_threadpool(clear_workspace_memories, ws)
    return {"workspace_id": ws, "deleted": deleted}


@router.delete("/memory/{memory_id}")
async def delete_one_shared_memory(
    memory_id: str,
    header_workspace: Annotated[str, Depends(get_workspace_id_header)],
    _: Annotated[str, Depends(get_current_user)],
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Delete one shared memory entry."""
    from knowledge_transfer_agent.core.database import delete_shared_memory

    ws = resolve_workspace_id(workspace_id, header_workspace)
    ok = await run_in_threadpool(delete_shared_memory, memory_id, ws)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")
    return {"workspace_id": ws, "id": memory_id, "deleted": True}


@router.get("/ingest/jobs/{job_id}")
async def get_ingest_job_status(
    job_id: str,
    _: Annotated[str, Depends(get_current_user)],
) -> dict[str, Any]:
    from knowledge_transfer_agent.core.database import get_ingest_job

    job = await run_in_threadpool(get_ingest_job, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job
