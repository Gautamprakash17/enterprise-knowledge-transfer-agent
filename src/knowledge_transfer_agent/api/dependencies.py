"""
FastAPI dependencies for auth, graph, config, and vector store.
"""

from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.workspaces import normalize_workspace_id

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_workspace_id_header(
    x_workspace_id: Annotated[str | None, Header(alias="X-Workspace-Id")] = None,
) -> str:
    """Resolve project workspace from header (default project if omitted)."""
    return normalize_workspace_id(x_workspace_id)


async def get_vector_store_dep(
    workspace_id: Annotated[str, Depends(get_workspace_id_header)],
):
    """
    Dependency: provide vector store for the active workspace.
    Raises VectorStoreError if not available.
    """
    from knowledge_transfer_agent.core.exceptions import VectorStoreError
    from knowledge_transfer_agent.retrieval.vector_store import get_vector_store

    try:
        return get_vector_store(workspace_id=workspace_id)
    except (FileNotFoundError, OSError) as e:
        raise VectorStoreError(
            f"No index for workspace '{workspace_id}'. Add documents to this project first.",
            details={"error": str(e), "workspace_id": workspace_id},
        ) from e


async def check_vector_store(
    workspace_id: Annotated[str, Depends(get_workspace_id_header)],
) -> bool:
    """Verify vector store is available for this workspace (health check)."""
    from knowledge_transfer_agent.retrieval.vector_store import get_vector_store

    try:
        get_vector_store(workspace_id=workspace_id)
        return True
    except (FileNotFoundError, OSError, Exception):
        return False


async def get_agent_service(
    vector_store: Any = Depends(get_vector_store_dep),
):
    """Dependency: provide AgentService with workspace-scoped vector store."""
    from knowledge_transfer_agent.services.agent_service import AgentService

    return AgentService(vector_store=vector_store)


def resolve_workspace_id(body_id: str | None, header_id: str) -> str:
    """Body workspace_id wins over X-Workspace-Id header."""
    if body_id and str(body_id).strip():
        return normalize_workspace_id(body_id)
    return normalize_workspace_id(header_id)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
    api_key: Annotated[str | None, Depends(_api_key_header)],
) -> str:
    """
    Validate API key or Bearer token when RBAC is enabled.
    Returns user/role identifier.
    """
    settings = get_settings()
    if not settings.enable_rbac:
        return "default"

    # Check API key first
    if api_key and api_key == settings.secret_key:
        return "api_user"

    # Check Bearer token
    if credentials and credentials.credentials == settings.secret_key:
        return "api_user"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key / Bearer token",
    )
