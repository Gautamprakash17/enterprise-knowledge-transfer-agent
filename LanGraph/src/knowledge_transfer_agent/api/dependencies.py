"""
FastAPI dependencies for auth, graph, config, and vector store.
"""

from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from knowledge_transfer_agent.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_vector_store_dep():
    """
    Dependency: provide vector store. Raises VectorStoreError if not available.
    """
    from knowledge_transfer_agent.core.exceptions import VectorStoreError
    from knowledge_transfer_agent.retrieval.vector_store import get_vector_store
    try:
        return get_vector_store()
    except (FileNotFoundError, OSError) as e:
        raise VectorStoreError(
            "Vector store not loaded. Run ingestion first.",
            details={"error": str(e)},
        ) from e


async def check_vector_store() -> bool:
    """Verify vector store is available (for health check)."""
    from knowledge_transfer_agent.retrieval.vector_store import get_vector_store
    try:
        get_vector_store()
        return True
    except (FileNotFoundError, OSError, Exception):
        return False


async def get_agent_service(
    vector_store: Any = Depends(get_vector_store_dep),
):
    """Dependency: provide AgentService with injected vector store."""
    from knowledge_transfer_agent.services.agent_service import AgentService
    return AgentService(vector_store=vector_store)


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
