"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from knowledge_transfer_agent import __version__
from knowledge_transfer_agent.api.middleware import LoggingMiddleware
from knowledge_transfer_agent.api.rate_limit import get_limiter
from knowledge_transfer_agent.api.routes import router
from knowledge_transfer_agent.config import get_settings
from knowledge_transfer_agent.core.exceptions import (
    AgentError,
    LLMError,
    RetrievalError,
    VectorStoreError,
)
from knowledge_transfer_agent.logging_config import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: setup and teardown."""
    setup_logging()
    yield
    # Cleanup if needed


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    limiter = get_limiter()
    app.state.limiter = limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi import _rate_limit_exceeded_handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Exception handlers for custom errors
    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError):
        return JSONResponse(
            status_code=503,
            content={"detail": exc.message, "type": "llm_error", "details": exc.details},
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(_: Request, exc: RetrievalError):
        return JSONResponse(
            status_code=503,
            content={"detail": exc.message, "type": "retrieval_error", "details": exc.details},
        )

    @app.exception_handler(VectorStoreError)
    async def vector_store_error_handler(_: Request, exc: VectorStoreError):
        return JSONResponse(
            status_code=503,
            content={"detail": exc.message, "type": "vector_store_error", "details": exc.details},
        )

    @app.exception_handler(AgentError)
    async def agent_error_handler(_: Request, exc: AgentError):
        return JSONResponse(
            status_code=500,
            content={"detail": exc.message, "type": "agent_error", "details": exc.details},
        )

    app.add_middleware(LoggingMiddleware)
    try:
        from slowapi.middleware import SlowAPIMiddleware
        app.add_middleware(SlowAPIMiddleware)
    except ImportError:
        pass
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1", tags=["Knowledge Transfer"])

    @app.get("/")
    async def root():
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/health",
            "ask": "/api/v1/ask",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "knowledge_transfer_agent.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
