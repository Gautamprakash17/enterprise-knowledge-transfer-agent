"""
FastAPI middleware.
"""

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from knowledge_transfer_agent.logging_config import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Basic logging middleware: request method/path, duration, status."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path
        start = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s %s %d %.2fms",
                method,
                path,
                response.status_code,
                duration_ms,
            )
            return response
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "%s %s failed after %.2fms: %s",
                method,
                path,
                duration_ms,
                str(e),
            )
            raise
