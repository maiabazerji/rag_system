"""Middleware for request ID tracking and context propagation.

Generates a unique request ID for each request and stores it in context
so it's automatically included in all logs for that request.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

request_id_context: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get the current request ID from context."""
    return request_id_context.get()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that generates and tracks request IDs.

    For each request:
    1. Generates a unique request ID: request-{timestamp}-{uuid}
    2. Stores in context var so all downstream code can access it
    3. Adds X-Request-ID header to response

    This enables request tracing across logs from multiple modules
    and services.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request, generating and tracking request ID."""
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        request_id = f"request-{timestamp}-{unique_id}"

        token = request_id_context.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_context.reset(token)
