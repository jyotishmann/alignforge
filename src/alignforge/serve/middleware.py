"""FastAPI middleware: request-ID propagation, timing, and concurrency limiting."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from alignforge.core.logging import request_id_var

log = structlog.get_logger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, bind to logging ContextVar, echo in response.

    Connector C8: from this middleware forward, every log line emitted
    anywhere in the request's call stack carries request_id automatically.
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        req_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:8]}"
        token = request_id_var.set(req_id)
        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            # SSE responses: prevent proxy buffering.
            if "text/event-stream" in response.headers.get("content-type", ""):
                response.headers["X-Accel-Buffering"] = "no"
                response.headers["Cache-Control"] = "no-cache"
            return response
        finally:
            request_id_var.reset(token)


class TimingMiddleware(BaseHTTPMiddleware):
    """Log request duration as a structured event."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        t0 = time.monotonic()
        response: Response = await call_next(request)
        duration_ms = round((time.monotonic() - t0) * 1000, 1)
        log.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )
        return response


class ConcurrencyLimiter(BaseHTTPMiddleware):
    """Reject requests beyond a concurrency cap with 503.

    Reject-immediately is preferable to queue-and-timeout for inference:
    a queued request would wait behind slow generation and then time out
    anyway, while a fast 503 lets the client retry with backoff.
    """

    def __init__(self, app: Any, max_concurrent: int = 4) -> None:
        super().__init__(app)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Only limit inference routes, not health checks.
        if request.url.path in ("/health", "/health/ready"):
            return await call_next(request)  # type: ignore[no-any-return]

        if not self._semaphore._value:  # semaphore is exhausted
            log.warning(
                "concurrency_limit_reached",
                path=request.url.path,
                max=self._max,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Server at capacity. Retry after a moment.",
                    "type": "capacity_error",
                },
                headers={"Retry-After": "2"},
            )

        async with self._semaphore:
            return await call_next(request)  # type: ignore[no-any-return]
