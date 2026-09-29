"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from alignforge.serve.middleware import (
    ConcurrencyLimiter,
    RequestIDMiddleware,
    TimingMiddleware,
)
from alignforge.serve.routes import chat, compare, evals, health, models, votes


def create_app(max_concurrent: int = 4) -> FastAPI:
    """Build and return the FastAPI application.

    Called once at process start. Each test can call it independently.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        from alignforge.core.registry import get_registry
        from alignforge.serve.deps import get_engine_registry

        get_registry()  # open the DB connection
        get_engine_registry()  # initialise engine singletons
        yield

    app = FastAPI(
        title="AlignForge Inference API",
        description="OpenAI-compatible inference for base, SFT, and DPO models.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Middleware (added in reverse order — last added = outermost).
    app.add_middleware(ConcurrencyLimiter, max_concurrent=max_concurrent)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes.
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(chat.router)
    app.include_router(compare.router)
    app.include_router(votes.router)
    app.include_router(evals.router)

    return app
