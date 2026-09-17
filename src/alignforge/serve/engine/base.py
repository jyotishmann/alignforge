"""InferenceEngine abstract base class and shared data types."""

from __future__ import annotations

# import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass  # field
from typing import Any

import structlog

log = structlog.get_logger()


@dataclass
class StreamChunk:
    """One token (or partial token) from a streaming generation."""

    text: str
    index: int = 0
    finish_reason: str | None = None  # "stop" | "length" | None


@dataclass
class HealthStatus:
    """Engine health report for /health/ready."""

    model_id: str
    backend: str
    status: str  # "ok" | "degraded" | "unavailable"
    detail: str = ""
    latency_ms: float | None = None


class InferenceEngine(ABC):
    """Abstract inference backend. Implement astream() and ahealth()."""

    @abstractmethod
    async def astream(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        params: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Yield token chunks for a chat completion request."""
        ...  # pragma: no cover

    @abstractmethod
    async def ahealth(self, model_id: str, weights_ref: str) -> HealthStatus:
        """Check whether this engine can serve the given model."""
        ...  # pragma: no cover

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        params: dict[str, Any],
    ) -> str:
        """Non-streaming completion — consume astream() and join.

        Implemented once here so every backend gets it for free.
        """
        parts: list[str] = []
        async for chunk in self.astream(messages, model_id, params):
            if chunk.text:
                parts.append(chunk.text)
        return "".join(parts)
