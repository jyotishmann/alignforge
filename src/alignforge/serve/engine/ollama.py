"""OllamaEngine: streams tokens from a local Ollama daemon via HTTP."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from alignforge.serve.engine.base import HealthStatus, InferenceEngine, StreamChunk

log = structlog.get_logger()


class OllamaEngine(InferenceEngine):
    """Proxies inference requests to a running Ollama daemon.

    One shared httpx.AsyncClient per engine instance — connection pooling
    means the first request after a cold start is the only slow one.
    """

    def __init__(self, host: str | None = None) -> None:
        self.host = host or os.environ.get("ALIGNFORGE_OLLAMA_HOST", "http://localhost:11434")
        self._client = httpx.AsyncClient(base_url=self.host, timeout=None)

    async def astream(
        self,
        messages: list[dict[str, str]],
        model_id: str,  # ignored here — weights_ref is the Ollama tag
        params: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Stream tokens from Ollama /api/chat."""
        weights_ref = params.get("_weights_ref", model_id)
        payload = {
            "model": weights_ref,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 0.9),
                "num_predict": params.get("max_tokens", 512),
                "seed": params.get("seed", 42),
            },
        }
        async with self._client.stream("POST", "/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)
                if content:
                    yield StreamChunk(
                        text=content,
                        finish_reason="stop" if done else None,
                    )
                elif done:
                    yield StreamChunk(text="", finish_reason="stop")

    async def ahealth(self, model_id: str, weights_ref: str) -> HealthStatus:
        """Check that the Ollama daemon is running and the tag is loaded."""
        import time

        t0 = time.monotonic()
        try:
            r = await self._client.get("/api/tags", timeout=3.0)
            r.raise_for_status()
            tags = [m["name"] for m in r.json().get("models", [])]
            latency = round((time.monotonic() - t0) * 1000, 1)
            if weights_ref in tags:
                return HealthStatus(
                    model_id=model_id,
                    backend="ollama",
                    status="ok",
                    latency_ms=latency,
                )
            return HealthStatus(
                model_id=model_id,
                backend="ollama",
                status="degraded",
                detail=f"Tag {weights_ref!r} not found in Ollama. Run: ollama pull {weights_ref}",
                latency_ms=latency,
            )
        except Exception as exc:
            return HealthStatus(
                model_id=model_id,
                backend="ollama",
                status="unavailable",
                detail=f"Ollama unreachable: {exc}",
            )
