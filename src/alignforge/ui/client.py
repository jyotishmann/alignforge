"""SSE streaming client — connects to the API and yields content deltas."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

# import os
from typing import Any

import httpx
import structlog

log = structlog.get_logger()

# Shared client — one per process, connection-pooled.
_client: httpx.AsyncClient | None = None


def get_http_client(api_base: str) -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=api_base,
            timeout=None,  # inference duration is unbounded
        )
    return _client


async def stream_chat(
    api_base: str,
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    request_id: str | None = None,
) -> AsyncIterator[str]:
    """Open a streaming chat request and yield content deltas.

    Parses OpenAI-format SSE frames. Yields each content string as received.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    client = get_http_client(api_base)
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
    }

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as response:
        response.raise_for_status()

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str == "[DONE]":
                return
            try:
                data = json.loads(payload_str)
                delta = data["choices"][0]["delta"].get("content")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


async def get_models(api_base: str) -> list[dict[str, Any]]:
    """Fetch the list of available models from the API."""
    client = get_http_client(api_base)
    try:
        r = await client.get("/v1/models", timeout=5.0)
        r.raise_for_status()
        return r.json().get("data", [])  # type: ignore[no-any-return]
    except Exception as exc:
        log.warning("get_models_failed", error=str(exc))
        return []


async def post_vote(
    api_base: str,
    prompt: str,
    response_a: str,
    response_b: str,
    model_a: str,
    model_b: str,
    winner: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Record a pairwise preference vote."""
    client = get_http_client(api_base)
    r = await client.post(
        "/v1/votes",
        json={
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "model_a": model_a,
            "model_b": model_b,
            "winner": winner,
            "session_id": session_id,
        },
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()  # type: ignore[no-any-return]


async def check_health(api_base: str) -> dict[str, Any]:
    """Check API readiness."""
    client = get_http_client(api_base)
    try:
        r = await client.get("/health/ready", timeout=3.0)
        return r.json()  # type: ignore[no-any-return]
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}
