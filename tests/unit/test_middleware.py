"""Middleware unit tests — concurrency limiter and request-ID."""

from __future__ import annotations

# import asyncio
# import json
# import logging
# from pathlib import Path
# from unittest.mock import AsyncMock, MagicMock, patch

# import pytest


class TestRequestIDMiddleware:
    def test_generates_id_when_absent(self, api_client) -> None:
        """If X-Request-ID is not provided, the server generates one."""
        r = api_client.get("/health")
        assert "x-request-id" in r.headers
        assert r.headers["x-request-id"].startswith("req_")

    def test_echoes_provided_id(self, api_client) -> None:
        r = api_client.get("/health", headers={"X-Request-ID": "custom-id-xyz"})
        assert r.headers["x-request-id"] == "custom-id-xyz"

    def test_sse_response_has_no_cache(self, api_client) -> None:
        """SSE responses must have Cache-Control: no-cache."""
        r = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "dpo",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert "no-cache" in r.headers.get("cache-control", "")

    def test_accel_buffering_disabled_for_sse(self, api_client) -> None:
        """X-Accel-Buffering: no must be set on streaming responses."""
        r = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "dpo",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert r.headers.get("x-accel-buffering") == "no"


class TestConcurrencyLimiter:
    def test_health_bypasses_limiter(self, api_client) -> None:
        """/health must succeed even if concurrency is maxed."""
        # The test client is synchronous, so we can't truly max concurrency,
        # but we can verify health returns 200 regardless.
        r = api_client.get("/health")
        assert r.status_code == 200
