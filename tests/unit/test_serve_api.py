"""API endpoint tests — complete suite using EchoEngine. No GPU required."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from alignforge.serve.engine.echo import EchoEngine


@pytest.fixture
def app_with_echo(tmp_path: Path) -> Generator[FastAPI, None, None]:
    """FastAPI app with EchoEngine and a seeded registry."""
    import os

    os.environ["ALIGNFORGE_REGISTRY_DB"] = str(tmp_path / "test.db")
    os.environ["ALIGNFORGE_DATA_DIR"] = str(tmp_path / "data")

    from alignforge.core.paths import get_paths
    from alignforge.core.registry import get_registry, reset_registry
    from alignforge.serve.deps import override_engines

    reset_registry()
    get_paths.cache_clear()

    reg = get_registry()
    reg.publish_model("base", "Base Model", "echo", "echo", sort_order=1)
    reg.publish_model("sft", "SFT Model", "echo", "echo", sort_order=2)
    reg.publish_model("dpo", "DPO Model", "echo", "echo", sort_order=3)

    override_engines({"echo": EchoEngine(token_delay=0.0)})

    from alignforge.serve.app import create_app

    application = create_app(max_concurrent=10)

    yield application

    reset_registry()
    get_paths.cache_clear()


@pytest.fixture
def client(app_with_echo: FastAPI) -> TestClient:
    return TestClient(app_with_echo)


class TestHealthEndpoints:
    def test_liveness(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_readiness_with_echo_models(self, client: TestClient) -> None:
        r = client.get("/health/ready")
        assert r.status_code in (200, 207)
        body = r.json()
        assert "models" in body
        assert len(body["models"]) == 3


class TestModelsRoute:
    def test_lists_enabled_models(self, client: TestClient) -> None:
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()["data"]
        ids = [m["id"] for m in data]
        assert "base" in ids
        assert "sft" in ids
        assert "dpo" in ids


class TestChatCompletions:
    def test_non_streaming(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "dpo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert len(body["choices"][0]["message"]["content"]) > 0

    def test_streaming_returns_event_stream(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "sft",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            },
            headers={"Accept": "text/event-stream"},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]

    def test_streaming_sse_frames(self, client: TestClient) -> None:
        """SSE frames must be parseable OpenAI-format JSON."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "base",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        lines = r.text.split("\n")
        data_lines = [
            label[6:] for label in lines if label.startswith("data: ") and label[6:] != "[DONE]"
        ]
        assert len(data_lines) > 0
        for raw in data_lines:
            parsed = json.loads(raw)
            assert "choices" in parsed
            assert parsed["object"] == "chat.completion.chunk"

    def test_done_terminator_present(self, client: TestClient) -> None:
        """[DONE] must appear as the final SSE frame."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "dpo",
                "messages": [{"role": "user", "content": "End?"}],
                "stream": True,
            },
        )
        assert "data: [DONE]" in r.text

    def test_unknown_model_returns_404(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent_model",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 404
        assert "available" in r.json()["detail"]

    def test_request_id_echoed_in_response(self, client: TestClient) -> None:
        """X-Request-ID must be echoed in the response headers."""
        r = client.post(
            "/v1/chat/completions",
            json={"model": "dpo", "messages": [{"role": "user", "content": "x"}]},
            headers={"X-Request-ID": "test-req-123"},
        )
        assert r.headers.get("x-request-id") == "test-req-123"


class TestCompareRoute:
    def test_compare_returns_all_models(self, client: TestClient) -> None:
        r = client.post(
            "/v1/compare",
            json={
                "models": ["base", "sft", "dpo"],
                "messages": [{"role": "user", "content": "Compare test"}],
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 3
        model_ids = {res["model_id"] for res in results}
        assert model_ids == {"base", "sft", "dpo"}

    def test_compare_partial_failure_isolated(self, client: TestClient) -> None:
        """A failed model must not block other results."""
        r = client.post(
            "/v1/compare",
            json={
                "models": ["base", "nonexistent_model"],
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        errors = [res for res in results if res.get("error")]
        ok = [res for res in results if not res.get("error")]
        assert len(errors) == 1
        assert len(ok) == 1


class TestVotesRoute:
    def test_vote_recorded(self, client: TestClient) -> None:
        r = client.post(
            "/v1/votes",
            json={
                "prompt": "How do I sort a list?",
                "response_a": "Use sorted().",
                "response_b": "Use .sort() which modifies in place.",
                "model_a": "sft",
                "model_b": "dpo",
                "winner": "b",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "recorded"
        assert body["vote_id"].startswith("vote-")
