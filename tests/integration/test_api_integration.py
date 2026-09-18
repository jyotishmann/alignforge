"""API integration tests — full stack with seeded registry and EchoEngine."""

from __future__ import annotations

import json


class TestChatCompletionsIntegration:
    def test_streaming_all_frames_parseable(self, api_client) -> None:
        """Every SSE data frame must be parseable as OpenAI JSON."""
        r = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "dpo",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert r.status_code == 200
        data_frames = [
            label[6:]
            for label in r.text.splitlines()
            if label.startswith("data: ") and label[6:] != "[DONE]"
        ]
        assert len(data_frames) >= 1
        for frame in data_frames:
            parsed = json.loads(frame)
            assert "choices" in parsed
            assert parsed["object"] == "chat.completion.chunk"

    def test_non_streaming_response_schema(self, api_client) -> None:
        r = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "sft",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": False,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        msg = body["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], str)
        assert len(msg["content"]) > 0

    def test_request_id_propagates_through_response(self, api_client) -> None:
        custom_id = "my-test-request-abc"
        r = api_client.post(
            "/v1/chat/completions",
            json={"model": "dpo", "messages": [{"role": "user", "content": "x"}]},
            headers={"X-Request-ID": custom_id},
        )
        assert r.headers.get("x-request-id") == custom_id

    def test_error_envelope_has_available_models(self, api_client) -> None:
        """404 for unknown model must list the available model IDs."""
        r = api_client.post(
            "/v1/chat/completions",
            json={
                "model": "completely_unknown",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 404
        detail = r.json()["detail"]
        assert "available" in detail
        assert "base" in detail["available"]


class TestHealthIntegration:
    def test_liveness_always_200(self, api_client) -> None:
        assert api_client.get("/health").status_code == 200

    def test_readiness_reports_all_registered_models(self, api_client) -> None:
        r = api_client.get("/health/ready")
        assert r.status_code in (200, 207)
        model_ids = {m["model_id"] for m in r.json()["models"]}
        assert {"base", "sft", "dpo"} <= model_ids

    def test_readiness_shape(self, api_client) -> None:
        r = api_client.get("/health/ready")
        body = r.json()
        assert "status" in body
        assert "models" in body
        for m in body["models"]:
            assert "model_id" in m
            assert "status" in m


class TestVotesIntegration:
    def test_vote_round_trip_to_registry(self, api_client, seeded_registry) -> None:
        """A vote posted to /v1/votes appears in the registry."""
        r = api_client.post(
            "/v1/votes",
            json={
                "prompt": "How do I sort a dict?",
                "response_a": "Use sorted().",
                "response_b": "Use dict.items() with sorted().",
                "model_a": "sft",
                "model_b": "dpo",
                "winner": "b",
                "session_id": "test-session-001",
            },
        )
        assert r.status_code == 200
        vote_id = r.json()["vote_id"]
        assert vote_id.startswith("vote-")

        # Verify the vote appears in the registry.
        votes = seeded_registry.list_votes()
        assert len(votes) >= 1
        matching = [v for v in votes if v["vote_id"] == vote_id]
        assert len(matching) == 1
        assert matching[0]["winner"] == "b"
        assert matching[0]["model_b"] == "dpo"


class TestCompareIntegration:
    def test_compare_returns_three_results(self, api_client) -> None:
        r = api_client.post(
            "/v1/compare",
            json={
                "models": ["base", "sft", "dpo"],
                "messages": [{"role": "user", "content": "What is Python?"}],
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 3
        assert all(res.get("response") for res in results if not res.get("error"))

    def test_compare_models_are_distinct(self, api_client) -> None:
        """Each column must return a response from its respective model."""
        r = api_client.post(
            "/v1/compare",
            json={
                "models": ["base", "sft", "dpo"],
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        results = r.json()["results"]
        model_ids = {res["model_id"] for res in results}
        assert model_ids == {"base", "sft", "dpo"}
