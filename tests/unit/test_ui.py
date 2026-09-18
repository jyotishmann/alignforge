"""UI unit tests — no Gradio runtime, no API server."""

from __future__ import annotations

import time
from unittest.mock import patch  #  AsyncMock, MagicMock,

import pytest


class TestColumnTelemetry:
    def test_ttft_is_none_before_first_token(self) -> None:
        from alignforge.ui.state import ColumnTelemetry

        t = ColumnTelemetry()
        assert t.ttft_ms is None

    def test_ttft_computed_after_first_token(self) -> None:
        from alignforge.ui.state import ColumnTelemetry

        t = ColumnTelemetry()
        t.t_start = time.monotonic() - 0.5
        t.t_first_token = time.monotonic()
        assert t.ttft_ms is not None
        assert t.ttft_ms >= 490

    def test_tokens_per_sec(self) -> None:
        from alignforge.ui.state import ColumnTelemetry

        t = ColumnTelemetry()
        t.t_start = time.monotonic() - 1.0
        t.n_tokens = 10
        tps = t.tokens_per_sec
        assert tps is not None
        assert 8 < tps < 12

    def test_telemetry_line_format(self) -> None:
        from alignforge.ui.state import ColumnTelemetry

        t = ColumnTelemetry()
        t.t_start = time.monotonic() - 2.0
        t.t_first_token = time.monotonic() - 1.9
        t.n_tokens = 20
        line = t.telemetry_line()
        assert "tok/s" in line
        assert "tokens" in line


class TestArenaSessionState:
    def test_default_column_model_ids(self) -> None:
        from alignforge.ui.state import ArenaSessionState

        s = ArenaSessionState()
        assert len(s.column_model_ids) == 3

    def test_shuffle_produces_permutation(self) -> None:
        from alignforge.ui.state import ArenaSessionState

        s = ArenaSessionState(column_model_ids=["base", "sft", "dpo"])
        orders = set()
        for _ in range(20):
            shuffled = s.shuffle_columns()
            orders.add(tuple(shuffled.column_order))
        # Over 20 shuffles, we should see more than one ordering.
        assert len(orders) > 1

    def test_model_at_respects_permutation(self) -> None:
        from alignforge.ui.state import ArenaSessionState

        s = ArenaSessionState(column_model_ids=["base", "sft", "dpo"])
        s.column_order = [2, 0, 1]  # dpo first, base second, sft third
        assert s.model_at(0) == "dpo"
        assert s.model_at(1) == "base"
        assert s.model_at(2) == "sft"

    def test_blind_mode_labels(self) -> None:
        from alignforge.ui.state import ArenaSessionState

        s = ArenaSessionState(column_model_ids=["base", "sft", "dpo"], blind_mode=True)
        assert s.label_for(0) in ("A", "B", "C")

    def test_non_blind_labels_show_model(self) -> None:
        from alignforge.ui.state import ArenaSessionState

        s = ArenaSessionState(
            column_model_ids=["base", "sft", "dpo"],
            blind_mode=False,
        )
        s.telemetry[0].display_name = "Base Model"
        assert "Base" in s.label_for(0)


class TestSSEClient:
    @pytest.mark.asyncio
    async def test_stream_chat_yields_deltas(self) -> None:
        """SSE client correctly parses data: frames and yields content."""
        import json

        from alignforge.ui.client import stream_chat

        # Build a fake SSE stream.
        def _make_frame(content: str, done: bool = False) -> str:
            chunk = {
                "id": "test",
                "object": "chat.completion.chunk",
                "model": "dpo",
                "choices": [
                    {
                        "delta": {"content": content},
                        "finish_reason": "stop" if done else None,
                    }
                ],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        fake_lines = [
            _make_frame("Hello "),
            _make_frame("world"),
            _make_frame("", done=True),
            "data: [DONE]\n\n",
        ]

        from typing import ClassVar

        class FakeResponse:
            status_code = 200
            headers: ClassVar[dict[str, str]] = {"content-type": "text/event-stream"}

            async def aiter_lines(self):
                for line in "".join(fake_lines).splitlines():
                    yield line

            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class FakeClient:
            def stream(self, *args, **kwargs):
                return FakeResponse()

        with patch("alignforge.ui.client.get_http_client", return_value=FakeClient()):
            deltas = []
            async for d in stream_chat(
                "http://localhost:8000", "dpo", [{"role": "user", "content": "Hi"}]
            ):
                deltas.append(d)

        assert deltas == ["Hello ", "world"]
