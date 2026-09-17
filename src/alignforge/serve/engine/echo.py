"""EchoEngine: deterministic fake for CI and testing. No ML dependencies."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog

from alignforge.serve.engine.base import HealthStatus, InferenceEngine, StreamChunk

log = structlog.get_logger()

# Canned responses keyed by model_id prefix.
_CANNED: dict[str, list[str]] = {
    "base": ["This ", "is ", "the ", "base ", "model ", "response."],
    "sft": ["This ", "is ", "the ", "SFT ", "model. ", "More direct."],
    "dpo": ["Direct ", "answer: ", "use ", "sorted(). "],
    "default": ["Echo ", "response ", "from ", "the ", "test ", "engine."],
}


class EchoEngine(InferenceEngine):
    """Yields pre-canned tokens with a small delay to simulate streaming.

    This engine is the sole backend used in CI tests. It exercises all
    streaming infrastructure without any model weights.
    """

    def __init__(
        self,
        token_delay: float = 0.01,
        tokens_per_model: dict[str, list[str]] | None = None,
    ) -> None:
        self.token_delay = token_delay
        self._tokens = tokens_per_model or _CANNED

    async def astream(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        params: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        # Choose tokens by model_id prefix match.
        tokens = self._tokens.get(model_id, self._tokens.get("default", ["Echo."]))
        for i, token in enumerate(tokens):
            await asyncio.sleep(self.token_delay)
            is_last = i == len(tokens) - 1
            yield StreamChunk(
                text=token,
                index=0,
                finish_reason="stop" if is_last else None,
            )

    async def ahealth(self, model_id: str, weights_ref: str) -> HealthStatus:
        return HealthStatus(
            model_id=model_id,
            backend="echo",
            status="ok",
            detail="Echo engine — always healthy.",
        )
