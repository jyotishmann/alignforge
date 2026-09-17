"""POST /v1/chat/completions — OpenAI-compatible chat endpoint with SSE."""

from __future__ import annotations

# import json
# import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request  # Depends, HTTPException
from fastapi.responses import StreamingResponse

from alignforge.serve.deps import get_engine_for
from alignforge.serve.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
) -> Any:
    """OpenAI-compatible chat completion. Supports stream=True (SSE)."""
    engine, _display_name, weights_ref = await get_engine_for(body.model, request)

    params: dict[str, Any] = {
        "temperature": body.temperature,
        "top_p": body.top_p,
        "max_tokens": body.max_tokens,
        "_weights_ref": weights_ref,
    }
    if body.seed is not None:
        params["seed"] = body.seed

    messages = [m.model_dump() for m in body.messages]
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if body.stream:
        return StreamingResponse(
            _stream_sse(engine, messages, body.model, params, completion_id),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )
    else:
        # Non-streaming: consume the async generator.
        full_text = await engine.acomplete(messages, body.model, params)
        return ChatCompletionResponse(
            id=completion_id,
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=full_text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(completion_tokens=len(full_text.split())),
        )


async def _stream_sse(
    engine: Any,
    messages: list[dict[str, str]],
    model: str,
    params: dict[str, Any],
    completion_id: str,
) -> AsyncIterator[str]:
    """Async generator yielding SSE frames in the OpenAI wire format."""
    # First chunk: role announcement (OpenAI-compatible pattern).
    first = ChatCompletionChunk(
        id=completion_id,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionDelta(role="assistant"), finish_reason=None
            )
        ],
    )
    yield f"data: {first.model_dump_json()}\n\n"

    # Content chunks.
    async for chunk in engine.astream(messages, model, params):
        if chunk.text:
            c = ChatCompletionChunk(
                id=completion_id,
                model=model,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionDelta(content=chunk.text),
                        finish_reason=chunk.finish_reason,
                    )
                ],
            )
            yield f"data: {c.model_dump_json()}\n\n"

    # Terminator — load-bearing: without \n\n clients buffer forever.
    yield "data: [DONE]\n\n"
