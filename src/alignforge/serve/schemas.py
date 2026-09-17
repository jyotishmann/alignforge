"""OpenAI-compatible request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Chat completion request ───────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = False
    seed: int | None = None


# ── Chat completion response (non-streaming) ──────────────────────────────


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)


# ── Streaming delta (SSE) ─────────────────────────────────────────────────


class ChatCompletionDelta(BaseModel):
    content: str | None = None
    role: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    model: str
    choices: list[ChatCompletionChunkChoice]


# ── Vote request ──────────────────────────────────────────────────────────


class VoteRequest(BaseModel):
    prompt: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    winner: Literal["a", "b", "tie"]
    session_id: str | None = None


class VoteResponse(BaseModel):
    vote_id: str
    status: str = "recorded"


# ── Error envelope ────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    error: str
    type: str = "api_error"
    request_id: str | None = None


# ── Model listing ─────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    display_name: str
    backend: str


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
