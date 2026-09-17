"""POST /v1/compare — generate from multiple models concurrently."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from alignforge.serve.deps import get_engine_for

router = APIRouter()


class CompareRequest(BaseModel):
    models: list[str]  # 2-3 model IDs
    messages: list[dict[str, str]]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512


class CompareResult(BaseModel):
    model_id: str
    display_name: str
    response: str
    time_ms: float
    error: str | None = None


class CompareResponse(BaseModel):
    results: list[CompareResult]


@router.post("/v1/compare", response_model=CompareResponse)
async def compare(body: CompareRequest, request: Request) -> CompareResponse:
    """Generate concurrently from multiple models. Returns when all complete."""

    async def _one(model_id: str) -> CompareResult:
        t0 = time.monotonic()
        try:
            engine, display_name, weights_ref = await get_engine_for(model_id, request)
            params: dict[str, Any] = {
                "temperature": body.temperature,
                "top_p": body.top_p,
                "max_tokens": body.max_tokens,
                "_weights_ref": weights_ref,
            }
            text = await engine.acomplete(body.messages, model_id, params)
            return CompareResult(
                model_id=model_id,
                display_name=display_name,
                response=text,
                time_ms=round((time.monotonic() - t0) * 1000, 1),
            )
        except Exception as exc:
            return CompareResult(
                model_id=model_id,
                display_name=model_id,
                response="",
                time_ms=round((time.monotonic() - t0) * 1000, 1),
                error=str(exc),
            )

    results = await asyncio.gather(*[_one(mid) for mid in body.models])
    return CompareResponse(results=list(results))
