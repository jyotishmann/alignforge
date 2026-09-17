"""GET /v1/models — list available models."""

from __future__ import annotations

from fastapi import APIRouter

from alignforge.serve.schemas import ModelInfo, ModelList

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List enabled served models. Gradio calls this on startup."""
    from alignforge.core.registry import get_registry

    reg = get_registry()
    served = reg.list_served_models(enabled_only=True)
    return ModelList(
        data=[
            ModelInfo(
                id=m["model_id"],
                display_name=m["display_name"],
                backend=m["backend"],
            )
            for m in served
        ]
    )
