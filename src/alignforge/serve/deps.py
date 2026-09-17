"""FastAPI dependencies: engine resolution and registry access."""

from __future__ import annotations

# from typing import Annotated, Any
import structlog
from fastapi import HTTPException, Request  # Depends

from alignforge.serve.engine.base import InferenceEngine
from alignforge.serve.engine.echo import EchoEngine
from alignforge.serve.engine.ollama import OllamaEngine
from alignforge.serve.engine.transformers import TransformersEngine

log = structlog.get_logger()

# Singleton engine instances — one per backend type.
_engines: dict[str, InferenceEngine] = {}


def get_engine_registry() -> dict[str, InferenceEngine]:
    """Initialise engines from config. Called once at startup."""
    from alignforge.core.config import AlignForgeConfig

    cfg = AlignForgeConfig()
    global _engines

    default = cfg.serve.default_engine
    if "echo" not in _engines:
        _engines["echo"] = EchoEngine()
    if default == "ollama" and "ollama" not in _engines:
        _engines["ollama"] = OllamaEngine(host=cfg.serve.ollama_host)
    if default == "transformers" and "transformers" not in _engines:
        _engines["transformers"] = TransformersEngine()

    return _engines


def override_engines(overrides: dict[str, InferenceEngine]) -> None:
    """Replace engine instances — used in tests."""
    global _engines
    _engines.update(overrides)


async def get_engine_for(
    model_id: str,
    request: Request,
) -> tuple[InferenceEngine, str, str]:
    """Resolve model_id → (engine, display_name, weights_ref).

    Raises 404 if model_id is unknown, 503 if engine is unavailable.
    This is the connector C4 implementation from the master document.
    """
    # from alignforge.core.errors import ServingError
    from alignforge.core.registry import get_registry

    reg = get_registry()
    served = reg.get_served_model(model_id)

    if served is None:
        available = [m["model_id"] for m in reg.list_served_models()]
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Unknown model_id: {model_id!r}",
                "available": available,
            },
        )

    backend = served["backend"]
    weights_ref = served["weights_ref"]
    display_name = served["display_name"]

    engines = get_engine_registry()
    engine = engines.get(backend) or engines.get("echo")

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail={"error": f"No engine available for backend {backend!r}"},
        )

    log.debug(
        "engine_resolved",
        model_id=model_id,
        backend=backend,
        weights_ref=weights_ref,
    )
    return engine, display_name, weights_ref
