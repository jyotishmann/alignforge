"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from alignforge.serve.deps import get_engine_registry

router = APIRouter()


@router.get("/health")
async def liveness() -> dict[str, str]:
    """Liveness: is the process running? Always 200 if the server is up."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> JSONResponse:
    """Readiness: can we serve traffic? Checks each registered model."""
    from alignforge.core.registry import get_registry

    reg = get_registry()
    served_models = reg.list_served_models(enabled_only=True)
    engines = get_engine_registry()

    model_statuses: list[dict[str, object]] = []
    all_ok = True

    for sm in served_models:
        backend = sm["backend"]
        engine = engines.get(backend) or engines.get("echo")
        if engine is None:
            model_statuses.append(
                {
                    "model_id": sm["model_id"],
                    "status": "unavailable",
                    "detail": f"No engine for backend {backend!r}",
                }
            )
            all_ok = False
            continue

        health = await engine.ahealth(sm["model_id"], sm["weights_ref"])
        model_statuses.append(
            {
                "model_id": sm["model_id"],
                "display_name": sm["display_name"],
                "status": health.status,
                "backend": health.backend,
                "detail": health.detail,
                "latency_ms": health.latency_ms,
            }
        )
        if health.status != "ok":
            all_ok = False

    status_code = 200 if all_ok else 207  # 207 Multi-Status for partial health
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if all_ok else "degraded",
            "models": model_statuses,
        },
    )
