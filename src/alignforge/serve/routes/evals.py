"""GET /v1/evals — list evaluation results for the UI eval browser."""

from __future__ import annotations

import json
from typing import cast

# from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/v1/evals")
async def list_evals() -> list[dict[str, object]]:
    """List eval reports in the reports directory."""
    from alignforge.core.paths import get_paths

    paths = get_paths()
    reports = sorted(paths.reports_dir.glob("eval_*.md"), reverse=True)
    result = []
    for r in reports[:20]:
        eid = r.stem.replace("eval_", "")
        metrics_file = paths.reports_dir / f"metrics_{eid}.json"
        summary: dict[str, object] = {"eval_id": eid}
        if metrics_file.exists():
            with metrics_file.open() as f:
                summary.update(json.load(f))
        result.append(summary)
    return result


@router.get("/v1/evals/{eval_id}")
async def get_eval(eval_id: str) -> dict[str, object]:
    """Return metrics JSON for one eval run."""
    from alignforge.core.paths import get_paths

    paths = get_paths()
    mfile = paths.reports_dir / f"metrics_{eval_id}.json"
    if not mfile.exists():
        raise HTTPException(status_code=404, detail=f"Eval {eval_id!r} not found.")
    with mfile.open() as f:
        return cast(dict[str, object], json.load(f))
