"""POST /v1/votes — record human preference votes (connector C6)."""

from __future__ import annotations

from fastapi import APIRouter

from alignforge.serve.schemas import VoteRequest, VoteResponse

router = APIRouter()


@router.post("/v1/votes", response_model=VoteResponse)
async def record_vote(body: VoteRequest) -> VoteResponse:
    """Record a pairwise preference vote. Connector C6: UI vote → DPO data."""
    from alignforge.core.registry import get_registry

    reg = get_registry()

    # Expand 3-way vote to pairwise rows.
    # For a 3-model arena, the UI sends one vote identifying the best model.
    # We expand to pairwise: winner > model_a, winner > model_b.
    vote_id = reg.record_vote(
        prompt=body.prompt,
        response_a=body.response_a,
        response_b=body.response_b,
        model_a=body.model_a,
        model_b=body.model_b,
        winner=body.winner,
        session_id=body.session_id,
    )
    return VoteResponse(vote_id=vote_id)
