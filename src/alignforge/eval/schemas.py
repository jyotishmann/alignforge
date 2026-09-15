"""Evaluation case schemas and response records."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One evaluation prompt with its metadata."""

    id: str = Field(..., description="Stable unique identifier, e.g. 'd01'.")
    prompt: str = Field(..., min_length=5)
    intent: str = Field(
        ...,
        description="What this case tests. Used by the judge as context.",
    )
    difficulty: int = Field(default=2, ge=1, le=3, description="1=easy, 2=medium, 3=hard.")
    verifiable: bool = Field(
        default=False,
        description="True if there is an objectively correct answer.",
    )
    expected_answer: str | None = Field(
        default=None,
        description="For verifiable cases: the correct answer or code output.",
    )
    suite: str = Field(default="", description="Populated at load time from the filename.")
    category: str = Field(default="", description="Sub-category within the suite.")


class GeneratedResponse(BaseModel):
    """A model's response to one EvalCase."""

    case_id: str
    suite: str
    model_id: str
    prompt: str
    response: str
    n_tokens: int
    time_seconds: float
    decode_params: dict[str, object] = Field(default_factory=dict)
    error: str | None = None  # populated if generation failed


class GenerationCheckpoint(BaseModel):
    """Tracks progress through a generation run for Colab-resilient resumption."""

    model_id: str
    suite: str
    total_cases: int
    completed_ids: list[str] = Field(default_factory=list)
    failed_ids: list[str] = Field(default_factory=list)

    def is_done(self, case_id: str) -> bool:
        return case_id in self.completed_ids or case_id in self.failed_ids

    @property
    def n_remaining(self) -> int:
        return self.total_cases - len(self.completed_ids) - len(self.failed_ids)
