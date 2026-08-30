"""Pydantic schemas for dataset records. Every raw row is validated into one of these."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class SFTExample(BaseModel):
    """Single-turn instruction → response pair (the Alpaca shape)."""

    instruction: str = Field(..., min_length=5)
    input: str = ""  # optional context
    response: str = Field(..., min_length=5)
    source: str = ""  # which HF dataset this came from

    @field_validator("instruction", "response")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field is blank after stripping whitespace.")
        return v

    @property
    def prompt_text(self) -> str:
        """The user-facing text (instruction + input), for filtering and dedup."""
        if self.input:
            return f"{self.instruction}\n\n{self.input}"
        return self.instruction

    @property
    def full_text(self) -> str:
        """The complete example text, for dedup hashing."""
        return f"{self.prompt_text}\n\n{self.response}"


class PreferenceExample(BaseModel):
    """Preference triplet: exactly what DPOTrainer consumes."""

    prompt: str = Field(..., min_length=5)
    chosen: str = Field(..., min_length=5)
    rejected: str = Field(..., min_length=5)
    source: str = ""

    @field_validator("prompt", "chosen", "rejected")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field is blank after stripping whitespace.")
        return v

    @property
    def full_text(self) -> str:
        """For dedup: hash over all three fields."""
        return f"{self.prompt}\n{self.chosen}\n{self.rejected}"

    def model_post_init(self, __context: object) -> None:
        """Reject identical chosen/rejected — zero-signal for DPO."""
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError(
                "chosen and rejected are identical after stripping — "
                "this pair provides no learning signal for DPO."
            )


class QuarantineRecord(BaseModel):
    """A row that failed validation, preserved for auditing."""

    raw: dict[str, Any]
    source: str
    error: str
    stage: str = "validation"  # or "normalisation", "filter", "dedup", etc.
