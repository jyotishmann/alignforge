"""Gradio session state. One instance per user session."""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ColumnTelemetry:
    """Per-column streaming metrics."""

    model_id: str = ""
    display_name: str = ""
    t_start: float = field(default_factory=time.monotonic)
    t_first_token: float | None = None
    n_tokens: int = 0
    error: str | None = None

    @property
    def ttft_ms(self) -> float | None:
        if self.t_first_token is None:
            return None
        return round((self.t_first_token - self.t_start) * 1000, 1)

    @property
    def tokens_per_sec(self) -> float | None:
        elapsed = time.monotonic() - self.t_start
        if elapsed < 0.1 or self.n_tokens == 0:
            return None
        return round(self.n_tokens / elapsed, 1)

    def telemetry_line(self) -> str:
        parts = []
        if self.ttft_ms is not None:
            parts.append(f"TTFT {self.ttft_ms:.0f}ms")
        if self.tokens_per_sec is not None:
            parts.append(f"{self.tokens_per_sec:.1f} tok/s")
        parts.append(f"{self.n_tokens} tokens")
        return " · ".join(parts) if parts else ""


@dataclass
class ArenaSessionState:
    """Full state for one user's arena session."""

    session_id: str = field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:8]}")

    # Which model is in each column (blind mode may shuffle these).
    column_model_ids: list[str] = field(default_factory=lambda: ["base", "sft", "dpo"])
    # Whether the user can see which model is which.
    blind_mode: bool = True
    # The random permutation applied in blind mode.
    column_order: list[int] = field(default_factory=lambda: [0, 1, 2])

    # Current conversation (same prompt sent to all models).
    current_prompt: str = ""
    # The last responses per column (for voting).
    last_responses: list[str] = field(default_factory=lambda: ["", "", ""])
    # Telemetry per column.
    telemetry: list[ColumnTelemetry] = field(
        default_factory=lambda: [ColumnTelemetry(), ColumnTelemetry(), ColumnTelemetry()]
    )

    # Available models from the API.
    available_models: list[dict[str, str]] = field(default_factory=list)

    def shuffle_columns(self) -> ArenaSessionState:
        """Return a new state with columns randomly permuted (blind mode)."""
        import copy

        new = copy.deepcopy(self)
        new.column_order = list(range(len(self.column_model_ids)))
        random.shuffle(new.column_order)
        return new

    def model_at(self, visual_col: int) -> str:
        """Return the model_id at visual column visual_col after permutation."""
        return self.column_model_ids[self.column_order[visual_col]]

    def label_for(self, visual_col: int) -> str:
        """Return the label for a visual column (A/B/C in blind, name otherwise)."""
        if self.blind_mode:
            return ["A", "B", "C"][visual_col]
        return self.telemetry[self.column_order[visual_col]].display_name or self.model_at(
            visual_col
        )
