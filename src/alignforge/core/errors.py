"""Error taxonomy. Every exception carries structured context, not just a message."""

from __future__ import annotations

from typing import Any


class AlignForgeError(Exception):
    """Root of all project exceptions. Catch this to catch everything of ours."""


class ConfigError(AlignForgeError):
    """Invalid, missing, or un-parseable configuration."""

    def __init__(self, message: str, *, key: str | None = None, value: Any = None) -> None:
        self.key = key
        self.value = value
        super().__init__(message)


class DataError(AlignForgeError):
    """Dataset loading, validation, or integrity failure."""

    def __init__(self, message: str, *, source: str | None = None) -> None:
        self.source = source
        super().__init__(message)


class RegistryError(AlignForgeError):
    """Run registry read/write failure."""

    def __init__(self, message: str, *, run_id: str | None = None) -> None:
        self.run_id = run_id
        super().__init__(message)


class ModelError(AlignForgeError):
    """Model loading, adaptation, or export failure."""


class TrainingError(AlignForgeError):
    """Training loop failure (OOM, divergence, checkpoint corruption)."""


class EvalError(AlignForgeError):
    """Evaluation generation, judging, or metrics failure."""


class ServingError(AlignForgeError):
    """Inference engine or API failure."""

    def __init__(self, message: str, *, model_id: str | None = None) -> None:
        self.model_id = model_id
        super().__init__(message)
