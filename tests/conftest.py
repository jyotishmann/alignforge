"""Root conftest — fixtures available to all test files."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli() -> CliRunner:
    """Invoke the CLI without spawning a subprocess."""
    return CliRunner()


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """A temporary directory pretending to be a project root."""
    for d in ("data", "artifacts", "models", "logs", "evals", "reports"):
        (tmp_path / d).mkdir()
    return tmp_path
