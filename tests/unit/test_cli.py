"""CLI smoke tests — verify the command tree is wired correctly."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from alignforge import __version__
from alignforge.cli import app


def test_version(cli: CliRunner) -> None:
    """The version command prints the package version and exits cleanly."""
    result = cli.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_doctor_runs_without_torch(cli: CliRunner) -> None:
    """Doctor must succeed even when optional ML extras are absent."""
    result = cli.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "AlignForge" in result.output
    # On a CPU dev environment, torch is absent. That is correct, not an error.


@pytest.mark.parametrize(
    "args",
    [
        ["data", "build"],
        # ["train", "sft"],
        ["train", "dpo"],
        ["eval", "all"],
        ["export", "gguf"],
        # ["registry", "list"],
        ["serve", "api"],
        ["serve", "ui"],
    ],
)
def test_stub_commands_exit_with_code_2(cli: CliRunner, args: list[str]) -> None:
    """Every stubbed command exits with code 2 and names its implementing part."""
    result = cli.invoke(app, args)
    assert result.exit_code == 2
    assert "Part" in result.output


def test_registry_list_runs(cli: CliRunner) -> None:
    """registry list runs successfully and shows a message."""
    result = cli.invoke(app, ["registry", "list"])
    assert result.exit_code == 0
    # Either shows runs or the empty message
    assert "No runs" in result.output or "run_id" in result.output
