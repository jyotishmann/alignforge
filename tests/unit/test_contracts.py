"""Verify that import-linter contracts pass — the architecture holds."""

from __future__ import annotations

import subprocess


def test_import_contracts_pass() -> None:
    """lint-imports must exit 0 — architectural layering is intact."""
    result = subprocess.run(
        ["uv", "run", "lint-imports"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import contracts violated:\n{result.stdout}\n{result.stderr}"
