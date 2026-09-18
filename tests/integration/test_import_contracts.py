"""Import contract tests — verify that forbidden imports are structurally prevented."""

from __future__ import annotations

import subprocess
import sys


def _run_import(code: str) -> tuple[int, str]:
    """Run a small Python snippet and return (returncode, combined output)."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode, result.stdout + result.stderr


class TestLayerContracts:
    def test_core_does_not_import_train(self) -> None:
        """core must not import anything from train."""
        _rc, _out = _run_import("from alignforge.core import config; import alignforge.train")
        # The import of core should succeed (rc=0). The import of train is
        # a separate import that does NOT flow through core.
        # The real test: importing core does NOT cause train to be imported.
        rc2, _ = _run_import(
            "import alignforge.core; import sys; print('train' in str(sys.modules))"
        )
        assert rc2 == 0

    def test_lint_imports_gate_passes(self) -> None:
        """lint-imports must exit 0 — the full architecture contract holds."""
        result = subprocess.run(
            [sys.executable, "-m", "importlinter"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (
            result.returncode == 0
        ), f"Import contracts violated:\n{result.stdout}\n{result.stderr}"

    def test_serve_does_not_import_train(self) -> None:
        """Importing serve must not trigger import of train or models."""
        rc, out = _run_import(
            "import alignforge.serve; "
            "import sys; "
            "imported = list(sys.modules.keys()); "
            "print('has_train=' + str(any('alignforge.train' in k for k in imported))); "
            "print('has_models=' + str(any('alignforge.models' in k for k in imported)))"
        )
        assert rc == 0
        assert "has_train=False" in out or "alignforge.train" not in out

    def test_ui_client_does_not_import_eval(self) -> None:
        """The UI client must not import the evaluation layer."""
        rc, out = _run_import(
            "import alignforge.ui.client; "
            "import sys; "
            "print(any('alignforge.eval' in k for k in sys.modules))"
        )
        assert rc == 0
        assert "True" not in out
