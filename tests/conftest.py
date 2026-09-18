"""Root conftest — shared fixtures for the entire test suite."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from alignforge.core.registry import Registry, get_registry

# ── CLI ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cli() -> CliRunner:
    return CliRunner()


# ── File system ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Temporary directory structured like a project root."""
    for d in ("data", "artifacts", "models", "logs", "evals", "reports"):
        (tmp_path / d).mkdir()
    return tmp_path


# ── Environment isolation ─────────────────────────────────────────────────


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    """Monkeypatches all ALIGNFORGE_ env vars to point to tmp_path.

    Resets the registry and paths singletons after each test so they
    don't bleed into adjacent tests.
    """
    monkeypatch.setenv("ALIGNFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ALIGNFORGE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ALIGNFORGE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ALIGNFORGE_REGISTRY_DB", str(tmp_path / "test.db"))

    for d in ("data", "artifacts", "models", "logs", "reports", "evals"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)

    yield tmp_path

    # Teardown: reset singletons so the next test gets a clean slate.
    from alignforge.core.paths import get_paths
    from alignforge.core.registry import reset_registry

    reset_registry()
    get_paths.cache_clear()


# ── Registry ──────────────────────────────────────────────────────────────


@pytest.fixture
def seeded_registry(isolated_env: Path) -> Registry:
    """A registry pre-populated with base/sft/dpo echo models."""

    reg = get_registry()
    reg.publish_model("base", "Base Model", "echo", "echo", sort_order=1)
    reg.publish_model("sft", "SFT Model", "echo", "echo", sort_order=2)
    reg.publish_model("dpo", "DPO Model", "echo", "echo", sort_order=3)
    return reg


# ── FastAPI test client ───────────────────────────────────────────────────


@pytest.fixture
def api_client(seeded_registry):
    """Full FastAPI test client backed by EchoEngine. No GPU, no network."""
    from fastapi.testclient import TestClient

    from alignforge.serve.app import create_app
    from alignforge.serve.deps import override_engines
    from alignforge.serve.engine.echo import EchoEngine

    override_engines({"echo": EchoEngine(token_delay=0.0)})
    application = create_app(max_concurrent=10)
    return TestClient(application)
