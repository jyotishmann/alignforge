"""Integration test: full data pipeline with --limit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.slow
def test_data_build_produces_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end data build with limit=20 produces a valid artifact."""
    # Point all paths to tmp_path.
    monkeypatch.setenv("ALIGNFORGE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ALIGNFORGE_REGISTRY_DB", str(tmp_path / "test.db"))

    # Reset singletons so they pick up the new env.
    from alignforge.core.paths import get_paths
    from alignforge.core.registry import reset_registry

    reset_registry()
    get_paths.cache_clear()

    from alignforge.core.config import load_config
    from alignforge.data.build import build_dataset

    cfg = load_config(
        overrides=[
            "data.name=test_build",
            "data.sources=[alpaca_cleaned]",
            "data.decontaminate=false",
        ]
    )
    artifact_dir, content_hash = build_dataset(cfg, limit=20, skip_embedding=True)

    # Check structure.
    assert (artifact_dir / "train.parquet").exists()
    assert (artifact_dir / "val.parquet").exists()
    assert (artifact_dir / "manifest.json").exists()

    with (artifact_dir / "manifest.json").open() as f:
        manifest = json.load(f)

    assert manifest["content_hash"] == content_hash
    assert manifest["n_total"] > 0

    # Funnel must be monotonically non-increasing.
    funnel = manifest["funnel"]
    stages = ["loaded", "validated"]
    values = [funnel.get(s, 0) for s in stages if s in funnel]
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1], f"Funnel not monotonic at stage {stages[i]}"

    # Cleanup.
    reset_registry()
    get_paths.cache_clear()
