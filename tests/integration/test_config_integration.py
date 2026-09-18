"""Config system integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alignforge.core.config import AlignForgeConfig, config_hash, load_config


def test_three_layer_merge(tmp_path: Path) -> None:
    """Component config overrides base; --set overrides component."""
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.dump(
            {
                "project": {"name": "base_project", "seed": 42},
                "sft": {"learning_rate": 2e-4, "num_train_epochs": 3},
            }
        )
    )

    component = tmp_path / "component.yaml"
    component.write_text(
        yaml.dump(
            {
                "sft": {"learning_rate": 1e-4},  # overrides base
            }
        )
    )

    cfg = load_config(
        component_path=component,
        overrides=["sft.num_train_epochs=1"],
        base_path=base,
    )

    assert cfg.project.name == "base_project"  # from base
    assert cfg.sft.learning_rate == 1e-4  # from component
    assert cfg.sft.num_train_epochs == 1  # from --set


def test_config_hash_is_stable_across_runs(tmp_path: Path) -> None:
    """Same config, same hash — deterministic serialisation."""
    cfg1 = AlignForgeConfig()
    cfg2 = AlignForgeConfig()
    assert config_hash(cfg1) == config_hash(cfg2)


def test_config_hash_changes_on_any_field(tmp_path: Path) -> None:
    """Even a single field change produces a different hash."""
    cfg_a = load_config(overrides=["project.seed=42"])
    cfg_b = load_config(overrides=["project.seed=43"])
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_deeply_nested_extra_key_rejected() -> None:
    """extra='forbid' applies to nested models, not just the root."""
    with pytest.raises(Exception, match="extra"):
        AlignForgeConfig.model_validate({"sft": {"nonexistent_nested_key": True}})


def test_none_override_does_not_wipe_base_value(tmp_path: Path) -> None:
    """A None value in a component config must not wipe the base default."""
    base = tmp_path / "base.yaml"
    base.write_text(yaml.dump({"project": {"seed": 99}}))

    component = tmp_path / "component.yaml"
    # Component has 'seed: null' — should NOT override base's 99.
    component.write_text("project:\n  name: override_name\n  seed: null\n")

    cfg = load_config(component_path=component, base_path=base)
    assert cfg.project.name == "override_name"  # overridden
    assert cfg.project.seed == 99  # NOT wiped by null
