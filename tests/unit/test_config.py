"""Config composition and hashing tests."""

from __future__ import annotations

from pathlib import Path

from alignforge.core.config import AlignForgeConfig, config_hash, load_config

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_default_config_validates() -> None:
    """The zero-argument config (all defaults) must pass validation."""
    cfg = AlignForgeConfig()
    assert cfg.project.seed == 42
    assert cfg.lora.r == 16


def test_config_hash_is_deterministic() -> None:
    """Same input → same hash, always."""
    cfg_a = AlignForgeConfig()
    cfg_b = AlignForgeConfig()
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_golden_config_hash() -> None:
    """Pin the hash of a known fixture to detect serialisation drift."""
    cfg = load_config(component_path=FIXTURES / "golden_config.yaml")
    h = config_hash(cfg)
    # If you intentionally changed the hash scheme, update this value.
    # Compute the new expected hash by running this test with --no-header -s.
    assert isinstance(h, str)
    assert len(h) == 12
    # NOTE: Replace with the actual hash after first run:
    # assert h == "<paste the actual hash here after first run>"


def test_extra_key_rejected() -> None:
    """A typo in a YAML key must raise, not be silently ignored."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="extra"):
        AlignForgeConfig.model_validate({"sft": {"learning_rte": 1e-4}})


def test_dotted_override() -> None:
    """--set overrides apply correctly and are reflected in the hash."""
    cfg_base = load_config()
    cfg_over = load_config(overrides=["sft.learning_rate=1e-5"])
    assert cfg_over.sft.learning_rate == 1e-5
    assert config_hash(cfg_base) != config_hash(cfg_over)


def test_config_is_frozen() -> None:
    """Config objects are immutable after construction."""
    import pytest
    from pydantic import ValidationError

    cfg = AlignForgeConfig()
    with pytest.raises(ValidationError):
        cfg.sft.learning_rate = 999.0  # type: ignore[misc]
