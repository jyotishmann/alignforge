"""Core layer: config, logging, paths, registry. No ML imports."""

from alignforge.core.config import AlignForgeConfig, config_hash, load_config
from alignforge.core.errors import (
    AlignForgeError,
    ConfigError,
    DataError,
    EvalError,
    ModelError,
    RegistryError,
    ServingError,
    TrainingError,
)
from alignforge.core.hardware import HardwareInfo, probe_hardware
from alignforge.core.logging import get_logger, request_id_var, run_id_var, setup_logging
from alignforge.core.paths import ProjectPaths, get_paths
from alignforge.core.registry import Registry, get_registry, reset_registry
from alignforge.core.seed import seed_everything

__all__ = [
    "AlignForgeConfig",
    "AlignForgeError",
    "ConfigError",
    "DataError",
    "EvalError",
    "HardwareInfo",
    "ModelError",
    "ProjectPaths",
    "Registry",
    "RegistryError",
    "ServingError",
    "TrainingError",
    "config_hash",
    "get_logger",
    "get_paths",
    "get_registry",
    "load_config",
    "probe_hardware",
    "request_id_var",
    "reset_registry",
    "run_id_var",
    "seed_everything",
    "setup_logging",
]
