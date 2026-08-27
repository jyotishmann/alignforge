"""Centralised path resolution. Every file operation goes through here."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def find_project_root() -> Path:
    """Walk up from this file until we find pyproject.toml."""
    current = Path(__file__).resolve().parent
    for ancestor in [current, *current.parents]:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    # Fallback: CWD (e.g., in a bare Colab cell).
    return Path.cwd()


class ProjectPaths:
    """Resolved, absolute paths for every artifact class. Created lazily."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or find_project_root()).resolve()

    def _dir(self, env_var: str, default: str) -> Path:
        """Resolve from env override or default, create if absent."""
        raw = os.environ.get(env_var, "")
        path = Path(raw) if raw else self.root / default
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        return self._dir("ALIGNFORGE_DATA_DIR", "data")

    @property
    def artifacts_dir(self) -> Path:
        return self._dir("ALIGNFORGE_ARTIFACTS_DIR", "artifacts")

    @property
    def models_dir(self) -> Path:
        return self._dir("ALIGNFORGE_MODELS_DIR", "models")

    @property
    def registry_db(self) -> Path:
        raw = os.environ.get("ALIGNFORGE_REGISTRY_DB", "")
        return Path(raw).resolve() if raw else self.root / "alignforge.db"

    @property
    def evals_dir(self) -> Path:
        return self.root / "evals"

    @property
    def reports_dir(self) -> Path:
        d = self.root / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def logs_dir(self) -> Path:
        d = self.root / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"


@lru_cache(maxsize=1)
def get_paths(root: Path | None = None) -> ProjectPaths:
    """Singleton accessor. Call without args for default project root."""
    return ProjectPaths(root)
