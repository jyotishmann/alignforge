"""Reusable CLI option types, so flags behave identically across commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

# Truncate input to the first N records. Plumbed to the EARLIEST load point so
# that `--limit 50` is genuinely fast, not just a smaller output.
LimitOpt = Annotated[
    int | None,
    typer.Option("--limit", "-n", min=1, help="Process only the first N records (debugging)."),
]

ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", exists=True, dir_okay=False, help="Component config YAML."),
]

# Dotted overrides applied last in the config merge (connector C1, step 4).
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override a config key, e.g. --set train.lr=1e-4."),
]

DryRunOpt = Annotated[
    bool,
    typer.Option("--dry-run", help="Resolve config and print the plan without executing."),
]

RunIdOpt = Annotated[
    str,
    typer.Argument(help="Run identifier from the registry, e.g. sft-20250312-9c1e."),
]
