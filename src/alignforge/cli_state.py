"""Typed state shared between the CLI root callback and every subcommand."""

from __future__ import annotations

from dataclasses import dataclass

import typer


@dataclass(slots=True)
class GlobalState:
    """Carried on `ctx.obj`. Extended in later parts (logger, paths)."""

    verbose: bool = False
    quiet: bool = False
    dry_run: bool = False


def get_state(ctx: typer.Context) -> GlobalState:
    """Typed accessor for ctx.obj — avoids `Any` leaking into every command."""
    if not isinstance(ctx.obj, GlobalState):  # pragma: no cover - defensive
        raise RuntimeError("CLI state missing; the root callback did not run.")
    return ctx.obj
