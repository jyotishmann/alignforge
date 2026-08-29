"""AlignForge command-line entry point. Commands are thin adapters only."""

from __future__ import annotations

import typer

from alignforge import __version__
from alignforge.cli_state import GlobalState

app = typer.Typer(
    name="alignforge",
    help="LLM post-training pipeline: QLoRA SFT → DPO → evaluation → serving.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

# Sub-applications, one per architecture layer. Populated in later parts.
data_app = typer.Typer(help="Build and inspect dataset artifacts.", no_args_is_help=True)
train_app = typer.Typer(
    help="Supervised fine-tuning and preference alignment.", no_args_is_help=True
)
eval_app = typer.Typer(help="Generation, judging, metrics and reports.", no_args_is_help=True)
export_app = typer.Typer(help="Merge adapters and export to GGUF / Ollama.", no_args_is_help=True)
registry_app = typer.Typer(help="Inspect and publish runs.", no_args_is_help=True)
serve_app = typer.Typer(help="Run the API and the UI.", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(export_app, name="export")
app.add_typer(registry_app, name="registry")
app.add_typer(serve_app, name="serve")


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug-level logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only."),
) -> None:
    """Runs before every subcommand; establishes shared state."""
    ctx.obj = GlobalState(verbose=verbose, quiet=quiet)


@app.command()
def version() -> None:
    """Print the installed AlignForge version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Report the environment: Python, optional extras, and GPU availability."""
    import importlib.util
    import platform
    import sys

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"AlignForge {__version__} environment")
    table.add_column("Component")
    table.add_column("Status")

    table.add_row("Python", f"{sys.version.split()[0]} ({platform.machine()})")

    # find_spec checks importability WITHOUT executing the module — much faster,
    # and avoids bitsandbytes' CUDA warnings on CPU-only machines.
    extras = {
        "data": "datasets",
        "train": "transformers",
        "eval": "scipy",
        "serve": "fastapi",
        "ui": "gradio",
    }
    for extra, probe in extras.items():
        ok = importlib.util.find_spec(probe) is not None
        table.add_row(f"extra: {extra}", "[green]installed[/]" if ok else "[dim]absent[/]")

    if importlib.util.find_spec("torch") is not None:
        import torch

        cuda = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if cuda else "cpu only"
        table.add_row("CUDA", f"[green]{detail}[/]" if cuda else f"[yellow]{detail}[/]")
    else:
        table.add_row("CUDA", "[dim]torch not installed[/]")

    console.print(table)


def _not_yet(part: str) -> None:
    """Uniform placeholder so the command tree is walkable before implementation."""
    typer.secho(f"Not implemented yet — see {part}.", fg=typer.colors.YELLOW)
    raise typer.Exit(code=2)


@data_app.command("build")
def data_build() -> None:
    """Build a versioned dataset artifact from HuggingFace sources."""
    _not_yet("Part 03")


@train_app.command("sft")
def train_sft() -> None:
    """Run QLoRA supervised fine-tuning."""
    _not_yet("Part 05")


@train_app.command("dpo")
def train_dpo() -> None:
    """Run DPO preference alignment."""
    _not_yet("Part 06")


@eval_app.command("all")
def eval_all() -> None:
    """Generate, judge, score and report in one pass."""
    _not_yet("Part 08")


@export_app.command("gguf")
def export_gguf() -> None:
    """Merge adapters and export a quantised GGUF."""
    _not_yet("Part 09")


@registry_app.command("list")
def registry_list(
    kind: str | None = typer.Option(None, "--kind", "-k", help="Filter by run kind."),
    limit: int = typer.Option(50, "--limit", "-n", min=1, help="Max rows to show."),
) -> None:
    """List runs recorded in the registry."""
    from rich.console import Console
    from rich.table import Table

    from alignforge.core.registry import get_registry

    reg = get_registry()
    runs = reg.list_runs(kind=kind, limit=limit)

    if not runs:
        typer.echo("No runs recorded yet.")
        raise typer.Exit()

    console = Console()
    table = Table(title=f"Runs ({len(runs)})")
    table.add_column("run_id")
    table.add_column("kind")
    table.add_column("status")
    table.add_column("config_hash")
    table.add_column("started_at")

    for r in runs:
        status = r["status"]
        style = {"completed": "green", "running": "yellow", "failed": "red"}.get(status, "")
        table.add_row(
            r["run_id"],
            r["kind"],
            f"[{style}]{status}[/{style}]" if style else status,
            r["config_hash"][:8],
            r.get("started_at", "")[:19],
        )
    console.print(table)


@serve_app.command("api")
def serve_api() -> None:
    """Run the FastAPI inference service."""
    _not_yet("Part 10")


@serve_app.command("ui")
def serve_ui() -> None:
    """Run the Gradio comparison UI."""
    _not_yet("Part 11")
