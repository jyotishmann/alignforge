"""AlignForge command-line entry point. Commands are thin adapters only."""

from __future__ import annotations

from pathlib import Path

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

    for lib in ["bitsandbytes", "peft", "trl"]:
        ok = importlib.util.find_spec(lib) is not None
        table.add_row(lib, "[green]installed[/]" if ok else "[dim]absent[/]")

    table.add_row("ollama", "[dim]check added in Part 10[/]")

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
def data_build(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, help="Data config YAML."
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", min=1, help="Process only first N rows."
    ),
    skip_embedding: bool = typer.Option(
        False, "--skip-embedding", help="Skip Stage 2 embedding filter."
    ),
    labels: Path | None = typer.Option(
        None, "--labels", exists=True, help="Hand-label JSON for threshold tuning."
    ),
    set_overrides: list[str] | None = typer.Option(None, "--set", "-s", help="Config overrides."),
) -> None:
    """Build a versioned dataset artifact from HuggingFace sources."""
    from alignforge.core.config import load_config
    from alignforge.data.build import build_dataset

    cfg = load_config(component_path=config, overrides=set_overrides or [])
    artifact_dir, content_hash = build_dataset(
        cfg,
        limit=limit,
        skip_embedding=skip_embedding,
        labels_path=labels,
    )
    typer.secho(f"Dataset built: {artifact_dir}", fg=typer.colors.GREEN)
    typer.echo(f"Content hash: {content_hash}")


@data_app.command("inspect")
def data_inspect(
    path: Path = typer.Argument(..., exists=True, help="Path to the dataset artifact directory."),
) -> None:
    """Show the manifest for a dataset artifact."""
    import json

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        typer.secho(f"No manifest.json found in {path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    from rich import print_json
    from rich.console import Console

    Console().print(f"\n[bold]Dataset:[/bold] {manifest.get('name', 'unknown')}")
    print_json(data=manifest)


@data_app.command("stats")
def data_stats(
    path: Path = typer.Argument(..., exists=True, help="Path to the dataset artifact directory."),
) -> None:
    """Show the funnel table and token length statistics."""
    import json

    from rich.console import Console
    from rich.table import Table

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        typer.secho(f"No manifest.json found in {path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    console = Console()

    # Funnel table.
    funnel = manifest.get("funnel", {})
    if funnel:
        table = Table(title="Data Funnel")
        table.add_column("Stage")
        table.add_column("Rows", justify="right")
        for stage, count in funnel.items():
            table.add_row(stage, str(count))
        console.print(table)

    # Token stats.
    ts = manifest.get("token_stats", {})
    if ts:
        table = Table(title="Token Length Distribution")
        for k, v in ts.items():
            table.add_row(k, str(v))
        console.print(table)

    console.print(f"\nContent hash: {manifest.get('content_hash', 'N/A')}")


@train_app.command("sft")
def train_sft(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None, "--config", "-c", exists=True, help="Training config YAML."
    ),
    model_config: Path | None = typer.Option(
        None, "--model-config", "-m", exists=True, help="Model config YAML."
    ),
    dataset_hash: str = typer.Option(
        ..., "--dataset-hash", "-d", help="Content hash from `data build`."
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", min=1, help="Train on first N rows (debug)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without loading model."),
    resume: str | None = typer.Option(None, "--resume", help="Resume from checkpoint name."),
    drive_sync: str | None = typer.Option(
        None, "--drive-sync", help="Google Drive path for Colab sync."
    ),
    set_overrides: list[str] | None = typer.Option(None, "--set", "-s", help="Config overrides."),
) -> None:
    """Run QLoRA supervised fine-tuning. Requires [train] extras installed."""
    from alignforge.core.config import load_config
    from alignforge.core.logging import setup_logging
    from alignforge.core.paths import get_paths
    from alignforge.train.run import run_sft

    # Layer the config: base → model → training component → --set overrides.
    paths = get_paths()
    overrides = set_overrides or []
    cfg = load_config(component_path=config, overrides=overrides)
    if model_config:
        from alignforge.core.config import load_config as _lc

        cfg = _lc(component_path=model_config, overrides=overrides)

    setup_logging(
        level=cfg.logging.level,
        fmt=cfg.logging.format,
        log_dir=paths.logs_dir,
    )

    run_id = run_sft(
        cfg=cfg,
        dataset_hash=dataset_hash,
        limit=limit,
        dry_run=dry_run,
        resume_from=resume,
    )

    if not dry_run:
        typer.secho(f"\nSFT complete. Run ID: {run_id}", fg=typer.colors.GREEN)
        typer.echo(f"View logs:  cat logs/alignforge.log | jq 'select(.run_id==\"{run_id}\")'")
        typer.echo("Registry:   alignforge registry list")


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
