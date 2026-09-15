"""AlignForge command-line entry point. Commands are thin adapters only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import typer

from alignforge import __version__
from alignforge.cli_state import GlobalState

log = structlog.get_logger()

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
def train_dpo(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c", exists=True),
    model_config: Path | None = typer.Option(None, "--model-config", "-m", exists=True),
    sft_run_id: str = typer.Option(..., "--sft-run", "-s", help="SFT run ID to align from."),
    pref_hash: str = typer.Option(
        ..., "--pref-hash", "-p", help="Preference dataset content hash."
    ),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
    dry_run: bool = typer.Option(False, "--dry-run"),
    set_overrides: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Run DPO preference alignment on top of an SFT checkpoint."""
    from alignforge.core.config import load_config
    from alignforge.core.logging import setup_logging
    from alignforge.core.paths import get_paths
    from alignforge.train.dpo_run import run_dpo

    paths = get_paths()
    cfg = load_config(component_path=config, overrides=set_overrides or [])
    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format, log_dir=paths.logs_dir)

    run_id = run_dpo(
        cfg=cfg,
        sft_run_id=sft_run_id,
        preference_dataset_hash=pref_hash,
        limit=limit,
        dry_run=dry_run,
    )

    if not dry_run:
        typer.secho(f"\nDPO complete. Run ID: {run_id}", fg=typer.colors.GREEN)
        typer.echo(f"Next step: alignforge eval all --models base,{sft_run_id},{run_id}")


@train_app.command("dpo-sweep")
def train_dpo_sweep(
    sft_run_id: str = typer.Option(..., "--sft-run", "-s"),
    pref_hash: str = typer.Option(..., "--pref-hash", "-p"),
    betas: str = typer.Option("0.05,0.1,0.3", "--betas", help="Comma-separated beta values."),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
) -> None:
    """Run DPO at multiple beta values and produce a comparison table."""
    from alignforge.core.config import load_config
    from alignforge.train.sweep import run_beta_sweep

    beta_list = [float(b.strip()) for b in betas.split(",")]
    cfg = load_config()
    results = run_beta_sweep(
        base_cfg=cfg,
        sft_run_id=sft_run_id,
        preference_hash=pref_hash,
        betas=beta_list,
        limit=limit,
    )
    typer.echo(f"\nSweep complete. {len(results)} runs.")
    typer.echo("Win rates added after: alignforge eval all ...")
    typer.echo("Report: reports/beta_sweep.md")


@eval_app.command("generate")
def eval_generate(
    models: str = typer.Option(
        ...,
        "--models",
        "-m",
        help="Comma-separated model IDs to generate from (e.g. base,sft,dpo).",
    ),
    suites: str | None = typer.Option(
        None,
        "--suites",
        help="Comma-separated suite names. Defaults to all registered suites.",
    ),
    params: str = typer.Option(
        "eval",
        "--params",
        help="Decode parameter set: 'eval' (sampled) or 'greedy' (deterministic).",
    ),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
    responses_dir: Path | None = typer.Option(
        None,
        "--responses-dir",
        help="Directory for response JSONL files. Default: evals/responses/",
    ),
) -> None:
    """Generate model responses for all eval suites. Checkpointed — safe to resume."""
    from alignforge.core.config import load_config
    from alignforge.core.paths import get_paths
    from alignforge.core.registry import get_registry
    from alignforge.eval.decode_params import NAMED_PARAMS
    from alignforge.eval.generation import run_generation
    from alignforge.eval.suites.loader import SUITE_REGISTRY, verify_suites_exist

    cfg = load_config()
    paths = get_paths()
    reg = get_registry()

    model_ids = [m.strip() for m in models.split(",")]
    suite_names = [s.strip() for s in suites.split(",")] if suites else list(SUITE_REGISTRY.keys())
    decode_params = NAMED_PARAMS.get(params, NAMED_PARAMS["eval"])

    # Verify suites exist before loading any model.
    problems = verify_suites_exist(suite_names, paths.evals_dir)
    if problems:
        for p in problems:
            typer.secho(f"  ⚠  {p}", fg=typer.colors.YELLOW)
        typer.echo("\nRun `scripts/build_eval_suites.py` to build MT-Bench and AlpacaEval subsets.")
        if any("domain_v1" in p for p in problems):
            typer.secho(
                "domain_v1 missing or too small — write your 50 cases first!",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    rdir = responses_dir or paths.evals_dir / "responses"

    # Need a ChatFormat for Transformers backends. Load the default tokenizer lazily.
    chat_format = _get_chat_format_for_eval(cfg)

    summary = run_generation(
        model_ids=model_ids,
        suite_names=suite_names,
        evals_dir=paths.evals_dir,
        responses_dir=rdir,
        registry=reg,
        chat_format=chat_format,
        params=decode_params,
        limit=limit,
    )

    typer.secho("\n✓ Generation complete:", fg=typer.colors.GREEN)
    for mid, n in summary.items():
        typer.echo(f"  {mid}: {n} responses")
    typer.echo(f"\nResponses written to: {rdir}")
    typer.echo("Next: alignforge eval judge --responses-dir <dir>")


def _get_chat_format_for_eval(cfg: Any) -> Any:
    """Load the chat format for eval — tokenizer only, no model weights."""
    try:
        from transformers import AutoTokenizer

        from alignforge.models.chat_format import ChatFormat

        tok = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code
        )
        return ChatFormat.from_tokenizer(tok)
    except Exception as exc:
        log.warning("chat_format_load_failed_using_echo", error=str(exc))
        # Return a minimal chat format object that works for the echo backend.
        from unittest.mock import MagicMock

        fmt = MagicMock()
        fmt.render_prompt = lambda msgs: msgs[-1]["content"] if msgs else ""
        return fmt


@eval_app.command("all")
def eval_all(
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated model IDs."),
    suites: str | None = typer.Option(None, "--suites"),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
) -> None:
    """Generate, judge, score and report in one pass. (judge+metrics added in Part 08.)"""
    # Phase 1: generation (available now).
    eval_generate(models=models, suites=suites, limit=limit)
    # Phases 2-3: judge + metrics — see Part 08.
    typer.secho(
        "\nGeneration done. Judge and metrics coming in Part 08.",
        fg=typer.colors.YELLOW,
    )


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
