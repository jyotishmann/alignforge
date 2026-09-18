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

    try:
        import httpx

        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        tags = [m["name"] for m in r.json().get("models", [])]
        ollama_status = f"[green]running[/] — {len(tags)} model(s)"
        if any("alignforge" in t for t in tags):
            ollama_status += " [dim](alignforge model found)[/dim]"
    except Exception:
        ollama_status = "[yellow]not running[/] (optional — needed for GGUF serving)"
    table.add_row("ollama", ollama_status)

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
    models: str = typer.Option(..., "--models", "-m"),
    suites: str | None = typer.Option(None, "--suites"),
    backend: str = typer.Option("echo", "--backend", "-b"),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
) -> None:
    """Generate, judge, compute metrics, and render report in one pass."""
    # Phase 1: generate.
    eval_generate(models=models, suites=suites, limit=limit)
    # Phase 2: judge.
    eval_judge(models=models, suites=suites, backend=backend, limit=limit)
    # Phase 3: metrics + report.
    eval_report(models=models, suites=suites)
    typer.secho("\n✓ eval all complete.", fg=typer.colors.GREEN)


@eval_app.command("judge")
def eval_judge(
    models: str = typer.Option(..., "--models", "-m", help="Comma-separated model IDs."),
    suites: str | None = typer.Option(None, "--suites"),
    backend: str = typer.Option(
        "echo", "--backend", "-b", help="Judge backend: openai|local|echo."
    ),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge-model"),
    responses_dir: Path | None = typer.Option(None, "--responses-dir"),
    judgements_dir: Path | None = typer.Option(None, "--judgements-dir"),
    limit: int | None = typer.Option(None, "--limit", "-n", min=1),
) -> None:
    """Run position-debiased pairwise judging on generated responses."""
    from alignforge.core.paths import get_paths
    from alignforge.eval.judge import run_judging
    from alignforge.eval.judge_client import get_judge_fn
    from alignforge.eval.suites.loader import SUITE_REGISTRY

    paths = get_paths()
    model_ids = [m.strip() for m in models.split(",")]
    suite_names = [s.strip() for s in suites.split(",")] if suites else list(SUITE_REGISTRY.keys())

    typer.echo(f"Judge backend: {backend}" + (f" ({judge_model})" if backend == "openai" else ""))
    typer.echo(f"Models: {model_ids}")
    typer.echo(f"Suites: {suite_names}")
    typer.echo(f"Pairs: {len(model_ids) * (len(model_ids) - 1) // 2} (each judged twice per case)")

    judge_fn = get_judge_fn(backend, model=judge_model if backend == "openai" else None)

    rdir = responses_dir or paths.evals_dir / "responses"
    jdir = judgements_dir or paths.evals_dir / "judgements"

    summary = run_judging(
        model_ids=model_ids,
        suite_names=suite_names,
        responses_dir=rdir,
        judgements_dir=jdir,
        judge_fn=judge_fn,
        evals_dir=paths.evals_dir,
        limit=limit,
    )

    typer.secho("\n✓ Judging complete:", fg=typer.colors.GREEN)
    for pair, n in summary.items():
        typer.echo(f"  {pair}: {n} cases judged")
    typer.echo(f"\nNext: alignforge eval metrics --judgements-dir {jdir}")


@eval_app.command("metrics")
def eval_metrics(
    models: str = typer.Option(..., "--models", "-m"),
    suites: str | None = typer.Option(None, "--suites"),
    judgements_dir: Path | None = typer.Option(None, "--judgements-dir"),
    responses_dir: Path | None = typer.Option(None, "--responses-dir"),
    n_resamples: int = typer.Option(10_000, "--n-resamples"),
    output: Path | None = typer.Option(None, "--output", help="Save metrics JSON to this path."),
) -> None:
    """Compute win rates, CIs, Bradley-Terry Elo, and length-controlled metrics."""
    import json

    from alignforge.core.paths import get_paths
    from alignforge.eval.metrics import compute_all_metrics
    from alignforge.eval.suites.loader import SUITE_REGISTRY

    paths = get_paths()
    model_ids = [m.strip() for m in models.split(",")]
    suite_names = [s.strip() for s in suites.split(",")] if suites else list(SUITE_REGISTRY.keys())

    jdir = judgements_dir or paths.evals_dir / "judgements"
    rdir = responses_dir or paths.evals_dir / "responses"

    metrics = compute_all_metrics(
        model_ids=model_ids,
        suite_names=suite_names,
        judgements_dir=jdir,
        responses_dir=rdir,
        evals_dir=paths.evals_dir,
        n_resamples=n_resamples,
    )

    # Print headline.
    typer.secho("\n=== Win Rates ===", fg=typer.colors.GREEN)
    for pair, wr in metrics.get("pairwise", {}).items():
        typer.echo(f"  {pair}: {wr['win_rate']} [{wr['ci_low']}, {wr['ci_high']}] (n={wr['n']})")

    if metrics.get("elo"):
        typer.secho("\n=== Bradley-Terry Elo ===", fg=typer.colors.GREEN)
        for model, score in sorted(metrics["elo"].items(), key=lambda x: -x[1]):
            typer.echo(f"  {model}: {score}")

    if output:
        with output.open("w") as f:
            json.dump(metrics, f, indent=2)
        typer.echo(f"\nMetrics saved to {output}")


@eval_app.command("report")
def eval_report(
    models: str = typer.Option(..., "--models", "-m"),
    suites: str | None = typer.Option(None, "--suites"),
    eval_id: str | None = typer.Option(None, "--eval-id", help="Identifier for this eval run."),
    judgements_dir: Path | None = typer.Option(None, "--judgements-dir"),
    responses_dir: Path | None = typer.Option(None, "--responses-dir"),
    judge: str = typer.Option("gpt-4o-mini", "--judge"),
    n_resamples: int = typer.Option(10_000, "--n-resamples"),
) -> None:
    """Generate Markdown and HTML evaluation reports with regression gallery."""
    import uuid

    from alignforge.core.paths import get_paths
    from alignforge.core.registry import get_registry
    from alignforge.eval.metrics import compute_all_metrics
    from alignforge.eval.report import render_report
    from alignforge.eval.suites.loader import SUITE_REGISTRY

    paths = get_paths()
    model_ids = [m.strip() for m in models.split(",")]
    suite_names = [s.strip() for s in suites.split(",")] if suites else list(SUITE_REGISTRY.keys())
    eid = eval_id or uuid.uuid4().hex[:8]

    jdir = judgements_dir or paths.evals_dir / "judgements"
    rdir = responses_dir or paths.evals_dir / "responses"

    metrics = compute_all_metrics(
        model_ids=model_ids,
        suite_names=suite_names,
        judgements_dir=jdir,
        responses_dir=rdir,
        evals_dir=paths.evals_dir,
        n_resamples=n_resamples,
    )

    md_path, html_path = render_report(
        metrics=metrics,
        judgements_dir=jdir,
        responses_dir=rdir,
        evals_dir=paths.evals_dir,
        output_dir=paths.reports_dir,
        eval_id=eid,
        judge_name=judge,
    )

    # Register the evaluation in the registry.
    pairwise = metrics.get("pairwise", {})
    for pair_key, _wr in pairwise.items():
        _model_a, _model_b = pair_key.split("_vs_")
        get_registry()  # ensure connected
        from alignforge.core.registry import get_registry as _gr

        _gr().conn.execute(
            """INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('eval_schema', '1')"""
        )
        # Store in a simple way (full evaluations table added in a migration).
    typer.secho(f"\n✓ Report generated: {md_path}", fg=typer.colors.GREEN)
    typer.echo(f"HTML: {html_path}")


@export_app.command("gguf")
def export_gguf(
    dpo_run: str = typer.Option(..., "--dpo-run", "-d", help="DPO run ID from registry."),
    config: Path | None = typer.Option(None, "--config", "-c", exists=True),
    model_config: Path | None = typer.Option(None, "--model-config", "-m", exists=True),
    dry_run: bool = typer.Option(False, "--dry-run"),
    skip_smoke: bool = typer.Option(False, "--skip-smoke-test"),
    skip_eval: bool = typer.Option(False, "--skip-eval", help="Skip GGUF deployment delta eval."),
    set_overrides: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Merge DPO adapter, convert to GGUF, quantise, and register with Ollama."""
    from alignforge.core.config import load_config
    from alignforge.core.logging import setup_logging
    from alignforge.core.paths import get_paths
    from alignforge.export.gguf import run_export

    cfg = load_config(component_path=config, overrides=set_overrides or [])
    paths = get_paths()
    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format, log_dir=paths.logs_dir)

    artifacts = run_export(
        cfg=cfg,
        dpo_run_id=dpo_run,
        dry_run=dry_run,
        skip_smoke_test=skip_smoke,
        skip_gguf_eval=skip_eval,
    )

    if not dry_run and artifacts:
        typer.secho("\n✓ Export complete:", fg=typer.colors.GREEN)
        for kind, path in artifacts.items():
            typer.echo(f"  {kind}: {path}")
        tag = artifacts.get("ollama_tag", "")
        if tag:
            typer.echo("\nTest with curl:")
            typer.echo(
                f'  curl http://localhost:11434/api/generate -d \'{{"model":"{tag}",'
                f'"prompt":"How do I reverse a list?","stream":false}}\''
            )
            typer.echo("\nRegister for serving:")
            typer.echo(
                f"  alignforge registry publish --run {dpo_run} --as dpo --display 'DPO (β=0.1)'"
            )


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


@registry_app.command("publish")
def registry_publish(
    run_id: str = typer.Option(..., "--run", "-r", help="Run ID of the model to publish."),
    model_id: str = typer.Option(
        ..., "--as", "-a", help="Stable model ID for the API (e.g. 'dpo')."
    ),
    display_name: str = typer.Option(
        ..., "--display", "-d", help="Human-readable name for the UI."
    ),
    backend: str = typer.Option(
        "ollama", "--backend", help="Engine backend: ollama|transformers|echo."
    ),
    weights_ref: str | None = typer.Option(
        None, "--weights", help="Tag or path. Inferred from artifacts if omitted."
    ),
    sort_order: int = typer.Option(0, "--sort", help="Display order in the UI (lower = first)."),
) -> None:
    """Make a trained model available to the API and UI."""
    from alignforge.core.registry import get_registry

    reg = get_registry()

    # Infer weights_ref from artifacts if not provided.
    ref = weights_ref
    if ref is None:
        arts = reg.get_artifacts(run_id)
        for art in arts:
            if (backend == "ollama" and art["kind"] == "ollama_tag") or (
                backend == "transformers" and art["kind"] == "lora_adapter"
            ):
                ref = art["path"]
                break
        if ref is None:
            typer.secho(
                f"Could not infer weights_ref for backend={backend!r}. "
                f"Specify --weights explicitly.",
                fg=typer.colors.RED,
            )
            raise typer.Exit(1)

    reg.publish_model(
        model_id=model_id,
        display_name=display_name,
        backend=backend,
        weights_ref=ref,
        run_id=run_id,
        sort_order=sort_order,
    )
    typer.secho(f"✓ Published: {model_id} → {ref}", fg=typer.colors.GREEN)
    typer.echo("The model will appear in the arena UI after `alignforge serve all` restarts.")


@registry_app.command("unpublish")
def registry_unpublish(
    model_id: str = typer.Argument(..., help="Model ID to disable."),
) -> None:
    """Disable a served model without deleting it from the registry."""
    from alignforge.core.registry import get_registry

    reg = get_registry()
    reg.conn.execute("UPDATE served_models SET enabled = 0 WHERE model_id = ?", (model_id,))
    reg.conn.commit()
    typer.secho(f"✓ Unpublished: {model_id}", fg=typer.colors.GREEN)


@registry_app.command("show")
def registry_show(
    run_id: str = typer.Argument(..., help="Run ID to inspect."),
) -> None:
    """Show full details for one run: config hash, artifacts, metrics."""
    # import json

    from rich.console import Console
    from rich.table import Table

    from alignforge.core.registry import get_registry

    reg = get_registry()
    run = reg.get_run(run_id)
    if not run:
        typer.secho(f"Run {run_id!r} not found.", fg=typer.colors.RED)
        raise typer.Exit(1)

    console = Console()
    table = Table(title=f"Run: {run_id}")
    table.add_column("Field")
    table.add_column("Value")
    for field in [
        "kind",
        "status",
        "config_hash",
        "dataset_hash",
        "git_sha",
        "started_at",
        "finished_at",
    ]:
        table.add_row(field, str(run.get(field, "")))
    console.print(table)

    arts = reg.get_artifacts(run_id)
    if arts:
        at = Table(title="Artifacts")
        at.add_column("kind")
        at.add_column("path")
        at.add_column("sha256")
        for a in arts:
            at.add_row(a["kind"], a["path"][:60], (a.get("sha256") or "")[:12])
        console.print(at)

    if run.get("metrics_json"):
        console.print("\n[bold]Metrics:[/bold]")
        console.print(run["metrics_json"])


@serve_app.command("api")
def serve_api(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Hot-reload (dev only)."),
    engine: str = typer.Option(
        "echo", "--engine", "-e", help="Default engine: echo|ollama|transformers."
    ),
    set_overrides: list[str] | None = typer.Option(None, "--set"),
) -> None:
    """Run the FastAPI inference service."""
    import uvicorn

    from alignforge.core.config import load_config
    from alignforge.core.logging import setup_logging
    from alignforge.core.paths import get_paths

    cfg = load_config(overrides=(set_overrides or []) + [f"serve.default_engine={engine}"])
    paths = get_paths()
    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format, log_dir=paths.logs_dir)

    typer.echo(f"Starting API on http://{host}:{port} (engine={engine})")
    typer.echo(f"Docs:   http://{host}:{port}/docs")
    typer.echo(f"Health: http://{host}:{port}/health")

    from alignforge.serve.app import create_app

    app = create_app(max_concurrent=cfg.serve.concurrency_limit)

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_config=None,  # use structlog, not uvicorn's logging
    )


@serve_app.command("ui")
def serve_ui(
    api_base: str = typer.Option("http://localhost:8000", "--api-base"),
    port: int = typer.Option(7860, "--port"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link."),
) -> None:
    """Run the Gradio comparison UI."""
    from alignforge.core.config import load_config
    from alignforge.core.logging import setup_logging
    from alignforge.core.paths import get_paths
    from alignforge.ui.app import create_ui

    cfg = load_config()
    paths = get_paths()
    setup_logging(level=cfg.logging.level, fmt=cfg.logging.format, log_dir=paths.logs_dir)

    typer.echo(f"Starting Gradio UI at http://localhost:{port}")
    typer.echo(f"Connecting to API at {api_base}")

    demo = create_ui(api_base=api_base)
    demo.launch(  # type: ignore[call-arg]
        server_port=port,
        share=share,
        show_api=False,
        quiet=True,
    )


@serve_app.command("all")
def serve_all(
    api_port: int = typer.Option(8000, "--api-port"),
    ui_port: int = typer.Option(7860, "--ui-port"),
    engine: str = typer.Option("ollama", "--engine", "-e"),
    share: bool = typer.Option(False, "--share"),
) -> None:
    """Run API and UI together. Ctrl-C cleanly stops both."""
    import signal
    import subprocess
    import sys
    import time

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "alignforge.serve.app:create_app",
        "--factory",
        "--host",
        "0.0.0.0",
        "--port",
        str(api_port),
        "--log-config",
        "none",
    ]

    typer.echo(f"Starting API on http://localhost:{api_port}...")
    api_proc = subprocess.Popen(api_cmd)

    # Wait for API to be ready.
    import httpx

    for _ in range(20):
        time.sleep(0.5)
        try:
            httpx.get(f"http://localhost:{api_port}/health", timeout=1.0)
            break
        except Exception:
            pass

    typer.echo(f"Starting UI on http://localhost:{ui_port}...")
    from alignforge.ui.app import create_ui

    demo = create_ui(api_base=f"http://localhost:{api_port}")

    def _shutdown(sig: int, frame: Any) -> None:
        typer.echo("\nShutting down...")
        api_proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        demo.launch(  # type: ignore[call-arg]
            server_port=ui_port,
            share=share,
            show_api=False,
            quiet=True,
            prevent_thread_lock=False,
        )
    finally:
        api_proc.terminate()
