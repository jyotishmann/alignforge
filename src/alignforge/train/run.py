"""Outer SFT training lifecycle: config → registry → train → complete/fail."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from alignforge.core.config import AlignForgeConfig, config_hash
from alignforge.core.hardware import probe_hardware
from alignforge.core.paths import get_paths
from alignforge.core.seed import seed_everything

log = structlog.get_logger()


def run_sft(
    cfg: AlignForgeConfig,
    dataset_hash: str,
    limit: int | None = None,
    dry_run: bool = False,
    resume_from: str | None = None,
) -> str:
    """Execute a complete SFT training run. Returns the run_id."""
    from alignforge.core.registry import get_registry

    paths = get_paths()
    hw = probe_hardware()
    c_hash = config_hash(cfg)
    run_id = _make_run_id("sft", c_hash)

    log.info("sft_run_starting", run_id=run_id, config_hash=c_hash, dry_run=dry_run)

    if dry_run:
        _print_dry_run_summary(cfg, hw, run_id, c_hash, dataset_hash)
        return run_id

    # Seed before any model or data initialisation.
    seed_everything(cfg.project.seed)

    # Register the run.
    reg = get_registry()
    git_sha = _get_git_sha()
    reg.create_run(
        run_id=run_id,
        kind="sft",
        config_hash=c_hash,
        dataset_hash=dataset_hash,
        git_sha=git_sha,
        hardware=hw.to_dict(),
    )

    output_dir = paths.artifacts_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Set run_id in logging context — all log lines inside this block carry it.
        from alignforge.core.logging import run_id_var

        token = run_id_var.set(run_id)

        adapter_path = _train(
            cfg=cfg,
            run_id=run_id,
            output_dir=output_dir,
            dataset_hash=dataset_hash,
            limit=limit,
            resume_from=resume_from,
        )

        # Register the adapter artifact.
        reg.record_artifact(
            run_id=run_id,
            kind="lora_adapter",
            path=str(adapter_path),
        )
        reg.complete_run(run_id)
        log.info("sft_run_complete", run_id=run_id, adapter=str(adapter_path))
        return run_id

    except Exception as exc:
        reg.fail_run(run_id, error=str(exc))
        log.error("sft_run_failed", run_id=run_id, error=str(exc))
        raise
    finally:
        if "token" in dir():
            run_id_var.reset(token)


def _train(
    cfg: AlignForgeConfig,
    run_id: str,
    output_dir: Path,
    dataset_hash: str,
    limit: int | None,
    resume_from: str | None,
) -> Path:
    """Inner train function — isolated so run_sft can wrap it cleanly."""
    from alignforge.models.chat_format import get_or_build_format
    from alignforge.models.loading import build_bnb_config, load_base_model, load_tokenizer
    from alignforge.models.lora import build_lora_model
    from alignforge.train.callbacks import ProbeGenerationCallback, VRAMCallback
    from alignforge.train.dataset import (
        get_response_template,
        load_sft_dataset,
    )
    from alignforge.train.sft import build_sft_trainer, build_training_args

    paths = get_paths()

    # 1. Load tokenizer (training: right-padding).
    tokenizer = load_tokenizer(cfg, padding_side="right")
    chat_format = get_or_build_format(cfg, tokenizer)

    # 2. Load and format the dataset.
    artifact_dir = _resolve_dataset(paths, dataset_hash, cfg)
    dataset = load_sft_dataset(
        artifact_dir=artifact_dir,
        tokenizer=tokenizer,
        max_seq_len=cfg.model.max_seq_len,
        limit=limit,
    )

    # 3. Load model.
    bnb_config = build_bnb_config(cfg)
    base_model = load_base_model(cfg, bnb_config)
    model = build_lora_model(cfg, base_model)

    # 4. Training args.
    training_args = build_training_args(cfg, output_dir=output_dir / "checkpoints")

    # 5. Response template and collator.
    response_template = get_response_template(chat_format)

    # 6. Callbacks.
    probe_prompts = _load_probe_prompts(paths)
    callbacks = [
        VRAMCallback(log_every_n_steps=cfg.sft.logging_steps),
        ProbeGenerationCallback(
            prompts=probe_prompts,
            chat_format=chat_format,
            tokenizer=tokenizer,
            every_n_steps=cfg.sft.eval_steps,
        ),
    ]

    # 7. Build and train.
    trainer = build_sft_trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        response_template=response_template,
        callbacks=callbacks,
    )

    checkpoint = _find_checkpoint(output_dir / "checkpoints", resume_from)
    trainer.train(resume_from_checkpoint=checkpoint)

    # 8. Save the final adapter.
    adapter_path = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    return adapter_path


def _resolve_dataset(paths: Any, dataset_hash: str, cfg: AlignForgeConfig) -> Path:
    """Find the dataset artifact directory by its content hash."""
    import json

    from alignforge.core.registry import get_registry

    reg = get_registry()
    row = reg.get_dataset(dataset_hash)
    if row:
        return Path(row["path"])

    # Fallback: scan the data directory for a manifest with matching hash.
    base = paths.data_dir / "processed" / cfg.data.name
    if base.exists():
        for child in base.iterdir():
            manifest = child / "manifest.json"
            if manifest.exists():
                with manifest.open() as f:
                    m = json.load(f)
                if m.get("content_hash") == dataset_hash:
                    return Path(child)
    from alignforge.core.errors import DataError

    raise DataError(
        f"Dataset with hash {dataset_hash!r} not found. Run `alignforge data build` to produce it.",
        source=dataset_hash,
    )


def _make_run_id(kind: str, c_hash: str) -> str:
    from alignforge.core.registry import Registry

    return Registry.make_run_id(kind, c_hash)


def _get_git_sha() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _load_probe_prompts(paths: Any) -> list[str]:
    """Load the first 5 prompts from the domain eval suite for probe generation."""
    import json

    probe_path = paths.evals_dir / "domain_v1.jsonl"
    prompts: list[str] = []
    if probe_path.exists():
        with probe_path.open() as f:
            for line in f:
                row = json.loads(line)
                prompts.append(row.get("prompt", ""))
                if len(prompts) >= 5:
                    break
    if not prompts:
        prompts = ["Explain what a Python generator is."]
    return prompts


def _find_checkpoint(checkpoint_dir: Path, resume_from: str | None) -> str | None:
    """Find the latest checkpoint directory, or a specific one if named."""
    if not checkpoint_dir.exists():
        return None
    if resume_from:
        specific = checkpoint_dir / resume_from
        if specific.exists():
            return str(specific)
        log.warning("checkpoint_not_found", requested=resume_from)
        return None
    # Find the latest: checkpoint-NNNN directories sorted numerically.
    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0,
    )
    return str(checkpoints[-1]) if checkpoints else None


def _print_dry_run_summary(
    cfg: AlignForgeConfig,
    hw: Any,
    run_id: str,
    c_hash: str,
    dataset_hash: str,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Dry run: {run_id}")
    table.add_column("Setting")
    table.add_column("Value")
    rows = [
        ("run_id", run_id),
        ("config_hash", c_hash),
        ("dataset_hash", dataset_hash),
        ("model", cfg.model.name_or_path),
        ("device", hw.device_name),
        ("vram_free_gb", str(hw.vram_free_gb)),
        ("bf16_available", str(hw.bf16_supported)),
        (
            "effective_batch",
            str(cfg.sft.per_device_train_batch_size * cfg.sft.gradient_accumulation_steps),
        ),
        ("learning_rate", str(cfg.sft.learning_rate)),
        ("epochs", str(cfg.sft.num_train_epochs)),
        ("lora_r", str(cfg.lora.r)),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    console.print("[yellow]Dry run — no model loaded, no training executed.[/yellow]")
