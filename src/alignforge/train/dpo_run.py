"""DPO training lifecycle: config → load SFT adapter → DPO train → registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from alignforge.core.config import AlignForgeConfig, config_hash
from alignforge.core.hardware import probe_hardware
from alignforge.core.paths import get_paths
from alignforge.core.seed import seed_everything

log = structlog.get_logger()


def run_dpo(
    cfg: AlignForgeConfig,
    sft_run_id: str,
    preference_dataset_hash: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> str:
    """Execute a DPO alignment run on top of an SFT checkpoint. Returns run_id."""
    from alignforge.core.registry import get_registry
    from alignforge.train.run import _get_git_sha, _make_run_id

    paths = get_paths()
    hw = probe_hardware()
    c_hash = config_hash(cfg)
    run_id = _make_run_id("dpo", c_hash)

    log.info(
        "dpo_run_starting",
        run_id=run_id,
        sft_run_id=sft_run_id,
        config_hash=c_hash,
        beta=cfg.dpo.beta,
        dry_run=dry_run,
    )

    if dry_run:
        _print_dpo_dry_run(cfg, hw, run_id, c_hash, sft_run_id, preference_dataset_hash)
        return run_id

    seed_everything(cfg.project.seed)

    reg = get_registry()
    reg.create_run(
        run_id=run_id,
        kind="dpo",
        config_hash=c_hash,
        dataset_hash=preference_dataset_hash,
        git_sha=_get_git_sha(),
        hardware=hw.to_dict(),
        notes=f"init_from_sft={sft_run_id}",
    )

    output_dir = paths.artifacts_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from alignforge.core.logging import run_id_var

        token = run_id_var.set(run_id)

        adapter_path = _train_dpo(
            cfg=cfg,
            run_id=run_id,
            sft_run_id=sft_run_id,
            output_dir=output_dir,
            preference_hash=preference_dataset_hash,
            limit=limit,
        )

        reg.record_artifact(run_id=run_id, kind="lora_adapter", path=str(adapter_path))
        reg.complete_run(run_id)
        log.info("dpo_run_complete", run_id=run_id, adapter=str(adapter_path))
        return run_id

    except Exception as exc:
        reg.fail_run(run_id, error=str(exc))
        log.error("dpo_run_failed", run_id=run_id, error=str(exc))
        raise
    finally:
        if "token" in dir():
            run_id_var.reset(token)


def _train_dpo(
    cfg: AlignForgeConfig,
    run_id: str,
    sft_run_id: str,
    output_dir: Path,
    preference_hash: str,
    limit: int | None,
) -> Path:
    """Inner DPO training. Isolated for clean lifecycle wrapping."""
    from alignforge.models.chat_format import get_or_build_format
    from alignforge.models.loading import build_bnb_config, load_base_model, load_tokenizer

    # from alignforge.models.lora import build_lora_model
    from alignforge.train.callbacks import wrap_callbacks
    from alignforge.train.dpo import build_dpo_trainer, build_dpo_training_args
    from alignforge.train.dpo_callbacks import DivergenceGuardCallback, DPOMetricsCallback
    from alignforge.train.dpo_dataset import load_preference_dataset

    paths = get_paths()

    # 1. Tokenizer.
    tokenizer = load_tokenizer(cfg, padding_side="left")  # left for generation-like DPO
    chat_format = get_or_build_format(cfg, tokenizer)

    # 2. Load preference dataset.
    artifact_dir = _resolve_pref_dataset(paths, preference_hash, cfg)
    dataset = load_preference_dataset(
        artifact_dir=artifact_dir,
        chat_format=chat_format,
        max_prompt_length=cfg.dpo.max_prompt_length,
        max_length=cfg.dpo.max_length,
        limit=limit,
    )

    # 3. Load the SFT adapter checkpoint as the starting point.
    #    DPO will add a new LoRA adapter on top. With ref_model=None,
    #    the adapter-off state IS π_ref (the SFT policy).
    sft_adapter_path = _resolve_sft_adapter(paths, sft_run_id)
    log.info("loading_sft_adapter", path=str(sft_adapter_path))

    bnb_config = build_bnb_config(cfg)
    base_model = load_base_model(cfg, bnb_config)

    # Load SFT adapter weights into the base model.
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base_model,
        str(sft_adapter_path),
        is_trainable=True,  # make these weights trainable for DPO
    )
    log.info(
        "sft_adapter_loaded",
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # 4. Build training args and callbacks.
    training_args = build_dpo_training_args(cfg, output_dir=output_dir / "checkpoints")

    dpo_metrics_cb = DPOMetricsCallback(beta=cfg.dpo.beta)
    divergence_cb = DivergenceGuardCallback(beta=cfg.dpo.beta)

    callbacks = wrap_callbacks([dpo_metrics_cb, divergence_cb])

    # 5. Build and train.
    trainer = build_dpo_trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        training_args=training_args,
        callbacks=callbacks,
    )

    trainer.train()

    # 6. Save the DPO adapter.
    adapter_path = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # 7. Log final DPO metrics summary.
    log.info(
        "dpo_final_metrics",
        run_id=run_id,
        mean_implicit_kl=round(dpo_metrics_cb.mean_kl(), 4),
    )
    return adapter_path


def _resolve_pref_dataset(paths: Any, pref_hash: str, cfg: AlignForgeConfig) -> Path:
    """Find the preference dataset directory by content hash."""
    import json

    from alignforge.core.registry import get_registry

    reg = get_registry()
    row = reg.get_dataset(pref_hash)
    if row:
        return Path(row["path"])

    base = paths.data_dir / "processed" / cfg.data.name
    if base.exists():
        for child in base.iterdir():
            manifest = child / "manifest.json"
            if manifest.exists():
                with manifest.open() as f:
                    m = json.load(f)
                if m.get("content_hash") == pref_hash:
                    return Path(child)

    from alignforge.core.errors import DataError

    raise DataError(
        f"Preference dataset with hash {pref_hash!r} not found. "
        f"Run `alignforge data build --config configs/data/dpo_dev_assistant.yaml`.",
        source=pref_hash,
    )


def _resolve_sft_adapter(paths: Any, sft_run_id: str) -> Path:
    """Find the SFT adapter directory from the run ID."""
    from alignforge.core.errors import RegistryError
    from alignforge.core.registry import get_registry

    reg = get_registry()
    artifacts = reg.get_artifacts(sft_run_id)
    for art in artifacts:
        if art["kind"] == "lora_adapter":
            p = Path(art["path"])
            if p.exists():
                return p

    # Fallback: scan artifacts dir directly.
    candidate = paths.artifacts_dir / sft_run_id / "adapter"
    if candidate.exists():
        return Path(candidate)

    raise RegistryError(
        f"SFT adapter for run {sft_run_id!r} not found. "
        f"Check `alignforge registry list --kind sft`.",
        run_id=sft_run_id,
    )


def _print_dpo_dry_run(
    cfg: AlignForgeConfig,
    hw: Any,
    run_id: str,
    c_hash: str,
    sft_run_id: str,
    pref_hash: str,
) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"DPO dry run: {run_id}")
    table.add_column("Setting")
    table.add_column("Value")
    for k, v in [
        ("run_id", run_id),
        ("sft_run_id (init from)", sft_run_id),
        ("preference_hash", pref_hash),
        ("config_hash", c_hash),
        ("beta", str(cfg.dpo.beta)),
        ("learning_rate", str(cfg.dpo.learning_rate)),
        ("ref_model", "adapter-disable (same model)"),
        ("device", hw.device_name),
        ("bf16_supported", str(hw.bf16_supported)),
        (
            "effective_batch",
            str(cfg.dpo.per_device_train_batch_size * cfg.dpo.gradient_accumulation_steps),
        ),
    ]:
        table.add_row(k, v)
    console.print(table)
    console.print("[yellow]Dry run — no model loaded.[/yellow]")
