"""DPO training: DPOTrainer builder, reference policy, and training args."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from alignforge.core.config import AlignForgeConfig
from alignforge.core.hardware import probe_hardware

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


def build_dpo_training_args(
    cfg: AlignForgeConfig,
    output_dir: Path,
) -> Any:
    """Build the DPOConfig: TRL's TrainingArguments subclass that carries both the
    general training settings and the DPO-specific ones (beta, lengths, loss).

    Key differences from SFT:
    - LR one OOM lower (DPO diverges at SFT rates)
    - Larger warmup ratio (DPO is less stable early)
    - No group_by_length (DPOTrainer needs matched prompt/chosen/rejected lengths)
    """
    from trl import DPOConfig

    hw = probe_hardware()
    use_fp16 = hw.device == "cuda" and not hw.bf16_supported
    use_bf16 = hw.device == "cuda" and hw.bf16_supported

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.dpo.num_train_epochs,
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
        learning_rate=cfg.dpo.learning_rate,
        lr_scheduler_type=cfg.dpo.lr_scheduler_type,
        warmup_ratio=cfg.dpo.warmup_ratio,
        optim=cfg.dpo.optim,
        gradient_checkpointing=cfg.dpo.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=cfg.dpo.logging_steps,
        save_steps=cfg.dpo.save_steps,
        eval_steps=cfg.dpo.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=False,  # DPO: take the final checkpoint, not best-loss
        report_to="none",
        dataloader_num_workers=0,
        # DPO's collator reads the raw prompt/chosen/rejected columns, so keep them.
        remove_unused_columns=False,
        seed=cfg.project.seed,
        # ── DPO-specific ──────────────────────────────────────────────────
        beta=cfg.dpo.beta,
        # "sigmoid" is the standard DPO loss (Rafailov et al. 2023);
        # "ipo" (Gheshlaghi Azar et al. 2023) and "hinge" are alternatives.
        loss_type="sigmoid",
        max_prompt_length=cfg.dpo.max_prompt_length,
        max_length=cfg.dpo.max_length,
        label_pad_token_id=-100,  # padding never contributes to the loss
        is_encoder_decoder=False,
        # Reference log-probs come from the same model with the adapter disabled,
        # computed per batch rather than in a precompute pass.
        precompute_ref_log_probs=False,
    )

    log.info(
        "dpo_training_args",
        beta=cfg.dpo.beta,
        lr=cfg.dpo.learning_rate,
        fp16=use_fp16,
        bf16=use_bf16,
        effective_batch=cfg.dpo.per_device_train_batch_size * cfg.dpo.gradient_accumulation_steps,
    )
    return args


def build_dpo_trainer(
    cfg: AlignForgeConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    training_args: Any,
    callbacks: list[Any] | None = None,
) -> Any:
    """Build DPOTrainer with adapter-disable reference policy.

    training_args is the DPOConfig from build_dpo_training_args: in TRL 0.11 it
    holds both the general training settings and beta/lengths/loss_type.

    ref_model=None is the key: TRL uses disable_adapter() / enable_adapter()
    on the PEFT model to compute reference log-probs from the same object.
    This halves VRAM vs a two-model setup. Valid only when the SFT adapter
    is the starting point (so adapter-off == SFT checkpoint).

    See 00_MASTER.md top of Part 06 for the full explanation.
    """
    from trl import DPOTrainer

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # ← the memory trick
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        callbacks=callbacks or [],
    )

    log.info(
        "dpo_trainer_built",
        ref_model="adapter-disable (same model, adapter toggled)",
        beta=cfg.dpo.beta,
        n_train=len(dataset["train"]),
        n_val=len(dataset["validation"]),
    )
    return trainer
