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
    """Build TrainingArguments for DPO.

    Key differences from SFT:
    - LR one OOM lower (DPO diverges at SFT rates)
    - Larger warmup ratio (DPO is less stable early)
    - No group_by_length (DPOTrainer needs matched prompt/chosen/rejected lengths)
    """
    from transformers import TrainingArguments

    hw = probe_hardware()
    use_fp16 = hw.device == "cuda" and not hw.bf16_supported
    use_bf16 = hw.device == "cuda" and hw.bf16_supported

    args = TrainingArguments(  # type: ignore[call-arg]
        output_dir=str(output_dir),
        num_train_epochs=cfg.dpo.num_train_epochs,
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
        learning_rate=cfg.dpo.learning_rate,
        lr_scheduler_type=cfg.dpo.lr_scheduler_type,
        warmup_ratio=cfg.dpo.warmup_ratio,
        optim=cfg.dpo.optim,
        gradient_checkpointing=cfg.dpo.gradient_checkpointing,
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
        remove_unused_columns=False,
        # group_by_length=False,        # must be off: paired examples need stable lengths
        seed=cfg.project.seed,
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

    ref_model=None is the key: TRL uses disable_adapter() / enable_adapter()
    on the PEFT model to compute reference log-probs from the same object.
    This halves VRAM vs a two-model setup. Valid only when the SFT adapter
    is the starting point (so adapter-off == SFT checkpoint).

    See 00_MASTER.md top of Part 06 for the full explanation.
    """
    from trl import DPOConfig, DPOTrainer

    dpo_config = DPOConfig(
        beta=cfg.dpo.beta,
        max_prompt_length=cfg.dpo.max_prompt_length,
        max_length=cfg.dpo.max_length,
        # loss_type="sigmoid" is the standard DPO loss (Rafailov et al. 2023).
        # "ipo" (Gheshlaghi Azar et al. 2023) and "hinge" are alternatives.
        loss_type="sigmoid",
        # Compute the reference log-probs from the same model with adapter disabled.
        is_encoder_decoder=False,
        # label_pad_token_id: use -100 so padding tokens don't contribute to loss.
        label_pad_token_id=-100,
        # The reference model is NONE — adapter-disable trick.
        precompute_ref_log_probs=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # ← the memory trick
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        callbacks=callbacks or [],
        **dpo_config.__dict__,  # spread DPOConfig fields into DPOTrainer
    )

    log.info(
        "dpo_trainer_built",
        ref_model="adapter-disable (same model, adapter toggled)",
        beta=cfg.dpo.beta,
        n_train=len(dataset["train"]),
        n_val=len(dataset["validation"]),
    )
    return trainer
