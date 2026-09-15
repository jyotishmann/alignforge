"""SFT training: TrainingArguments, SFTTrainer wiring, and the train() call."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from alignforge.core.config import AlignForgeConfig
from alignforge.core.hardware import probe_hardware

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


def build_training_args(
    cfg: AlignForgeConfig,
    output_dir: Path,
) -> Any:
    """Build HuggingFace TrainingArguments from the typed config.

    Dtype flags are driven by the hardware probe — never hardcoded.
    """
    from transformers import TrainingArguments

    hw = probe_hardware()
    use_fp16 = hw.device == "cuda" and not hw.bf16_supported
    use_bf16 = hw.device == "cuda" and hw.bf16_supported

    args = TrainingArguments(  # type: ignore[call-arg]
        output_dir=str(output_dir),
        num_train_epochs=cfg.sft.num_train_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        learning_rate=cfg.sft.learning_rate,
        lr_scheduler_type=cfg.sft.lr_scheduler_type,
        warmup_ratio=cfg.sft.warmup_ratio,
        weight_decay=cfg.sft.weight_decay,
        max_grad_norm=cfg.sft.max_grad_norm,
        optim=cfg.sft.optim,
        gradient_checkpointing=cfg.sft.gradient_checkpointing,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=cfg.sft.logging_steps,
        save_steps=cfg.sft.save_steps,
        eval_steps=cfg.sft.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,  # keep only 3 latest checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # no wandb/tensorboard by default
        dataloader_num_workers=0,  # 0 is safer on Colab (fork issues)
        remove_unused_columns=False,  # we manage columns ourselves
        # group_by_length=True,  # batch similar lengths → less padding waste
        ddp_find_unused_parameters=False,  # not DDP but avoids a warning
        seed=cfg.project.seed,
    )

    log.info(
        "training_args_built",
        fp16=use_fp16,
        bf16=use_bf16,
        effective_batch=cfg.sft.per_device_train_batch_size * cfg.sft.gradient_accumulation_steps,
    )
    return args


def build_sft_trainer(
    cfg: AlignForgeConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    training_args: Any,
    response_template: str,
    callbacks: list[Any] | None = None,
) -> Any:
    """Build the SFTTrainer with completion-only loss masking."""
    from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

    # Completion-only collator — masks instruction tokens to -100.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,  # causal LM, not masked LM
    )

    # Verify the collator works on a sample before committing to training.
    _verify_collator(collator, dataset["train"], tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg.model.max_seq_len,
        packing=cfg.sft.packing,
        callbacks=callbacks or [],
    )
    return trainer


def _verify_collator(
    collator: Any,
    train_dataset: Any,
    tokenizer: Any,
) -> None:
    """Run the collator on the first example and check labels are not all -100."""
    sample = [train_dataset[0]]
    batch = collator(sample)
    labels = batch["labels"][0].tolist()
    real_count = sum(1 for lbl in labels if lbl != -100)

    if real_count == 0:
        from alignforge.core.errors import TrainingError

        raise TrainingError(
            f"Completion-only collator masked ALL label tokens in the first example. "
            f"The response template does not appear in any tokenised position. "
            f"Text preview: {train_dataset[0].get('text', '')[:200]!r}"
        )

    log.info(
        "collator_verified",
        total_tokens=len(labels),
        real_label_tokens=real_count,
        masked_tokens=len(labels) - real_count,
    )
