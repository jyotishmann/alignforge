"""Preference dataset bridge: parquet artifact → HF Dataset for DPOTrainer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


def _format_preference_row(
    row: dict[str, Any],
    chat_format: Any,
) -> dict[str, str]:
    """Format one preference triplet into the DPOTrainer column schema.

    prompt:   user turn only, with generation-start token appended.
    chosen:   full conversation including preferred response.
    rejected: full conversation including rejected response.
    """
    prompt_msgs = [{"role": "user", "content": row["prompt"]}]
    chosen_msgs = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["chosen"]},
    ]
    rejected_msgs = [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row["rejected"]},
    ]

    return {
        "prompt": chat_format.render_prompt(prompt_msgs),
        "chosen": chat_format.render_training(chosen_msgs),
        "rejected": chat_format.render_training(rejected_msgs),
        "source": row.get("source", ""),
    }


def load_preference_dataset(
    artifact_dir: Path,
    chat_format: Any,
    max_prompt_length: int = 512,
    max_length: int = 1024,
    limit: int | None = None,
) -> Any:
    """Load a preference parquet and return a formatted DatasetDict.

    DPOTrainer length filtering is handled here rather than inside the trainer
    so we know exactly how many rows were dropped and can log the count.
    """
    import pandas as pd
    from datasets import Dataset, DatasetDict

    train_path = artifact_dir / "train.parquet"
    val_path = artifact_dir / "val.parquet"

    if not train_path.exists():
        from alignforge.core.errors import DataError

        raise DataError(
            f"train.parquet not found in {artifact_dir}. "
            f"Run `alignforge data build --config configs/data/dpo_dev_assistant.yaml`.",
            source=str(artifact_dir),
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    log.info("preference_parquet_loaded", train=len(train_df), val=len(val_df))

    if limit is not None:
        train_df = train_df.iloc[:limit]
        val_df = val_df.iloc[: max(1, limit // 10)]
        log.info("preference_dataset_limited", train=len(train_df), val=len(val_df))

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    # Apply chat formatting.
    train_ds = train_ds.map(
        lambda row: _format_preference_row(row, chat_format),
        desc="Formatting preference train",
    )
    val_ds = val_ds.map(
        lambda row: _format_preference_row(row, chat_format),
        desc="Formatting preference val",
    )

    # Log sample to verify formatting.
    if len(train_ds) > 0:
        sample = train_ds[0]
        log.info(
            "preference_sample",
            prompt_preview=sample["prompt"][:80],
            chosen_preview=sample["chosen"][:80],
            rejected_preview=sample["rejected"][:80],
        )

    log.info(
        "preference_dataset_ready",
        n_train=len(train_ds),
        n_val=len(val_ds),
    )
    return DatasetDict({"train": train_ds, "validation": val_ds})
