"""Bridge from parquet artifacts to HuggingFace Dataset objects for SFTTrainer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


def _load_parquet(path: Path) -> Any:
    """Load a parquet file into a HuggingFace Dataset."""
    import pandas as pd
    from datasets import Dataset

    df = pd.read_parquet(path)
    log.info("parquet_loaded", path=str(path), n_rows=len(df))
    return Dataset.from_pandas(df, preserve_index=False)


def _format_sft_row(
    row: dict[str, Any],
    chat_template_fn: Any,
) -> dict[str, str]:
    """Convert raw instruction/input/response → formatted 'text' field."""
    from alignforge.data.formatting import format_sft_example

    text = format_sft_example(
        instruction=row.get("instruction", ""),
        input_text=row.get("input", ""),
        response=row.get("response", ""),
        chat_template_fn=chat_template_fn,
    )
    return {"text": text, "source": row.get("source", "")}


def load_sft_dataset(
    artifact_dir: Path,
    tokenizer: Any,
    max_seq_len: int = 1024,
    limit: int | None = None,
) -> Any:
    """Load a parquet artifact and return a formatted DatasetDict.

    Args:
        artifact_dir: path to the processed dataset directory (has train.parquet, val.parquet).
        tokenizer: the model tokenizer — used for apply_chat_template and length filtering.
        max_seq_len: sequences longer than this are dropped with a warning.
        limit: if set, truncate each split to the first N rows before formatting.
    """
    from datasets import DatasetDict

    train_path = artifact_dir / "train.parquet"
    val_path = artifact_dir / "val.parquet"

    if not train_path.exists():
        from alignforge.core.errors import DataError

        raise DataError(
            f"train.parquet not found in {artifact_dir}. Run `alignforge data build` first.",
            source=str(artifact_dir),
        )

    train_ds = _load_parquet(train_path)
    val_ds = _load_parquet(val_path)

    # Apply --limit early, before formatting.
    if limit is not None:
        train_ds = train_ds.select(range(min(limit, len(train_ds))))
        val_ds = val_ds.select(range(min(max(1, limit // 10), len(val_ds))))
        log.info("dataset_limited", train=len(train_ds), val=len(val_ds))

    # Apply chat template to every row.
    chat_fn = tokenizer.apply_chat_template
    train_ds = train_ds.map(
        lambda row: _format_sft_row(row, chat_fn),
        remove_columns=[c for c in train_ds.column_names if c not in ("source",)],
        desc="Formatting training examples",
    )
    val_ds = val_ds.map(
        lambda row: _format_sft_row(row, chat_fn),
        remove_columns=[c for c in val_ds.column_names if c not in ("source",)],
        desc="Formatting validation examples",
    )

    # Length filter — drop rows that exceed max_seq_len after formatting.
    def _is_short_enough(row: dict[str, Any]) -> bool:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        return len(ids) <= max_seq_len

    n_before_train = len(train_ds)
    train_ds = train_ds.filter(_is_short_enough, desc="Length filtering train")
    n_before_val = len(val_ds)
    val_ds = val_ds.filter(_is_short_enough, desc="Length filtering val")

    n_dropped_train = n_before_train - len(train_ds)
    n_dropped_val = n_before_val - len(val_ds)
    if n_dropped_train > 0 or n_dropped_val > 0:
        log.warning(
            "length_filter_applied",
            train_dropped=n_dropped_train,
            val_dropped=n_dropped_val,
            max_seq_len=max_seq_len,
        )

    log.info(
        "sft_dataset_ready",
        n_train=len(train_ds),
        n_val=len(val_ds),
    )
    return DatasetDict({"train": train_ds, "validation": val_ds})


def get_response_template(chat_format: Any) -> str:
    """Return the response template token sequence for completion-only masking.

    Derived from the ChatFormat's model so it stays in sync with C10.
    For ChatML (Qwen2.5, Mistral-instruct): '<|im_start|>assistant\\n'
    For LLaMA-3: '<|start_header_id|>assistant<|end_header_id|>\\n\\n'
    """
    # Render a single dummy user turn and inspect what precedes the assistant.
    # The portion after the user turn and before the response is the template.
    dummy_msgs = [
        {"role": "user", "content": "DUMMY_USER_CONTENT"},
        {"role": "assistant", "content": "DUMMY_ASSISTANT_CONTENT"},
    ]
    full = chat_format.render_training(dummy_msgs)

    # Find the split point: everything after the user content and before
    # the assistant content is the response template.
    user_end = full.find("DUMMY_USER_CONTENT") + len("DUMMY_USER_CONTENT")
    assistant_start = full.find("DUMMY_ASSISTANT_CONTENT")

    if user_end < 0 or assistant_start < 0 or assistant_start <= user_end:
        from alignforge.core.errors import ModelError

        raise ModelError(
            "Could not derive response template from ChatFormat. "
            "Inspect the rendered template manually and set it explicitly."
        )

    template = full[user_end:assistant_start]
    log.info("response_template_derived", template=repr(template))
    return str(template)


def verify_labels_not_all_masked(labels: list[int]) -> None:
    """Assert that at least some label tokens are real (not -100).

    Fails fast if the response template is wrong
    """
    real = sum(1 for lbl in labels if lbl != -100)
    if real == 0:
        from alignforge.core.errors import TrainingError

        raise TrainingError(
            "All label tokens are masked (-100). The response template does not "
            "match any position in the tokenised text. "
            "Check that get_response_template() returns the correct sequence for "
            "your model's chat format."
        )
