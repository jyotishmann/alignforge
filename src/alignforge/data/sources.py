"""Source specifications — adapters from HF dataset schemas to our canonical schemas."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceSpec:
    """Declares how to load one HuggingFace dataset and map it to our schema."""

    name: str  # short label for logs/manifest
    hf_id: str  # e.g. "yahma/alpaca-cleaned"
    revision: str = "main"
    split: str = "train"
    kind: str = "sft"  # "sft" or "preference"
    field_map: dict[str, str] = field(default_factory=dict)
    # field_map: {our_field: their_field}
    # e.g. {"instruction": "instruction", "response": "output"}
    categories_keep: list[str] = field(default_factory=list)
    # If set, only keep rows where the "category" field is in this list.
    category_field: str = "category"


# ── SFT sources ─────────────────────────────────────────────────────────

ALPACA_CLEANED = SourceSpec(
    name="alpaca_cleaned",
    hf_id="yahma/alpaca-cleaned",
    kind="sft",
    field_map={"instruction": "instruction", "input": "input", "response": "output"},
)

DOLLY_15K = SourceSpec(
    name="dolly_15k",
    hf_id="databricks/databricks-dolly-15k",
    kind="sft",
    field_map={"instruction": "instruction", "input": "context", "response": "response"},
    categories_keep=[
        "closed_qa",
        "information_extraction",
        "general_qa",
        "classification",
        "summarization",
    ],
    category_field="category",
)

OASST1 = SourceSpec(
    name="oasst1",
    hf_id="OpenAssistant/oasst1",
    kind="sft",
    # OASST has a tree structure; the loader handles flattening.
    field_map={"instruction": "parent_text", "response": "text"},
)

# ── Preference sources ──────────────────────────────────────────────────

ULTRAFEEDBACK = SourceSpec(
    name="ultrafeedback",
    hf_id="HuggingFaceH4/ultrafeedback_binarized",
    kind="preference",
    split="train_prefs",
    field_map={"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
)

ANTHROPIC_HH = SourceSpec(
    name="anthropic_hh",
    hf_id="Anthropic/hh-rlhf",
    kind="preference",
    split="train",
    field_map={"prompt": "chosen", "chosen": "chosen", "rejected": "rejected"},
    # Anthropic HH stores full conversations; the loader parses them.
)

# ── Registries ──────────────────────────────────────────────────────────

SFT_SOURCES = {"alpaca_cleaned": ALPACA_CLEANED, "dolly_15k": DOLLY_15K, "oasst1": OASST1}
PREF_SOURCES = {"ultrafeedback": ULTRAFEEDBACK, "anthropic_hh": ANTHROPIC_HH}
ALL_SOURCES = {**SFT_SOURCES, **PREF_SOURCES}
