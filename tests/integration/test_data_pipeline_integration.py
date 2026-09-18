"""Data pipeline integration — synthetic dataset, no HuggingFace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignforge.data.schemas import PreferenceExample, SFTExample


def _make_sft_examples(n: int) -> list[SFTExample]:
    """Generate n synthetic SFT examples."""
    examples = []
    for i in range(n):
        examples.append(
            SFTExample(
                instruction=f"How do I use feature {i} in Python programming?",
                input=f"I am trying to do task {i} with Python code.",
                response=f"To use feature {i}, write: result = function_{i}(input_{i})",
                source="synthetic",
            )
        )
    return examples


def test_normalise_cleans_whitespace() -> None:
    from alignforge.data.normalise import normalise_examples

    messy = [
        SFTExample(
            instruction="How   do  I  sort   a   list?",
            response="Use  sorted()   or  list.sort()",
            source="test",
        )
    ]
    cleaned, quarantined = normalise_examples(messy)
    assert len(cleaned) == 1
    assert len(quarantined) == 0
    assert "  " not in cleaned[0].instruction


def test_dedup_removes_exact_copies() -> None:
    from alignforge.data.dedup import deduplicate

    examples = _make_sft_examples(5)
    examples_with_dupe = [*examples, examples[0]]  # one duplicate
    unique, stats = deduplicate(examples_with_dupe)
    assert len(unique) == 5
    assert stats["exact_removed"] == 1


def test_decontaminate_drops_eval_overlap() -> None:
    from alignforge.data.decontaminate import decontaminate

    eval_prompts = ["How do I reverse a list in Python programming language?"]
    train_examples = [
        SFTExample(
            instruction="How do I reverse a list in Python programming language?",
            response="Use reversed().",
            source="test",
        ),
        SFTExample(
            instruction="What is the capital of France?",
            response="Paris.",
            source="test",
        ),
    ]
    clean, dropped = decontaminate(train_examples, eval_prompts, n=13)
    assert dropped == 1
    assert len(clean) == 1
    assert clean[0].instruction == "What is the capital of France?"


def test_parquet_manifest_round_trip(tmp_path: Path) -> None:
    """Write → read a parquet artifact and verify the manifest content hash."""
    from alignforge.data.writer import write_dataset_artifact

    records = [
        {"instruction": f"Q{i}", "input": "", "response": f"A{i}", "source": "test"}
        for i in range(20)
    ]
    artifact_dir, content_hash = write_dataset_artifact(
        records=records,
        output_dir=tmp_path / "out",
        name="test_ds",
        kind="sft",
        val_ratio=0.1,
        funnel={"loaded": 20, "final": 20},
        seed=42,
    )

    # Verify files exist.
    assert (artifact_dir / "train.parquet").exists()
    assert (artifact_dir / "val.parquet").exists()
    assert (artifact_dir / "manifest.json").exists()

    # Verify manifest content.
    with (artifact_dir / "manifest.json").open() as f:
        manifest = json.load(f)

    assert manifest["content_hash"] == content_hash
    assert manifest["n_total"] == 20
    assert len(content_hash) == 12

    # Verify hash is reproducible.
    _, content_hash_2 = write_dataset_artifact(
        records=records,
        output_dir=tmp_path / "out2",
        name="test_ds",
        kind="sft",
        val_ratio=0.1,
        seed=42,
    )
    assert content_hash == content_hash_2


def test_preference_schema_validation() -> None:
    """Preference examples with identical chosen/rejected are rejected."""
    with pytest.raises(Exception, match="identical"):
        PreferenceExample(
            prompt="What is X?",
            chosen="Same answer.",
            rejected="Same answer.",
        )


def test_filter_stage1_keyword_match() -> None:
    """The keyword filter correctly identifies technical content."""
    from alignforge.data.filters import keyword_filter

    assert keyword_filter("How do I use Python generators?") is True
    assert keyword_filter("What is the best pizza topping?") is False
    assert keyword_filter("Fix this Docker container exit code 137") is True
    assert keyword_filter("Tell me a poem about autumn leaves") is False
