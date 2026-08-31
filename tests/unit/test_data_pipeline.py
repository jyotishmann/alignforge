"""Tests for dedup and decontamination."""

from __future__ import annotations

from alignforge.data.decontaminate import decontaminate
from alignforge.data.dedup import exact_dedup
from alignforge.data.schemas import SFTExample


def _make(instruction: str, response: str = "Answer.") -> SFTExample:
    return SFTExample(instruction=instruction, response=response)


class TestExactDedup:
    def test_removes_exact_copies(self) -> None:
        examples = [_make("How to sort?"), _make("How to sort?"), _make("How to filter?")]
        unique, removed = exact_dedup(examples)
        assert len(unique) == 2
        assert removed == 1

    def test_preserves_order(self) -> None:
        examples = [_make("First"), _make("Second"), _make("First")]
        unique, _ = exact_dedup(examples)
        assert unique[0].instruction == "First"
        assert unique[1].instruction == "Second"


class TestDecontamination:
    def test_removes_overlapping_prompt(self) -> None:
        eval_prompts = ["How do I reverse a list in Python efficiently?"]
        examples = [
            _make("How do I reverse a list in Python efficiently?"),
            _make("What is the capital of France?"),
        ]
        clean, dropped = decontaminate(examples, eval_prompts, n=13)
        assert dropped == 1
        assert len(clean) == 1
        assert clean[0].instruction == "What is the capital of France?"

    def test_no_eval_prompts_is_noop(self) -> None:
        examples = [_make("Anything")]
        clean, dropped = decontaminate(examples, [], n=13)
        assert dropped == 0
        assert len(clean) == 1

    def test_short_text_not_falsely_dropped(self) -> None:
        """Text shorter than n should never be flagged."""
        eval_prompts = ["A very long evaluation prompt that has many words"]
        examples = [_make("Short text")]
        _, dropped = decontaminate(examples, eval_prompts, n=13)
        assert dropped == 0
