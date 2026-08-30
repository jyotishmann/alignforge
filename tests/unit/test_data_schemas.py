"""Dataset schema validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alignforge.data.schemas import PreferenceExample, SFTExample


class TestSFTExample:
    def test_valid(self) -> None:
        ex = SFTExample(instruction="How do I sort a list?", response="Use sorted().")
        assert ex.source == ""
        assert "sort" in ex.prompt_text

    def test_blank_instruction_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SFTExample(instruction="   ", response="Something valid.")

    def test_short_response_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SFTExample(instruction="A valid question here", response="ok")

    def test_input_included_in_prompt_text(self) -> None:
        ex = SFTExample(
            instruction="Summarise this", input="Long article text...", response="Summary."
        )
        assert "Long article" in ex.prompt_text


class TestPreferenceExample:
    def test_valid(self) -> None:
        ex = PreferenceExample(
            prompt="What is Python?",
            chosen="Python is a programming language.",
            rejected="Python is a type of snake.",
        )
        assert "Python" in ex.full_text

    def test_identical_chosen_rejected_fails(self) -> None:
        with pytest.raises(ValidationError, match="identical"):
            PreferenceExample(
                prompt="What is X?",
                chosen="Same answer here.",
                rejected="Same answer here.",
            )

    def test_whitespace_identical_fails(self) -> None:
        """Identical after strip should also fail."""
        with pytest.raises(ValidationError, match="identical"):
            PreferenceExample(
                prompt="What is X?",
                chosen="  Same answer  ",
                rejected="Same answer",
            )
