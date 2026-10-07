"""Tests for dedup and decontamination."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignforge.data.build import _publish_artifact
from alignforge.data.decontaminate import build_eval_ngram_index, decontaminate, is_contaminated
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

    def test_decontamination_is_word_level(self) -> None:
        """Shared short phrases are not contamination; a copied eval prompt is."""
        evals = [
            "Write a Python function that reverses a linked list in place without recursion please."
        ]
        index = build_eval_ngram_index(evals, n=13)
        assert not is_contaminated(
            "Write a Python program to sort the following list.", index, n=13
        )
        assert is_contaminated(evals[0], index, n=13)


def _stage(root: Path, content: str) -> Path:
    staged = root / ".staging"
    staged.mkdir(parents=True)
    (staged / "manifest.json").write_text(content)
    return staged


def test_publish_is_immutable_and_updates_latest(tmp_path: Path) -> None:
    first = _publish_artifact(_stage(tmp_path, "a"), tmp_path, "aaa111")
    second = _publish_artifact(_stage(tmp_path, "b"), tmp_path, "bbb222")
    assert first == tmp_path / "aaa111" and second == tmp_path / "bbb222"
    assert (first / "manifest.json").read_text() == "a"  # older build untouched
    assert (tmp_path / "latest_hash").read_text().strip() == "bbb222"
    assert not (tmp_path / ".staging").exists()


def test_republishing_same_hash_keeps_original(tmp_path: Path) -> None:
    _publish_artifact(_stage(tmp_path, "original"), tmp_path, "aaa111")
    _publish_artifact(_stage(tmp_path, "duplicate"), tmp_path, "aaa111")
    assert (tmp_path / "aaa111" / "manifest.json").read_text() == "original"
    assert not (tmp_path / ".staging").exists()


def test_resolve_dataset_rejects_overwritten_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry row pointing at overwritten data must not be trusted."""
    from types import SimpleNamespace

    from alignforge.core.errors import DataError
    from alignforge.train.run import resolve_dataset

    stale = tmp_path / "processed" / "ds" / "latest"  # now holds a different build
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text(json.dumps({"content_hash": "bbb"}))
    good = tmp_path / "processed" / "ds" / "aaa"
    good.mkdir()
    (good / "manifest.json").write_text(json.dumps({"content_hash": "aaa"}))

    fake_registry = SimpleNamespace(get_dataset=lambda h: {"path": str(stale)})
    monkeypatch.setattr("alignforge.core.registry.get_registry", lambda: fake_registry)
    paths = SimpleNamespace(data_dir=tmp_path)

    assert resolve_dataset(paths, "aaa") == good  # stale row skipped, immutable dir found
    with pytest.raises(DataError):
        resolve_dataset(paths, "ccc")  # nothing matches: refuse to train
