"""Evaluation generation unit tests — no GPU, no model download."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignforge.eval.schemas import EvalCase, GeneratedResponse, GenerationCheckpoint


class TestEvalCaseSchema:
    def test_valid_case(self) -> None:
        case = EvalCase(id="d01", prompt="How do I sort a list?", intent="concision test")
        assert case.id == "d01"
        assert case.difficulty == 2
        assert case.verifiable is False

    def test_verifiable_case_has_expected_answer(self) -> None:
        case = EvalCase(
            id="d02",
            prompt="Is set membership O(1)?",
            intent="factual",
            verifiable=True,
            expected_answer="Yes, average O(1).",
        )
        assert case.expected_answer == "Yes, average O(1)."

    def test_short_prompt_rejected(self) -> None:
        with pytest.raises(ValueError):
            EvalCase(id="x", prompt="ok", intent="test")


class TestSuiteLoader:
    def test_load_valid_suite(self, tmp_path: Path) -> None:
        from alignforge.eval.suites.loader import load_suite

        suite_file = tmp_path / "test_suite.jsonl"
        cases_data = [
            {"id": f"t{i:02d}", "prompt": f"Question {i} with enough content?", "intent": "test"}
            for i in range(5)
        ]
        with suite_file.open("w") as f:
            for c in cases_data:
                f.write(json.dumps(c) + "\n")

        cases = load_suite("test_suite", tmp_path)
        assert len(cases) == 5
        assert all(c.suite == "test_suite" for c in cases)

    def test_limit_respected(self, tmp_path: Path) -> None:
        from alignforge.eval.suites.loader import load_suite

        suite_file = tmp_path / "test_suite.jsonl"
        with suite_file.open("w") as f:
            for i in range(10):
                f.write(
                    json.dumps(
                        {
                            "id": f"t{i}",
                            "prompt": f"Question number {i} about Python?",
                            "intent": "t",
                        }
                    )
                    + "\n"
                )

        cases = load_suite("test_suite", tmp_path, limit=3)
        assert len(cases) == 3

    def test_missing_suite_returns_empty(self, tmp_path: Path) -> None:
        from alignforge.eval.suites.loader import load_suite

        cases = load_suite("nonexistent", tmp_path)
        assert cases == []


class TestCheckpointResume:
    def test_completed_cases_skipped(self) -> None:
        ckpt = GenerationCheckpoint(
            model_id="test",
            suite="s",
            total_cases=5,
            completed_ids=["d01", "d02"],
        )
        assert ckpt.is_done("d01") is True
        assert ckpt.is_done("d03") is False
        assert ckpt.n_remaining == 3

    def test_failed_cases_marked_done(self) -> None:
        ckpt = GenerationCheckpoint(
            model_id="test",
            suite="s",
            total_cases=5,
            failed_ids=["d04"],
        )
        assert ckpt.is_done("d04") is True
        assert ckpt.n_remaining == 4


class TestEchoGeneration:
    def test_echo_generator_returns_response(self, tmp_path: Path) -> None:
        """Echo backend produces a valid GeneratedResponse schema."""
        from unittest.mock import MagicMock

        from alignforge.eval.decode_params import EVAL_PARAMS
        from alignforge.eval.generation import ModelGenerator

        registry = MagicMock()
        registry.get_served_model.return_value = {
            "backend": "echo",
            "weights_ref": "echo",
            "model_id": "test_echo",
        }
        chat_format = MagicMock()

        gen = ModelGenerator(
            model_id="test_echo",
            registry=registry,
            chat_format=chat_format,
            params=EVAL_PARAMS,
        )
        case = EvalCase(
            id="d01", prompt="How do I reverse a list?", intent="test", suite="test_suite"
        )
        resp = gen.generate(case)
        assert isinstance(resp, GeneratedResponse)
        assert resp.case_id == "d01"
        assert resp.model_id == "test_echo"
        assert resp.error is None
        assert len(resp.response) > 0
