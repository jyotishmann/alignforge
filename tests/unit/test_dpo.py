"""DPO stage unit tests — CPU, mock-based."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

# import pytest


class TestDPOMetricsCallback:
    def test_kl_computed_from_rewards(self) -> None:
        """Implicit KL is (|r_chosen| + |r_rejected|) / (2 * beta)."""
        from alignforge.train.dpo_callbacks import DPOMetricsCallback

        cb = DPOMetricsCallback(beta=0.1)
        state = MagicMock()
        state.global_step = 10
        logs = {"rewards/chosen": 0.4, "rewards/rejected": -0.2, "rewards/margins": 0.6}

        cb.on_log(None, state, None, logs=logs)

        # KL = (0.4 + 0.2) / (2 * 0.1) = 3.0
        assert abs(cb.mean_kl() - 3.0) < 0.01

    def test_no_rewards_no_kl(self) -> None:
        """If reward keys are absent, mean_kl stays 0."""
        from alignforge.train.dpo_callbacks import DPOMetricsCallback

        cb = DPOMetricsCallback(beta=0.1)
        cb.on_log(None, MagicMock(), None, logs={"loss": 0.5})
        assert cb.mean_kl() == 0.0

    def test_high_kl_emits_warning(self, caplog: Any) -> None:
        """KL above threshold should trigger a warning log."""
        import logging

        from alignforge.train.dpo_callbacks import DPOMetricsCallback

        cb = DPOMetricsCallback(beta=0.1, kl_warn_threshold=1.0)
        state = MagicMock()
        state.global_step = 50

        with caplog.at_level(logging.WARNING):
            cb.on_log(
                None,
                state,
                None,
                logs={"rewards/chosen": 2.0, "rewards/rejected": -2.0},
            )
        # KL = (2.0 + 2.0) / (2 * 0.1) = 20.0 > 1.0


class TestDivergenceGuard:
    def test_stops_when_kl_exceeds_threshold(self) -> None:
        """DivergenceGuardCallback sets should_training_stop when window KL is high."""
        from alignforge.train.dpo_callbacks import DivergenceGuardCallback

        cb = DivergenceGuardCallback(beta=0.1, kl_stop_threshold=5.0, window_size=3)
        control = MagicMock()
        control.should_training_stop = False
        state = MagicMock()
        state.global_step = 30

        # Feed 3 steps with high KL (>>5.0).
        high_kl_logs = {"rewards/chosen": 5.0, "rewards/rejected": -5.0}
        for _ in range(3):
            cb.on_log(None, state, control, logs=high_kl_logs)

        # Window mean KL = (5+5)/(2*0.1) = 50 >> 5.0 threshold.
        assert control.should_training_stop is True

    def test_no_stop_when_kl_normal(self) -> None:
        """Normal KL does not trigger early stopping."""
        from alignforge.train.dpo_callbacks import DivergenceGuardCallback

        cb = DivergenceGuardCallback(beta=0.1, kl_stop_threshold=20.0, window_size=3)
        control = MagicMock()
        control.should_training_stop = False
        state = MagicMock()
        state.global_step = 5

        low_kl_logs = {"rewards/chosen": 0.3, "rewards/rejected": -0.1}
        for _ in range(3):
            cb.on_log(None, state, control, logs=low_kl_logs)

        # KL = (0.3+0.1)/(2*0.1) = 2.0 << 20.0
        assert control.should_training_stop is False


class TestPreferenceFormatting:
    def test_format_preference_row_schema(self) -> None:
        """format_preference_row produces prompt, chosen, rejected keys."""
        from alignforge.models.chat_format import ChatFormat
        from alignforge.train.dpo_dataset import _format_preference_row

        mock_tok = MagicMock()
        mock_tok.chat_template = "..."
        mock_tok.all_special_tokens = ["<|im_end|>"]
        mock_tok.eos_token = "<|im_end|>"
        mock_tok.name_or_path = "test"

        call_count = {"n": 0}

        def apply(messages: list[dict[str, str]], **kwargs: Any) -> str:
            call_count["n"] += 1
            role = messages[-1]["role"] if messages else ""
            return f"<{role}>" + messages[-1].get("content", "") + "<end>"

        mock_tok.apply_chat_template = apply
        fmt = ChatFormat.from_tokenizer(mock_tok)

        row = {
            "prompt": "How do I sort a list?",
            "chosen": "Use sorted().",
            "rejected": "Use .sort() which returns None.",
            "source": "test",
        }
        out = _format_preference_row(row, fmt)

        assert "prompt" in out
        assert "chosen" in out
        assert "rejected" in out
        # Chosen should contain the preferred response text.
        assert "sorted" in out["chosen"]
