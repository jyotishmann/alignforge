"""Training stage unit tests — run on CPU with mocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestTrainingArgs:
    def test_fp16_on_non_bf16_hardware(self) -> None:
        """T4 (no bf16) should get fp16=True, bf16=False."""
        pytest.importorskip("transformers")
        from alignforge.core.config import AlignForgeConfig
        from alignforge.core.hardware import HardwareInfo
        from alignforge.train.sft import build_training_args

        cfg = AlignForgeConfig()
        mock_hw = HardwareInfo(
            device="cuda",
            device_name="Tesla T4",
            compute_capability=(7, 5),
            vram_total_gb=16.0,
            vram_free_gb=14.0,
            bf16_supported=False,
            recommended_dtype="float16",
        )
        with patch("alignforge.train.sft.probe_hardware", return_value=mock_hw):
            args = build_training_args(cfg, output_dir=Path("/tmp/test"))
        assert args.fp16 is True
        assert args.bf16 is False

    def test_bf16_on_ampere_hardware(self) -> None:
        """Ampere (bf16 available) should get fp16=False, bf16=True."""
        pytest.importorskip("transformers")
        from alignforge.core.config import AlignForgeConfig
        from alignforge.core.hardware import HardwareInfo
        from alignforge.train.sft import build_training_args

        cfg = AlignForgeConfig()
        mock_hw = HardwareInfo(
            device="cuda",
            device_name="A100",
            compute_capability=(8, 0),
            vram_total_gb=40.0,
            vram_free_gb=38.0,
            bf16_supported=True,
            recommended_dtype="bfloat16",
        )
        with patch("alignforge.train.sft.probe_hardware", return_value=mock_hw):
            args = build_training_args(cfg, output_dir=Path("/tmp/test"))
        assert args.fp16 is False
        assert args.bf16 is True


class TestVRAMCallback:
    def test_logs_every_n_steps(self) -> None:
        """VRAMCallback only logs on multiples of log_every_n_steps."""
        from alignforge.train.callbacks import VRAMCallback

        cb = VRAMCallback(log_every_n_steps=5)
        mock_state = MagicMock()
        mock_state.global_step = 5

        log_calls: list[dict[str, Any]] = []

        with patch("alignforge.train.callbacks.log") as mock_log:
            mock_log.info = lambda event, **kw: log_calls.append({"event": event, **kw})
            with patch("alignforge.train.callbacks.torch", create=True) as mt:
                mt.cuda.is_available.return_value = True
                mt.cuda.max_memory_allocated.return_value = 8 * 1024**3
                mt.cuda.get_device_properties.return_value.total_mem = 16 * 1024**3

                cb.on_step_end(None, mock_state, None)

        # Should have logged (step 5 is a multiple of 5).
        assert any(c.get("event") == "vram_step" for c in log_calls)


class TestResponseTemplate:
    def test_derives_chatML_template(self) -> None:
        """get_response_template returns the assistant start tokens."""
        from alignforge.models.chat_format import ChatFormat
        from alignforge.train.dataset import get_response_template

        # Minimal ChatML mock.
        mock_tok = MagicMock()
        mock_tok.chat_template = "..."
        mock_tok.all_special_tokens = ["<|im_end|>"]
        mock_tok.eos_token = "<|endoftext|>"
        mock_tok.name_or_path = "test"

        def mock_apply(messages: list[dict[str, str]], **kwargs: Any) -> str:
            result = ""
            for msg in messages:
                result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            if kwargs.get("add_generation_prompt"):
                result += "<|im_start|>assistant\n"
            return result

        mock_tok.apply_chat_template = mock_apply
        fmt = ChatFormat.from_tokenizer(mock_tok)
        template = get_response_template(fmt)

        # For ChatML, the response template should be the assistant turn start.
        assert "<|im_end|>" in template
        assert "<|im_start|>assistant" in template
