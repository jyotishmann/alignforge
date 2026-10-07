"""Training stage unit tests — run on CPU with mocks."""

from __future__ import annotations

import sys
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
        """Logs once per N steps, with peak/total converted to GB."""
        from alignforge.train.callbacks import VRAMCallback

        # The callback imports torch inside the method, so patch sys.modules.
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.return_value = 8 * 1024**3
        fake_torch.cuda.get_device_properties.return_value.total_memory = 16 * 1024**3

        cb = VRAMCallback(log_every_n_steps=5)
        state = MagicMock(global_step=5)
        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch("alignforge.train.callbacks.log") as mock_log,
        ):
            for _ in range(5):
                cb.on_step_end(None, state, None)

        events = [c.args[0] for c in mock_log.info.call_args_list]
        assert events.count("vram_step") == 1
        kwargs = mock_log.info.call_args.kwargs
        assert kwargs["peak_gb"] == 8.0
        assert kwargs["total_gb"] == 16.0
        assert kwargs["pct"] == 50.0


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


def test_dpo_training_args_are_a_dpo_config(tmp_path: Path) -> None:
    """DPOTrainer needs one DPOConfig carrying both training and DPO settings."""
    trl = pytest.importorskip("trl")
    from alignforge.core.config import load_config
    from alignforge.train.dpo import build_dpo_training_args

    root = Path(__file__).resolve().parents[2]
    cfg = load_config(component_path=root / "configs/train/dpo_qlora.yaml")
    args = build_dpo_training_args(cfg, output_dir=tmp_path)

    assert isinstance(args, trl.DPOConfig)
    assert args.beta == cfg.dpo.beta
    assert args.max_length == cfg.dpo.max_length
    assert args.remove_unused_columns is False


def test_wrap_callbacks_forwards_on_log() -> None:
    """on_log must reach our callbacks (the DPO KL metric and divergence guard need it)."""
    pytest.importorskip("transformers")
    from transformers import TrainerCallback

    from alignforge.train.callbacks import wrap_callbacks

    class Recorder:
        def __init__(self) -> None:
            self.logs: list[object] = []

        def on_log(self, args: object, state: object, control: object, **kw: object) -> None:
            self.logs.append(kw.get("logs"))

    rec = Recorder()
    (adapter,) = wrap_callbacks([rec])
    control = object()
    assert isinstance(adapter, TrainerCallback)
    assert adapter.on_log(None, None, control, logs={"loss": 1.0}) is control
    assert rec.logs == [{"loss": 1.0}]
    assert adapter.on_train_begin(None, None, control) is control  # unimplemented: no-op
