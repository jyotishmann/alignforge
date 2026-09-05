"""Model layer unit tests — run on CPU, no model download."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """A mock tokenizer with the minimum interface our code touches."""
    tok = MagicMock()
    tok.pad_token = None
    tok.pad_token_id = None
    tok.eos_token = "<|endoftext|>"
    tok.eos_token_id = 0
    tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tok.all_special_tokens = ["<|im_end|>", "<|endoftext|>"]
    tok.name_or_path = "test-model"

    def apply_chat_template(messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Minimal implementation of the ChatML template."""
        result = ""
        for msg in messages:
            result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        if kwargs.get("add_generation_prompt"):
            result += "<|im_start|>assistant\n"
        return result

    tok.apply_chat_template = apply_chat_template
    tok.vocab_size = 1000
    tok.padding_side = "right"
    return tok


@pytest.fixture
def mock_torch_model() -> MagicMock:
    """A mock model with a named_modules interface for target discovery."""
    torch = pytest.importorskip("torch")

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(64, 64, bias=False)
            self.v_proj = torch.nn.Linear(64, 64, bias=False)
            self.gate_proj = torch.nn.Linear(64, 64, bias=False)

    return ToyModel()


# ── Tests ──────────────────────────────────────────────────────────────


class TestTokenizerSetup:
    def test_pad_token_set_when_absent(self, mock_tokenizer: MagicMock) -> None:
        """Tokenizer without pad_token gets EOS assigned."""
        from alignforge.core.config import AlignForgeConfig
        from alignforge.models.loading import load_tokenizer

        cfg = AlignForgeConfig()
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "alignforge.models.loading.AutoTokenizer",
                MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer)),
            )
            tok = load_tokenizer(cfg)

        assert tok.pad_token == "<|endoftext|>"

    def test_chat_template_required(self, mock_tokenizer: MagicMock) -> None:
        """Missing chat_template raises ModelError before training begins."""
        from alignforge.core.config import AlignForgeConfig
        from alignforge.core.errors import ModelError
        from alignforge.models.loading import load_tokenizer

        mock_tokenizer.chat_template = None
        cfg = AlignForgeConfig()

        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(
                "alignforge.models.loading.AutoTokenizer",
                MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer)),
            )
            with pytest.raises(ModelError, match="chat_template"):
                load_tokenizer(cfg)


class TestChatFormat:
    def test_render_training(self, mock_tokenizer: MagicMock) -> None:
        from alignforge.models.chat_format import ChatFormat

        fmt = ChatFormat.from_tokenizer(mock_tokenizer)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        out = fmt.render_training(messages)
        assert "<|im_start|>user" in out
        assert "hello" in out
        assert "<|im_end|>" in out

    def test_render_prompt_adds_generation_token(self, mock_tokenizer: MagicMock) -> None:
        from alignforge.models.chat_format import ChatFormat

        fmt = ChatFormat.from_tokenizer(mock_tokenizer)
        messages = [{"role": "user", "content": "test"}]
        out = fmt.render_prompt(messages)
        assert out.endswith("<|im_start|>assistant\n")

    def test_stop_tokens_extracted(self, mock_tokenizer: MagicMock) -> None:
        from alignforge.models.chat_format import ChatFormat

        fmt = ChatFormat.from_tokenizer(mock_tokenizer)
        assert "<|im_end|>" in fmt.stop_tokens

    def test_ollama_template_contains_placeholder(self, mock_tokenizer: MagicMock) -> None:
        from alignforge.models.chat_format import ChatFormat

        fmt = ChatFormat.from_tokenizer(mock_tokenizer)
        ollama = fmt.to_ollama_template()
        assert "{{ .Prompt }}" in ollama
        assert "{{ .Response }}" in ollama


@pytest.mark.skipif("torch" not in sys.modules, reason="torch not installed")
class TestTargetModuleDiscovery:
    def test_discovers_standard_projections(self, mock_torch_model: Any) -> None:
        from alignforge.models.lora import discover_target_modules

        targets = discover_target_modules(mock_torch_model)
        assert "q_proj" in targets
        assert "v_proj" in targets
        assert "gate_proj" in targets

    def test_count_trainable_params(self, mock_torch_model: Any) -> None:
        from alignforge.models.lora import count_trainable_params

        result = count_trainable_params(mock_torch_model)
        assert result["total_params"] > 0
        assert 0 <= result["trainable_pct"] <= 100
