"""ChatFormat — single source of truth for chat template.

Four consumers:
  1. data/formatting.py (training data)
  2. train/ (SFT and DPO trainers)
  3. serve/engine/transformers.py (HF inference)
  4. export/ollama.py (Ollama Modelfile)

The parity test (test_chat_format_parity) asserts byte equality between
consumers 1/2/3 and 4. Run it after any model or template change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatFormat:
    """Wraps a tokenizer's chat template with a stable programmatic interface.

    Constructed once per process from a loaded tokenizer; passed to consumers.
    """

    model_name: str
    # The raw Jinja2 template string — stored for Ollama conversion.
    _jinja_template: str = field(repr=False)
    # Stop tokens that terminate generation (used by Ollama and the serving engines).
    stop_tokens: list[str] = field(default_factory=list)
    # The apply_chat_template callable from the tokenizer.
    _template_fn: Any = field(repr=False, default=None)

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> ChatFormat:
        """Build a ChatFormat from a loaded HuggingFace tokenizer."""
        if tokenizer.chat_template is None:
            from alignforge.core.errors import ModelError

            raise ModelError("Tokenizer has no chat_template; cannot construct ChatFormat.")

        # Derive stop tokens. For ChatML-format models (Qwen, Mistral-instruct):
        # the end-of-turn token is <|im_end|>. For LLaMA-3: <|eot_id|>.
        # We read from the special_tokens_map if available, else use a known list.
        stop: list[str] = []
        for candidate in ["<|im_end|>", "<|eot_id|>", "</s>", "<|endoftext|>"]:
            if candidate in tokenizer.all_special_tokens:
                stop.append(candidate)
        if not stop:
            stop = [tokenizer.eos_token]

        return cls(
            model_name=tokenizer.name_or_path,
            _jinja_template=tokenizer.chat_template,
            stop_tokens=stop,
            _template_fn=tokenizer.apply_chat_template,
        )

    def render(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
    ) -> str:
        """Render a list of messages to the model's chat format (as a string)."""
        if self._template_fn is None:
            raise RuntimeError("ChatFormat was constructed without a template_fn.")
        result = self._template_fn(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        return result  # type: ignore[no-any-return]

    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        """Render messages and append the generation prompt (for inference)."""
        return self.render(messages, add_generation_prompt=True)

    def render_training(self, messages: list[dict[str, str]]) -> str:
        """Render a complete conversation (no generation prompt, for SFT)."""
        return self.render(messages, add_generation_prompt=False)

    def to_ollama_template(self) -> str:
        """Convert the Jinja2 chat template to Ollama Go-template syntax.

        This handles the ChatML format used by Qwen2.5-Instruct and variants.

        The Ollama TEMPLATE block receives:
          .System    — system message (may be empty)
          .Prompt    — user message
          .Response  — assistant message (during generation, empty at first)
        """
        # Qwen2.5 uses ChatML format:
        #   <|im_start|>system\n{system}<|im_end|>\n
        #   <|im_start|>user\n{user}<|im_end|>\n
        #   <|im_start|>assistant\n{assistant}<|im_end|>\n
        # We hardcode this rather than parse Jinja2 generically.
        # If your model uses a different format, override this method.

        jinja = self._jinja_template
        stop = self.stop_tokens[0] if self.stop_tokens else "<|im_end|>"

        # Detect format by checking for characteristic tokens in the template.
        if "<|im_start|>" in jinja:
            # ChatML format (Qwen2.5, Mistral-Instruct-v0.2+, etc.)
            return self._chatml_ollama_template(stop)
        elif "<|begin_of_text|>" in jinja:
            # LLaMA-3 format
            return self._llama3_ollama_template()
        else:
            # Generic fallback.
            return self._generic_ollama_template(stop)

    def _chatml_ollama_template(self, stop: str) -> str:
        return (
            "{{ if .System }}"
            "<|im_start|>system\n{{ .System }}" + stop + "\n"
            "{{ end }}"
            "<|im_start|>user\n{{ .Prompt }}" + stop + "\n"
            "<|im_start|>assistant\n"
            "{{ .Response }}" + stop + "\n"
        )

    def _llama3_ollama_template(self) -> str:
        return (
            "<|begin_of_text|>"
            "{{ if .System }}"
            "<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>"
            "{{ end }}"
            "<|start_header_id|>user<|end_header_id|>\n\n{{ .Prompt }}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{{ .Response }}<|eot_id|>"
        )

    def _generic_ollama_template(self, stop: str) -> str:
        """Best-effort fallback. Validate with test_chat_format_parity."""
        return (
            "{{ if .System }}### System:\n{{ .System }}\n{{ end }}"
            "### User:\n{{ .Prompt }}\n"
            "### Assistant:\n{{ .Response }}" + stop + "\n"
        )

    def to_modelfile_block(self) -> str:
        """Return the full TEMPLATE block for an Ollama Modelfile."""
        ollama_template = self.to_ollama_template()
        stop_lines = "\n".join(f'PARAMETER stop "{s}"' for s in self.stop_tokens)
        return f'TEMPLATE """{ollama_template}"""\n{stop_lines}'


# ── Registry ─────────────────────────────────────────────────────────────

_formats: dict[str, ChatFormat] = {}


def register_format(name: str, fmt: ChatFormat) -> None:
    """Register a ChatFormat under a model name."""
    _formats[name] = fmt


def get_format(name: str) -> ChatFormat:
    """Retrieve a registered ChatFormat. Raises KeyError if not registered."""
    if name not in _formats:
        raise KeyError(
            f"No ChatFormat registered for {name!r}. "
            f"Call register_format() after loading the tokenizer."
        )
    return _formats[name]


def get_or_build_format(cfg: Any, tokenizer: Any) -> ChatFormat:
    """Get existing or build and register a new ChatFormat."""
    from alignforge.core.config import AlignForgeConfig

    key = cfg.model.name_or_path if isinstance(cfg, AlignForgeConfig) else str(cfg)
    if key not in _formats:
        fmt = ChatFormat.from_tokenizer(tokenizer)
        _formats[key] = fmt
    return _formats[key]
