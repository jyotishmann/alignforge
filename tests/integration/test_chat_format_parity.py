"""Parity test: HuggingFace apply_chat_template vs Ollama Go template.

This is the most important test in the repository (see connector C10).
It must pass before any GGUF export. Run it after any model or template change.

Marked 'integration' because it downloads the Qwen2.5-1.5B tokenizer.
Run with: pytest tests/integration/test_chat_format_parity.py -m integration -v
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_chatformat_from_tokenizer() -> None:
    """ChatFormat builds correctly from the Qwen2.5-1.5B tokenizer."""
    from transformers import AutoTokenizer

    from alignforge.models.chat_format import ChatFormat

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=False)
    fmt = ChatFormat.from_tokenizer(tokenizer)

    assert len(fmt.stop_tokens) >= 1
    assert "<|im_end|>" in fmt.stop_tokens


@pytest.mark.integration
def test_chatformat_render_matches_tokenizer() -> None:
    """render() output matches tokenizer.apply_chat_template directly."""
    from transformers import AutoTokenizer

    from alignforge.models.chat_format import ChatFormat

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=False)
    fmt = ChatFormat.from_tokenizer(tokenizer)

    messages = [
        {"role": "user", "content": "How do I reverse a list in Python?"},
        {"role": "assistant", "content": "Use `list.reverse()` or `reversed()`."},
    ]

    via_format = fmt.render_training(messages)
    via_tokenizer = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    assert via_format == via_tokenizer, (
        "ChatFormat.render() does not match tokenizer.apply_chat_template.\n"
        f"ChatFormat output:\n{via_format!r}\n\n"
        f"Tokenizer output:\n{via_tokenizer!r}"
    )


@pytest.mark.integration
def test_ollama_template_parity() -> None:
    """The Ollama Go template produces the same prompt as the HF path.

    We substitute manually to simulate what Ollama renders at runtime,
    because we cannot execute Go templates in Python.
    """
    from transformers import AutoTokenizer

    from alignforge.models.chat_format import ChatFormat

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=False)
    fmt = ChatFormat.from_tokenizer(tokenizer)

    # Single user turn — tests the most critical path.
    user_prompt = "How do I sort a Python dictionary by value?"
    messages = [{"role": "user", "content": user_prompt}]

    # HF reference: the prompt that the model would receive at inference.
    hf_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Simulate Ollama's Go template rendering for the user turn.
    # The Ollama template (no system) should render to:
    #   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    stop = fmt.stop_tokens[0]
    expected_ollama_render = f"<|im_start|>user\n{user_prompt}{stop}\n" f"<|im_start|>assistant\n"

    assert hf_prompt == expected_ollama_render, (
        "HF prompt does not match the expected Ollama rendering.\n"
        "This means the GGUF model will receive a different prompt than evaluated.\n"
        f"\nHF output:          {hf_prompt!r}"
        f"\nExpected Ollama:    {expected_ollama_render!r}"
    )
