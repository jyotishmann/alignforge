"""Pinned chat format test — golden file for the Ollama template output.

This test downloads the Qwen2.5-1.5B tokenizer (~50MB) on first run.
Subsequent runs use the HuggingFace cache.

The expected values below are set once from the first successful run.
If they change, investigate before updating — a change here means the
GGUF model will receive different prompts than the evaluated checkpoint.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_qwen_stop_tokens_are_pinned() -> None:
    """The stop token list must match the expected Qwen2.5 ChatML tokens."""
    from transformers import AutoTokenizer

    from alignforge.models.chat_format import ChatFormat

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    fmt = ChatFormat.from_tokenizer(tok)

    # This is the pinned expected value. Do NOT change without investigating.
    assert (
        "<|im_end|>" in fmt.stop_tokens
    ), "Qwen2.5 stop token changed. The GGUF Modelfile must be updated."


@pytest.mark.integration
def test_qwen_ollama_template_contains_required_tokens() -> None:
    """The Ollama template must contain all required ChatML tokens."""
    from transformers import AutoTokenizer

    from alignforge.models.chat_format import ChatFormat

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    fmt = ChatFormat.from_tokenizer(tok)
    tmpl = fmt.to_ollama_template()

    required_tokens = [
        "<|im_start|>",
        "<|im_end|>",
        "{{ .Prompt }}",
        "{{ .Response }}",
    ]
    for token in required_tokens:
        assert (
            token in tmpl
        ), f"Required token {token!r} missing from Ollama template.\nTemplate:\n{tmpl}"


@pytest.mark.integration
def test_hf_and_simulated_ollama_match_for_user_turn() -> None:
    """The HF prompt and the simulated Ollama rendering must be byte-identical."""
    from transformers import AutoTokenizer

    from alignforge.models.chat_format import ChatFormat

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    fmt = ChatFormat.from_tokenizer(tok)

    test_prompt = "How do I reverse a list in Python?"
    messages = [{"role": "user", "content": test_prompt}]

    hf_output = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    stop = fmt.stop_tokens[0]
    expected = f"<|im_start|>user\n{test_prompt}{stop}\n<|im_start|>assistant\n"

    assert hf_output == expected, (
        f"HF output does not match expected Ollama rendering.\n"
        f"HF:       {hf_output!r}\n"
        f"Expected: {expected!r}"
    )
