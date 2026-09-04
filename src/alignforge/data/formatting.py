"""Chat template formatting — connector C10 (training-time half)."""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


def format_sft_example(
    instruction: str,
    input_text: str,
    response: str,
    chat_template_fn: Any,
) -> str:
    """Format one SFT example into chat-templated text.

    Args:
        chat_template_fn: a callable(messages, tokenize=False, add_generation_prompt=False)
            — typically tokenizer.apply_chat_template.
    """
    user_content = instruction
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]

    formatted: str = chat_template_fn(messages, tokenize=False, add_generation_prompt=False)
    return formatted


def format_sft_batch(
    examples: list[Any],
    chat_template_fn: Any,
) -> list[dict[str, str]]:
    """Format a list of SFTExample into dicts with a 'text' field."""
    formatted = []
    for ex in examples:
        text = format_sft_example(
            instruction=ex.instruction,
            input_text=ex.input,
            response=ex.response,
            chat_template_fn=chat_template_fn,
        )
        formatted.append({"text": text, "source": ex.source})
    return formatted


def format_preference_batch(
    examples: list[Any],
    chat_template_fn: Any,
) -> list[dict[str, str]]:
    """Format preference examples for DPOTrainer.

    DPOTrainer expects columns: prompt, chosen, rejected — each as plain text
    or as message lists. We use message lists for clarity.
    """
    formatted = []
    for ex in examples:
        prompt_msgs = [{"role": "user", "content": ex.prompt}]
        chosen_msgs = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.chosen},
        ]
        rejected_msgs = [
            {"role": "user", "content": ex.prompt},
            {"role": "assistant", "content": ex.rejected},
        ]
        formatted.append(
            {
                "prompt": chat_template_fn(prompt_msgs, tokenize=False, add_generation_prompt=True),
                "chosen": chat_template_fn(
                    chosen_msgs, tokenize=False, add_generation_prompt=False
                ),
                "rejected": chat_template_fn(
                    rejected_msgs, tokenize=False, add_generation_prompt=False
                ),
                "source": ex.source,
            }
        )
    return formatted
