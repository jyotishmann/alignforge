"""N-gram decontamination: drop training rows that overlap with eval prompts."""

from __future__ import annotations

import re
from typing import Any

import structlog

log = structlog.get_logger()

_WORD = re.compile(r"\w+")


def _ngrams(text: str, n: int) -> set[str]:
    """Word-level n-grams over lowercased tokens (the GPT-3 13-gram convention).

    Text shorter than n words is indexed whole, so an exact copy of a short
    eval prompt is still caught.
    """
    tokens = _WORD.findall(text.lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_eval_ngram_index(eval_prompts: list[str], n: int = 13) -> set[str]:
    """Build a set of all n-grams from all eval prompts."""
    index: set[str] = set()
    for prompt in eval_prompts:
        index.update(_ngrams(prompt, n))
    log.info("decontamination_index_built", n_prompts=len(eval_prompts), n_grams=len(index))
    return index


def is_contaminated(text: str, eval_index: set[str], n: int = 13, min_overlap: int = 1) -> bool:
    """True if text shares at least `min_overlap` n-grams with the eval index."""
    text_ngrams = _ngrams(text, n)
    overlap = text_ngrams & eval_index
    return len(overlap) >= min_overlap


def decontaminate(
    examples: list[Any],
    eval_prompts: list[str],
    n: int = 13,
    min_overlap: int = 1,
) -> tuple[list[Any], int]:
    """Remove training rows contaminated by eval prompts. Returns (clean, n_dropped)."""
    if not eval_prompts:
        return examples, 0

    index = build_eval_ngram_index(eval_prompts, n=n)

    clean: list[Any] = []
    n_dropped = 0

    for ex in examples:
        text = ex.prompt_text if hasattr(ex, "prompt_text") else ex.prompt
        if is_contaminated(text, index, n=n, min_overlap=min_overlap):
            n_dropped += 1
            log.debug("decontaminated_row", source=ex.source, text_preview=text[:80])
        else:
            clean.append(ex)

    log.info("decontamination_done", before=len(examples), after=len(clean), dropped=n_dropped)
    return clean, n_dropped
