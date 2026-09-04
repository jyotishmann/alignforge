"""Token length statistics and length filtering."""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


def compute_token_lengths(
    texts: list[str],
    tokenizer: Any,
) -> list[int]:
    """Tokenise each text and return the length list."""
    lengths: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(ids))
    return lengths


def length_stats(lengths: list[int]) -> dict[str, int]:
    """Percentile summary of token lengths."""
    if not lengths:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0, "n": 0}
    s = sorted(lengths)
    n = len(s)
    return {
        "p50": s[n // 2],
        "p90": s[int(n * 0.9)],
        "p95": s[int(n * 0.95)],
        "p99": s[int(n * 0.99)],
        "max": s[-1],
        "n": n,
    }


def filter_by_length(
    records: list[dict[str, Any]],
    lengths: list[int],
    max_len: int,
) -> tuple[list[dict[str, Any]], int]:
    """Drop records exceeding max_len tokens. Returns (survivors, n_dropped)."""
    survivors = []
    n_dropped = 0
    for rec, length in zip(records, lengths, strict=True):
        if length <= max_len:
            rec["n_tokens"] = length
            survivors.append(rec)
        else:
            n_dropped += 1
    log.info("length_filter", before=len(records), after=len(survivors), max_len=max_len)
    return survivors, n_dropped
