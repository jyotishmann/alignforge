"""Exact and near-duplicate removal via hashing and MinHash/LSH."""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

log = structlog.get_logger()


def _text_of(ex: Any) -> str:
    """Extract the full text for hashing, regardless of schema type."""
    return ex.full_text


def exact_dedup(examples: list[Any]) -> tuple[list[Any], int]:
    """Remove byte-identical duplicates. Returns (unique, n_removed)."""
    seen: set[str] = set()
    unique: list[Any] = []
    for ex in examples:
        h = hashlib.sha256(_text_of(ex).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(ex)
    n_removed = len(examples) - len(unique)
    log.info("exact_dedup", before=len(examples), after=len(unique), removed=n_removed)
    return unique, n_removed


def near_dedup(
    examples: list[Any],
    jaccard_threshold: float = 0.85,
    num_perm: int = 128,
) -> tuple[list[Any], int]:
    """Remove near-duplicates via MinHash/LSH. Returns (unique, n_removed)."""
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    unique: list[Any] = []
    n_removed = 0

    for i, ex in enumerate(examples):
        text = _text_of(ex)
        # Shingle into 5-grams of characters for MinHash.
        shingles = set()
        for j in range(len(text) - 4):
            shingles.add(text[j : j + 5])

        mh = MinHash(num_perm=num_perm)
        for s in shingles:
            mh.update(s.encode("utf-8"))

        key = str(i)
        if lsh.query(mh):
            n_removed += 1
        else:
            lsh.insert(key, mh)
            unique.append(ex)

    log.info("near_dedup", before=len(examples), after=len(unique), removed=n_removed)
    return unique, n_removed


def deduplicate(
    examples: list[Any], jaccard_threshold: float = 0.85
) -> tuple[list[Any], dict[str, int]]:
    """Full dedup pipeline: exact → near. Returns (unique, stats)."""
    stage1, exact_removed = exact_dedup(examples)
    stage2, near_removed = near_dedup(stage1, jaccard_threshold=jaccard_threshold)
    stats = {
        "exact_removed": exact_removed,
        "near_removed": near_removed,
        "total_removed": exact_removed + near_removed,
    }
    return stage2, stats
