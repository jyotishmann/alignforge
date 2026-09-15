"""Fixed decode parameters applied identically to all evaluated models.

These are committed constants, not config fields, because changing them
invalidates all historical comparisons. Treat them like the eval set itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeParams:
    """Generation parameters. Frozen so accidental mutation is a hard error."""

    temperature: float
    top_p: float
    max_new_tokens: int
    do_sample: bool
    repetition_penalty: float = 1.0
    seed: int = 42  # for reproducibility when do_sample=True

    def to_dict(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }


# ── Standard params for judge-based win-rate evaluation ─────────────────
EVAL_PARAMS = DecodeParams(
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=512,
    do_sample=True,
    seed=42,
)

# ── Greedy params for verifiable exact-match cases ───────────────────────
GREEDY_PARAMS = DecodeParams(
    temperature=0.0,
    top_p=1.0,
    max_new_tokens=256,
    do_sample=False,
    seed=42,
)

# ── Mapping for the CLI ──────────────────────────────────────────────────
NAMED_PARAMS: dict[str, DecodeParams] = {
    "eval": EVAL_PARAMS,
    "greedy": GREEDY_PARAMS,
}
