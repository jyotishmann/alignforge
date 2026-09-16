"""GGUF deployment delta: compare GGUF quality to safetensors checkpoint."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


def run_gguf_eval_delta(
    ollama_tag: str,
    reference_eval_id: str,
    suite_names: list[str],
    evals_dir: Path,
    responses_dir: Path,
    judgements_dir: Path,
    judge_fn: Any,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run eval on the GGUF model and compare to the safetensors eval.

    Returns a dict with:
      - gguf_win_rate: win rate of GGUF vs safetensors DPO
      - delta: GGUF win_rate minus 0.5 (positive = GGUF is better, negative = degraded)
      - n: number of cases
    """
    from alignforge.core.registry import get_registry
    from alignforge.eval.decode_params import EVAL_PARAMS
    from alignforge.eval.generation import run_generation  # ModelGenerator
    from alignforge.eval.judge import run_judging
    from alignforge.eval.metrics import bootstrap_ci

    # We compare two pseudo-model IDs: the GGUF (via Ollama) and
    # the safetensors DPO (also via Ollama if already registered,
    # otherwise generate from the previous responses).
    gguf_model_id = f"gguf_{ollama_tag.replace(':', '_')}"

    # Generate responses from the GGUF model.
    reg = get_registry()
    # Register GGUF temporarily for generation.
    reg.publish_model(
        model_id=gguf_model_id,
        display_name=f"GGUF ({ollama_tag})",
        backend="ollama",
        weights_ref=ollama_tag,
        sort_order=99,
    )

    # Find existing safetensors responses to use as the comparison baseline.
    safetensors_model_id = "dpo"  # the ID used in the Part 08 eval

    # Generate GGUF responses.
    from alignforge.models.chat_format import get_format

    try:
        chat_format = get_format(ollama_tag)
    except KeyError:
        # Fall back to a minimal format if not registered.
        from unittest.mock import MagicMock

        chat_format = MagicMock()
        chat_format.render_prompt = lambda msgs: msgs[-1]["content"] if msgs else ""

    run_generation(
        model_ids=[gguf_model_id],
        suite_names=suite_names,
        evals_dir=evals_dir,
        responses_dir=responses_dir,
        registry=reg,
        chat_format=chat_format,
        params=EVAL_PARAMS,
        limit=limit,
    )

    # Judge GGUF vs safetensors DPO.
    summary = run_judging(  # noqa: F841
        model_ids=[gguf_model_id, safetensors_model_id],
        suite_names=suite_names,
        responses_dir=responses_dir,
        judgements_dir=judgements_dir / "gguf_delta",
        judge_fn=judge_fn,
        evals_dir=evals_dir,
        limit=limit,
    )

    # Compute win rate: positive delta means GGUF is better (unlikely).
    # Typically we expect GGUF to be slightly worse (negative delta).
    pair_key = f"{gguf_model_id}_vs_{safetensors_model_id}"
    jfile = judgements_dir / "gguf_delta" / pair_key
    all_j: list[dict[str, Any]] = []
    for suite in suite_names:
        f = jfile / f"{suite}.jsonl"
        if f.exists():
            import json

            with f.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        with contextlib.suppress(json.JSONDecodeError):
                            all_j.append(json.loads(line))

    if not all_j:
        return {"gguf_win_rate": None, "delta": None, "n": 0}

    wr = bootstrap_ci(all_j, gguf_model_id, safetensors_model_id, n_resamples=1000)
    delta = round(wr["win_rate"] - 0.5, 4)  # 0 = no degradation, negative = degraded

    result = {
        "gguf_win_rate": wr["win_rate"],
        "ci_low": wr["ci_low"],
        "ci_high": wr["ci_high"],
        "delta_from_parity": delta,
        "n": wr["n"],
        "interpretation": (
            "GGUF model performs comparably to the safetensors checkpoint."
            if abs(delta) < 0.03
            else f"GGUF model is {'better' if delta > 0 else 'degraded'} by "
            f"{abs(delta):.2f} win-rate points vs the safetensors checkpoint."
        ),
    }
    log.info("gguf_eval_delta", **result)
    return result
