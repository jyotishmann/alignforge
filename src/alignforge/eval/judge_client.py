"""Judge client abstraction: OpenAI API and local Transformers backends."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Any

import structlog

log = structlog.get_logger()

JudgeFn = Callable[[str, str], str]  # (system_prompt, user_prompt) -> response_text


# ── OpenAI API backend ────────────────────────────────────────────────────

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


def make_openai_judge(
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    base_url: str = "https://api.openai.com/v1",
    max_retries: int = 6,
) -> JudgeFn:
    """Judge backed by any OpenAI-compatible chat-completions API (OpenAI, Groq, ...).

    Requires ALIGNFORGE_JUDGE_API_KEY (or OPENAI_API_KEY). Uses temperature=0 for
    deterministic verdicts. Rate limits and server errors are retried with
    exponential backoff (honouring Retry-After); other client errors such as a
    bad key or unknown model fail fast, since retrying cannot fix them.
    """
    api_key = os.environ.get("ALIGNFORGE_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OSError("No judge API key found. Set ALIGNFORGE_JUDGE_API_KEY or OPENAI_API_KEY.")

    import httpx

    url = f"{base_url.rstrip('/')}/chat/completions"

    def _backoff(attempt: int, retry_after: str | None) -> float:
        try:
            hinted = float(retry_after) if retry_after else 0.0
        except ValueError:  # Retry-After may be an HTTP date; fall back to exponential
            hinted = 0.0
        return max(hinted, min(60.0, 2.0**attempt)) + random.uniform(0.0, 1.0)

    def _call(system_prompt: str, user_prompt: str) -> str:
        for attempt in range(max_retries):
            last = attempt == max_retries - 1
            try:
                r = httpx.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": temperature,
                        "max_tokens": 512,
                    },
                    timeout=60.0,
                )
            except httpx.TransportError as exc:  # network failure or timeout
                if last:
                    raise
                wait = _backoff(attempt, None)
                log.warning(
                    "judge_api_error", attempt=attempt, error=str(exc), wait_s=round(wait, 1)
                )
                time.sleep(wait)
                continue
            if r.status_code in _RETRYABLE_STATUS and not last:
                wait = _backoff(attempt, r.headers.get("retry-after"))
                log.warning(
                    "judge_api_retry", attempt=attempt, status=r.status_code, wait_s=round(wait, 1)
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])
        raise RuntimeError("judge retry loop exited without a result")

    return _call


# ── Local Transformers backend ────────────────────────────────────────────


def make_local_judge(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 512,
) -> JudgeFn:
    """Returns a judge function backed by a local Transformers model.

    Use when: no API access, or for cross-checking judge lineage bias.
    Warning: using the same model family as the evaluated models introduces
    self-enhancement bias. Note this in the eval report if used.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading_local_judge", model=model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    def _call(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=3000)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        n_prompt = inputs["input_ids"].shape[1]
        result = tok.decode(out[0][n_prompt:], skip_special_tokens=True)
        return result if isinstance(result, str) else result[0]

    return _call


# ── Echo backend (for testing) ────────────────────────────────────────────


def make_echo_judge(always_prefer: str = "A") -> JudgeFn:
    """Deterministic fake judge for testing. always_prefer: 'A', 'B', or 'tie'."""
    import json as _json

    def _call(system_prompt: str, user_prompt: str) -> str:
        return _json.dumps(
            {
                "verdict": always_prefer,
                "rationale": "Echo judge — deterministic test fixture.",
                "dimension_scores": {
                    "Correctness": 3,
                    "Directness": 3,
                    "Concision": 3,
                    "Actionability": 3,
                    "Calibration": 3,
                },
                "confidence": "high",
            }
        )

    return _call


def get_judge_fn(backend: str = "openai", **kwargs: Any) -> JudgeFn:
    """Factory: resolve backend name to a judge function."""
    if backend == "openai":
        return make_openai_judge(**kwargs)
    elif backend == "local":
        return make_local_judge(**kwargs)
    elif backend == "echo":
        return make_echo_judge(**kwargs)
    else:
        raise ValueError(f"Unknown judge backend: {backend!r}. Options: openai, local, echo.")
