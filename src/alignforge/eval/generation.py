"""Batched evaluation generation: load models, generate responses, checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

from alignforge.eval.decode_params import EVAL_PARAMS, DecodeParams
from alignforge.eval.schemas import EvalCase, GeneratedResponse, GenerationCheckpoint

log = structlog.get_logger()


# ── Per-model generator: wraps any backend ───────────────────────────────
class ModelGenerator:
    """Generates responses from one model for evaluation cases.

    Abstracts over backends (Transformers adapter, Ollama GGUF, echo).
    The backend is resolved from the served_models registry, exactly as
    the API does — so eval and API share the same code path.
    """

    def __init__(
        self,
        model_id: str,
        registry: Any,
        chat_format: Any,
        params: DecodeParams = EVAL_PARAMS,
    ) -> None:
        self.model_id = model_id
        self.params = params
        self._backend = self._resolve_backend(model_id, registry, chat_format)

    def _resolve_backend(self, model_id: str, registry: Any, chat_format: Any) -> Any:
        """Resolve model_id to a concrete generation function."""
        served = registry.get_served_model(model_id)
        if served is None:
            # Fallback: treat model_id as a direct artifact path or run_id.
            return _make_transformers_generator(model_id, chat_format, self.params)

        backend = served["backend"]
        weights_ref = served["weights_ref"]

        if backend == "echo":
            return _make_echo_generator(model_id)
        elif backend == "ollama":
            return _make_ollama_generator(weights_ref, chat_format, self.params)
        else:  # "transformers"
            return _make_transformers_generator(weights_ref, chat_format, self.params)

    def generate(self, case: EvalCase) -> GeneratedResponse:
        """Generate a response for one eval case. Never raises — errors are recorded."""
        t0 = time.time()
        try:
            response, n_tokens = self._backend(case.prompt)
            return GeneratedResponse(
                case_id=case.id,
                suite=case.suite,
                model_id=self.model_id,
                prompt=case.prompt,
                response=response,
                n_tokens=n_tokens,
                time_seconds=round(time.time() - t0, 3),
                decode_params=self.params.to_dict(),
            )
        except Exception as exc:
            log.warning(
                "generation_error",
                model_id=self.model_id,
                case_id=case.id,
                error=str(exc),
            )
            return GeneratedResponse(
                case_id=case.id,
                suite=case.suite,
                model_id=self.model_id,
                prompt=case.prompt,
                response="",
                n_tokens=0,
                time_seconds=round(time.time() - t0, 3),
                error=str(exc),
            )


def _make_echo_generator(model_id: str) -> Any:
    """Deterministic fake for testing. Returns a canned response."""

    def _generate(prompt: str) -> tuple[str, int]:
        response = (
            f"[Echo response from {model_id}] This is a test response for prompt: {prompt[:50]}..."
        )
        return response, len(response.split())

    return _generate


def _make_ollama_generator(
    model_tag: str,
    chat_format: Any,
    params: DecodeParams,
) -> Any:
    """HTTP generator via local Ollama daemon."""
    import httpx

    ollama_host = "http://localhost:11434"

    def _generate(prompt: str) -> tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": model_tag,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_new_tokens,
                "seed": params.seed,
            },
        }
        r = httpx.post(f"{ollama_host}/api/chat", json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        response = data["message"]["content"]
        n_tokens = data.get("eval_count", len(response.split()))
        return response, n_tokens

    return _generate


def _make_transformers_generator(
    adapter_path_or_run_id: str,
    chat_format: Any,
    params: DecodeParams,
) -> Any:
    """HuggingFace Transformers generator for adapter checkpoints.

    Loads once and closes over the model/tokenizer.
    """
    import torch

    log.info("loading_eval_model", ref=adapter_path_or_run_id)
    model, tokenizer = _load_eval_model(adapter_path_or_run_id, chat_format)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _generate(prompt: str) -> tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        formatted = chat_format.render_prompt(messages)

        # Left-pad for generation (see Part 05 prose on padding side).
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                temperature=params.temperature if params.do_sample else 1.0,
                top_p=params.top_p if params.do_sample else 1.0,
                do_sample=params.do_sample,
                repetition_penalty=params.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        n_prompt = inputs["input_ids"].shape[1]
        gen_ids = out[0][n_prompt:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return response, len(gen_ids)

    return _generate


def _load_eval_model(ref: str, chat_format: Any) -> tuple[Any, Any]:
    """Load model+tokenizer for evaluation. Handles run_id and direct paths."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from alignforge.core.paths import get_paths
    from alignforge.core.registry import get_registry

    paths = get_paths()

    # Try to resolve as a run_id first.
    reg = get_registry()
    p = Path(ref)
    if not p.exists():
        # Check artifacts dir.
        candidate = paths.artifacts_dir / ref / "adapter"
        if candidate.exists():
            p = candidate
        else:
            arts = reg.get_artifacts(ref)
            for art in arts:
                if art["kind"] == "lora_adapter":
                    p = Path(art["path"])
                    break

    if not p.exists():
        raise FileNotFoundError(f"Cannot find model weights for ref: {ref!r}")

    # Check if it's a PEFT adapter or a full model.
    is_adapter = (p / "adapter_config.json").exists()

    if is_adapter:
        from peft import PeftModel

        # Need the base model name from the adapter config.
        with (p / "adapter_config.json").open() as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path", "")
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(p))
        model.eval()
    else:
        # Full model (merged).
        tokenizer = AutoTokenizer.from_pretrained(str(p))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(p),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()  # type: ignore[no-untyped-call]

    return model, tokenizer


def run_generation(
    model_ids: list[str],
    suite_names: list[str],
    evals_dir: Path,
    responses_dir: Path,
    registry: Any,
    chat_format: Any,
    params: DecodeParams = EVAL_PARAMS,
    limit: int | None = None,
) -> dict[str, int]:
    """Generate responses for all (model, suite) pairs.

    Returns {model_id: total_completed} summary.
    Checkpointed: safe to re-run after interruption.
    """
    from alignforge.eval.suites.loader import load_suite

    summary: dict[str, int] = {}

    for model_id in model_ids:
        log.info("generation_model_start", model_id=model_id)
        generator = ModelGenerator(
            model_id=model_id,
            registry=registry,
            chat_format=chat_format,
            params=params,
        )

        total_completed = 0
        for suite_name in suite_names:
            cases = load_suite(suite_name, evals_dir, limit=limit)
            if not cases:
                log.warning("suite_empty_skipping", suite=suite_name, model=model_id)
                continue

            out_dir = responses_dir / model_id
            out_dir.mkdir(parents=True, exist_ok=True)
            response_file = out_dir / f"{suite_name}.jsonl"
            ckpt_file = out_dir / f"{suite_name}.ckpt.json"

            # Load or initialise checkpoint.
            ckpt = _load_checkpoint(ckpt_file, model_id, suite_name, len(cases))

            log.info(
                "suite_generation_start",
                model_id=model_id,
                suite=suite_name,
                total=len(cases),
                remaining=ckpt.n_remaining,
            )

            completed_this_suite = 0
            with response_file.open("a") as out_f:
                for case in cases:
                    if ckpt.is_done(case.id):
                        log.debug("case_skipped_checkpoint", case_id=case.id)
                        continue

                    resp = generator.generate(case)
                    out_f.write(resp.model_dump_json() + "\n")
                    out_f.flush()  # flush after every case — disk writes, not buffer

                    if resp.error:
                        ckpt.failed_ids.append(case.id)
                    else:
                        ckpt.completed_ids.append(case.id)
                    _save_checkpoint(ckpt_file, ckpt)

                    completed_this_suite += 1
                    if completed_this_suite % 10 == 0:
                        log.info(
                            "generation_progress",
                            model_id=model_id,
                            suite=suite_name,
                            done=len(ckpt.completed_ids),
                            failed=len(ckpt.failed_ids),
                            total=len(cases),
                        )

            total_completed += len(ckpt.completed_ids)
            log.info(
                "suite_generation_done",
                model_id=model_id,
                suite=suite_name,
                completed=len(ckpt.completed_ids),
                failed=len(ckpt.failed_ids),
            )

        summary[model_id] = total_completed
        log.info("model_generation_done", model_id=model_id, total=total_completed)

    return summary


def _load_checkpoint(path: Path, model_id: str, suite: str, total: int) -> GenerationCheckpoint:
    if path.exists():
        try:
            with path.open() as f:
                return GenerationCheckpoint.model_validate(json.load(f))
        except Exception:
            pass
    return GenerationCheckpoint(model_id=model_id, suite=suite, total_cases=total)


def _save_checkpoint(path: Path, ckpt: GenerationCheckpoint) -> None:
    with path.open("w") as f:
        f.write(ckpt.model_dump_json(indent=2))
