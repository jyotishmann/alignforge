"""TransformersEngine: in-process PEFT adapters with thread→async streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import structlog

from alignforge.serve.engine.base import HealthStatus, InferenceEngine, StreamChunk

log = structlog.get_logger()

_DONE = object()  # sentinel: producer puts this when generation is complete
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transformers-gen")


class TransformersEngine(InferenceEngine):
    """Runs HuggingFace models in-process with hot-swappable PEFT adapters.

    The thread→async bridge:
      1. TextIteratorStreamer is passed to model.generate() in a thread.
      2. A producer coroutine drains the streamer into an asyncio.Queue.
      3. astream() is an async generator that pops from the queue.

    This pattern ensures the event loop is never blocked by generate(),
    which means other requests can be served concurrently.
    """

    def __init__(self) -> None:
        self._models: dict[str, tuple[Any, Any]] = {}  # {weights_ref: (model, tokenizer)}

    def _load(self, weights_ref: str) -> tuple[Any, Any]:
        """Load model+tokenizer, caching by weights_ref."""
        if weights_ref not in self._models:
            log.info("transformers_engine_loading", weights_ref=weights_ref)
            import json
            from pathlib import Path

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            path = Path(weights_ref)
            is_adapter = (path / "adapter_config.json").exists()

            if is_adapter:
                with (path / "adapter_config.json").open() as f:
                    adapter_cfg = json.load(f)
                base_name = adapter_cfg.get("base_model_name_or_path", "")
                tok = AutoTokenizer.from_pretrained(base_name)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "left"
                from peft import PeftModel

                base = AutoModelForCausalLM.from_pretrained(
                    base_name, torch_dtype=torch.float16, device_map="auto"
                )
                model = PeftModel.from_pretrained(base, str(path))
            else:
                tok = AutoTokenizer.from_pretrained(str(path))
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.padding_side = "left"
                model = AutoModelForCausalLM.from_pretrained(
                    str(path), torch_dtype=torch.float16, device_map="auto"
                )
            model.eval()
            self._models[weights_ref] = (model, tok)
            log.info("transformers_engine_loaded", weights_ref=weights_ref)
        return self._models[weights_ref]

    async def astream(
        self,
        messages: list[dict[str, str]],
        model_id: str,
        params: dict[str, Any],
    ) -> AsyncIterator[StreamChunk]:
        """Bridge: generate() in thread, yield tokens as async stream."""
        from transformers import TextIteratorStreamer

        weights_ref = params.get("_weights_ref", model_id)
        model, tok = await asyncio.get_event_loop().run_in_executor(
            _executor, self._load, weights_ref
        )

        # Prepare prompt.
        from alignforge.core.config import AlignForgeConfig
        from alignforge.models.chat_format import get_or_build_format

        # Build a minimal config for the chat format.
        cfg = AlignForgeConfig()
        try:
            chat_fmt = get_or_build_format(cfg, tok)
            formatted = chat_fmt.render_prompt(messages)
        except Exception:
            formatted = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tok(formatted, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # TextIteratorStreamer feeds tokens to the queue as they are generated.
        streamer = TextIteratorStreamer(
            tok,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        queue: asyncio.Queue[str | object] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _drain_streamer() -> None:
            """Runs in a thread: drains streamer and puts tokens in the queue."""
            import torch

            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": int(params.get("max_tokens", 512)),
                "temperature": float(params.get("temperature", 0.7)),
                "do_sample": params.get("temperature", 0.7) > 0,
                "top_p": float(params.get("top_p", 0.9)),
                "pad_token_id": tok.pad_token_id,
                "eos_token_id": tok.eos_token_id,
            }
            try:
                with torch.no_grad():
                    model.generate(**gen_kwargs)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, f"__ERROR__:{exc}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        # Start generation in thread.
        asyncio.get_event_loop().run_in_executor(_executor, _drain_streamer)

        # Drain queue in the async generator.
        while True:
            token = await queue.get()
            if token is _DONE:
                yield StreamChunk(text="", finish_reason="stop")
                return
            if isinstance(token, str) and token.startswith("__ERROR__:"):
                error = token[10:]
                log.error("transformers_generate_error", error=error)
                yield StreamChunk(text=f"\n[Generation error: {error}]", finish_reason="stop")
                return
            yield StreamChunk(text=str(token))

    async def ahealth(self, model_id: str, weights_ref: str) -> HealthStatus:
        from pathlib import Path

        p = Path(weights_ref)
        if (p.exists() and (p / "adapter_config.json").exists()) or (p / "config.json").exists():
            return HealthStatus(model_id=model_id, backend="transformers", status="ok")
        return HealthStatus(
            model_id=model_id,
            backend="transformers",
            status="unavailable",
            detail=f"Weights not found at {weights_ref!r}",
        )
