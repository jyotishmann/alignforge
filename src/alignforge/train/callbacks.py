"""Training callbacks: VRAM telemetry and probe generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


class VRAMCallback:
    """Log peak VRAM usage after every N optimizer steps.

    Resets the PyTorch peak-memory counter at the start of each step
    so each log entry reflects the peak for that step alone, not since
    process start.
    """

    def __init__(self, log_every_n_steps: int = 10) -> None:
        self.log_every_n_steps = log_every_n_steps
        self._step = 0

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._step += 1
        if self._step % self.log_every_n_steps != 0:
            return
        try:
            import torch

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_gb = round(peak_bytes / (1024**3), 3)
                total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                log.info(
                    "vram_step",
                    step=state.global_step,
                    peak_gb=peak_gb,
                    total_gb=total_gb,
                    pct=round(100 * peak_gb / total_gb, 1) if total_gb > 0 else 0,
                )
        except ImportError:
            pass


class ProbeGenerationCallback:
    """Generate and log responses to fixed prompts every N steps.

    Lets you watch the model learn the domain register qualitatively,
    not just by watching loss decrease.
    """

    def __init__(
        self,
        prompts: list[str],
        chat_format: Any,
        tokenizer: Any,
        every_n_steps: int = 100,
        max_new_tokens: int = 128,
        output_dir: Any = None,
    ) -> None:
        self.prompts = prompts[:5]  # cap at 5 — probes should be fast
        self.chat_format = chat_format
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self._history: list[dict[str, Any]] = []

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        if state.global_step % self.every_n_steps != 0:
            return
        if model is None:
            return

        try:
            import torch

            log.info("probe_generation_start", step=state.global_step, n_prompts=len(self.prompts))
            model.eval()

            # Switch tokenizer to left-padding for generation.
            original_padding = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"

            results = []
            with torch.no_grad():
                for prompt in self.prompts:
                    messages = [{"role": "user", "content": prompt}]
                    formatted = self.chat_format.render_prompt(messages)
                    inputs = self.tokenizer(
                        formatted, return_tensors="pt", truncation=True, max_length=512
                    ).to(model.device)

                    out = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,  # greedy for reproducibility
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    # Decode only the newly generated tokens.
                    gen_ids = out[0][inputs["input_ids"].shape[1] :]
                    response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

                    results.append(
                        {
                            "step": state.global_step,
                            "prompt": prompt,
                            "response": response,
                        }
                    )
                    log.info(
                        "probe_result",
                        step=state.global_step,
                        prompt=prompt[:80],
                        response=response[:200],
                    )

            self.tokenizer.padding_side = original_padding
            model.train()

            self._history.extend(results)
            self._save_history()

        except Exception as exc:
            # Probe generation failure should never abort training.
            log.warning("probe_generation_failed", error=str(exc))

    def _save_history(self) -> None:
        if self.output_dir is None:
            return
        import json

        out = Path(self.output_dir) / "probe_generations.jsonl"
        with out.open("w") as f:
            for entry in self._history:
                f.write(json.dumps(entry) + "\n")


class DriveSyncCallback:
    """Sync checkpoints to Google Drive after each save. Colab-specific.

    Enable with: alignforge train sft --drive-sync /content/drive/MyDrive/alignforge
    Only useful in Colab; silently no-ops if the Drive path doesn't exist.
    """

    def __init__(self, drive_dir: str) -> None:
        self.drive_dir = Path(drive_dir)
        self._enabled = self.drive_dir.exists()
        if not self._enabled:
            log.warning(
                "drive_sync_disabled",
                reason=f"Drive directory {drive_dir!r} does not exist.",
            )

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not self._enabled:
            return
        import shutil

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return
        dest = self.drive_dir / "checkpoints" / f"checkpoint-{state.global_step}"
        try:
            shutil.copytree(checkpoint_dir, dest, dirs_exist_ok=True)
            log.info(
                "drive_sync_complete",
                step=state.global_step,
                dest=str(dest),
            )
        except Exception as exc:
            log.warning("drive_sync_failed", error=str(exc))


def wrap_callbacks(callbacks: list[Any]) -> list[Any]:
    """Wrap our plain-class callbacks in the HuggingFace TrainerCallback interface."""
    from transformers import TrainerCallback

    class _Adapter(TrainerCallback):
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def on_step_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            if hasattr(self._inner, "on_step_begin"):
                self._inner.on_step_begin(args, state, control, **kw)
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            if hasattr(self._inner, "on_step_end"):
                self._inner.on_step_end(args, state, control, **kw)
            return control

        def on_evaluate(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            if hasattr(self._inner, "on_evaluate"):
                self._inner.on_evaluate(args, state, control, **kw)
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            if hasattr(self._inner, "on_save"):
                self._inner.on_save(args, state, control, **kw)
            return control

    return [_Adapter(cb) for cb in callbacks]
