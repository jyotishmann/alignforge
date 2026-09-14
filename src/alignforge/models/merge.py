"""Adapter merge: combine a LoRA adapter with its fp16 base for export."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from alignforge.core.config import AlignForgeConfig

if TYPE_CHECKING:
    pass

log = structlog.get_logger()


def merge_adapter_into_base(
    cfg: AlignForgeConfig,
    adapter_path: Path,
    output_path: Path,
) -> Path:
    """Load fp16 base + adapter, merge, save as a single set of safetensors.

    Why fp16 (not 4-bit) for the merge:
      The adapter updates are small residual corrections. Merging into a 4-bit
      base quantises those corrections to 4-bit, destroying their precision.
      Loading in fp16 preserves them, and the subsequent GGUF quantisation
      (Q4_K_M) applies a better-designed compression in a single step.

    This operation is CPU-feasible (~3 GB RAM for Qwen2.5-1.5B).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(
        "merge_adapter_start",
        base=cfg.model.name_or_path,
        adapter=str(adapter_path),
        output=str(output_path),
    )

    # Load the tokenizer first — it must be saved alongside the merged model.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    # Load base in fp16, NOT 4-bit.
    # device_map="cpu" keeps it on CPU if no GPU.
    device = "cuda" if _cuda_available() else "cpu"
    log.info("merge_loading_base", device=device)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    # Apply and merge the adapter.
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()

    log.info("merge_saving", output=str(output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    log.info("merge_complete", output=str(output_path))
    return output_path


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
