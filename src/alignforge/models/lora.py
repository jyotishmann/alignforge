"""LoRA configuration and target module discovery."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import structlog

from alignforge.core.config import AlignForgeConfig
from alignforge.core.errors import ModelError

if TYPE_CHECKING:
    pass

log = structlog.get_logger()

# Patterns that match the standard attention and MLP projection layers
# across Qwen2.5, Phi-3, LLaMA, and Mistral family models.
# Exact names differ by architecture.
_TARGET_PATTERNS: list[str] = [
    r"q_proj$",
    r"k_proj$",
    r"v_proj$",
    r"o_proj$",  # LLaMA-style attention
    r"qkv_proj$",
    r"dense$",  # Phi-3 attention
    r"gate_proj$",
    r"up_proj$",
    r"down_proj$",  # LLaMA-style MLP
    r"fc1$",
    r"fc2$",  # Phi-3 MLP
    r"c_attn$",
    r"c_proj$",  # GPT-2 style
]


def discover_target_modules(model: Any) -> list[str]:
    """Walk the module tree and return names of all linear layers matching patterns.

    Returns exact module names (not patterns), suitable for LoraConfig.target_modules.
    """
    try:
        import bitsandbytes as bnb

        linear_types: tuple[type, ...] = (
            __import__("torch").nn.Linear,
            bnb.nn.Linear4bit,
            bnb.nn.Linear8bitLt,
        )
    except ImportError:
        import torch

        linear_types = (torch.nn.Linear,)

    compiled = [re.compile(p) for p in _TARGET_PATTERNS]
    target_names: list[str] = []

    for name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        short_name = name.split(".")[-1]  # e.g. "q_proj" from "model.layers.0.self_attn.q_proj"
        if any(pat.match(short_name) for pat in compiled):
            target_names.append(name)

    if not target_names:
        raise ModelError(
            "No target modules found by automatic discovery. "
            "The model architecture may not match known patterns. "
            "Set lora.target_modules explicitly in the model config."
        )

    # Return the short names (deduplicated) — LoraConfig expects leaf names.
    short_unique = sorted({name.split(".")[-1] for name in target_names})
    log.info(
        "target_modules_discovered",
        n_layers=len(target_names),
        unique_names=short_unique,
    )
    return short_unique


def count_trainable_params(model: Any) -> dict[str, float]:
    """Count and log trainable vs total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total > 0 else 0.0
    result = {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(pct, 3),
    }
    log.info("trainable_params", **result)
    return result


def build_lora_model(
    cfg: AlignForgeConfig,
    base_model: Any,
) -> Any:
    """Apply QLoRA adapters to the base model.

    Order:
      1. prepare_model_for_kbit_training (layernorm fp32, grad ckpt, embed grad)
      2. build LoraConfig with auto-discovered or explicit target_modules
      3. get_peft_model
      4. count and log trainable parameters
    """
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    # Step 1: prepare the 4-bit model for training.
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # Note: use_reentrant=False is the modern setting. The old True default
    # conflicts with some PyTorch 2.x autograd internals and can cause
    # occasional checkpointing errors on longer sequences.

    # Step 2: resolve target modules.
    if cfg.lora.target_modules == "auto":
        target_modules = discover_target_modules(base_model)
    else:
        target_modules = list(cfg.lora.target_modules)

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        # B is initialised to 0, A to Gaussian — so ΔW = BA = 0 at step 0.
        # This preserves the pretrained model's function at initialisation.
        init_lora_weights=True,
    )

    # Step 3: apply adapters.
    model = get_peft_model(base_model, lora_cfg)

    # Step 4: report.
    count_trainable_params(model)

    return model
