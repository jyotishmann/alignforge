"""Model loading: BitsAndBytesConfig, base model, and tokenizer setup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import structlog
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignforge.core.config import AlignForgeConfig
from alignforge.core.hardware import probe_hardware

if TYPE_CHECKING:
    pass  # avoid importing torch at module scope — keep the layer importable on CPU

log = structlog.get_logger()


def build_bnb_config(cfg: AlignForgeConfig) -> Any:
    """Build the BitsAndBytesConfig for 4-bit NF4 QLoRA.

    Resolves 'auto' compute dtype using the hardware probe so that
    T4 (Turing, cc 7.5) gets float16 and Ampere+ gets bfloat16.
    See ADR-002 for the QLoRA trade-off reasoning.
    """
    import torch
    from transformers import BitsAndBytesConfig

    hw = probe_hardware()

    compute_dtype_str = cfg.quant.bnb_4bit_compute_dtype
    if compute_dtype_str == "auto":
        compute_dtype_str = cast(Literal["auto", "float16", "bfloat16"], hw.recommended_dtype)

    compute_dtype = torch.float16 if compute_dtype_str == "float16" else torch.bfloat16

    log.info(
        "bnb_config_built",
        quant_type=cfg.quant.bnb_4bit_quant_type,
        compute_dtype=compute_dtype_str,
        double_quant=cfg.quant.bnb_4bit_use_double_quant,
        bf16_available=hw.bf16_supported,
    )

    return BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=cfg.quant.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_base_model(cfg: AlignForgeConfig, bnb_config: Any) -> Any:
    """Load the base causal LM in 4-bit NF4.

    device_map='auto' handles single-GPU, multi-GPU, and CPU transparently.
    attn_implementation degrades gracefully when flash_attention_2 is unavailable.
    """
    from importlib.util import find_spec

    import torch

    hw = probe_hardware()

    # Match outer dtype to compute dtype so no silent casting occurs.
    compute_dtype_str = cfg.quant.bnb_4bit_compute_dtype
    if compute_dtype_str == "auto":
        compute_dtype_str = cast(Literal["auto", "float16", "bfloat16"], hw.recommended_dtype)
    torch_dtype = torch.float16 if compute_dtype_str == "float16" else torch.bfloat16

    # Flash Attention 2 requires Ampere+ and a compatible build.
    attn_impl = "eager"
    if hw.bf16_supported and find_spec("flash_attn") is not None:
        attn_impl = "flash_attention_2"
    else:
        log.info("flash_attention_not_installed", fallback="eager")

    log.info(
        "loading_base_model",
        model=cfg.model.name_or_path,
        attn_impl=attn_impl,
        torch_dtype=compute_dtype_str,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        attn_implementation=attn_impl,
    )

    model.config.use_cache = False  # required for gradient checkpointing
    model.config.pretraining_tp = 1  # disable tensor parallelism for single-GPU

    log.info(
        "base_model_loaded",
        n_params=sum(p.numel() for p in model.parameters()),
    )
    return model


def load_tokenizer(
    cfg: AlignForgeConfig,
    padding_side: str = "right",
) -> Any:
    """Load and configure the tokenizer.

    padding_side: 'right' for training (loss mask alignment),
                  'left' for generation (attention alignment).
    """

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
        padding_side=padding_side,
    )

    # Set pad_token if absent — use EOS, which is standard for instruct models.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log.info("pad_token_set_to_eos", eos=tokenizer.eos_token)

    # Validate chat template — fail loudly before training, not mid-loop.
    if tokenizer.chat_template is None:
        from alignforge.core.errors import ModelError

        raise ModelError(
            f"Tokenizer for {cfg.model.name_or_path!r} has no chat_template. "
            f"Instruction-following training requires a chat template. "
            f"Check the model revision or add a template manually."
        )

    log.info(
        "tokenizer_loaded",
        vocab_size=tokenizer.vocab_size,
        padding_side=tokenizer.padding_side,
        pad_token=tokenizer.pad_token,
    )
    return tokenizer


def set_tokenizer_padding_for_generation(tokenizer: Any) -> None:
    """Switch padding to left for generation. Call before eval/inference."""
    tokenizer.padding_side = "left"


def set_tokenizer_padding_for_training(tokenizer: Any) -> None:
    """Switch padding to right for training. Restores after eval."""
    tokenizer.padding_side = "right"
