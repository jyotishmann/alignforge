"""Model layer module: loading, LoRA, ChatFormat, and adapter merge."""

from alignforge.models.chat_format import ChatFormat, get_format, get_or_build_format
from alignforge.models.loading import (
    build_bnb_config,
    load_base_model,
    load_tokenizer,
    set_tokenizer_padding_for_generation,
    set_tokenizer_padding_for_training,
)
from alignforge.models.lora import (
    build_lora_model,
    count_trainable_params,
    discover_target_modules,
)
from alignforge.models.merge import merge_adapter_into_base

__all__ = [
    "ChatFormat",
    "build_bnb_config",
    "build_lora_model",
    "count_trainable_params",
    "discover_target_modules",
    "get_format",
    "get_or_build_format",
    "load_base_model",
    "load_tokenizer",
    "merge_adapter_into_base",
    "set_tokenizer_padding_for_generation",
    "set_tokenizer_padding_for_training",
]
