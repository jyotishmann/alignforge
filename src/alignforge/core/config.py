"""Layered configuration with hash-based run identity (connector C1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Mixins ──────────────────────────────────────────────────────────────
class _Strict(BaseModel):
    """All config nodes forbid extra keys and are frozen after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ── Leaf schemas ────────────────────────────────────────────────────────
class ProjectConfig(_Strict):
    name: str = "alignforge"
    seed: int = Field(default=42, ge=0)


class PathsConfig(_Strict):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    registry_db: str = "alignforge.db"
    evals_dir: str = "evals"
    reports_dir: str = "reports"
    log_dir: str = "logs"


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["console", "json"] = "console"
    log_dir: str = "logs"


class ModelConfig(_Strict):
    name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = "main"
    torch_dtype: Literal["auto", "float16", "bfloat16"] = "auto"
    max_seq_len: int = Field(default=1024, ge=128, le=32768)
    trust_remote_code: bool = False


class QuantConfig(_Strict):
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # Compute dtype resolved at runtime by hardware probe — "auto" means
    # fp16 on Turing (T4), bf16 on Ampere+.
    bnb_4bit_compute_dtype: Literal["auto", "float16", "bfloat16"] = "auto"


class LoraConfig(_Strict):
    r: int = Field(default=16, ge=4, le=128)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    bias: Literal["none", "all", "lora_only"] = "none"
    # "auto" means discover all linear layers at runtime (Part 04).
    target_modules: list[str] | Literal["auto"] = "auto"
    task_type: str = "CAUSAL_LM"

    @model_validator(mode="after")
    def _alpha_scaling_check(self) -> LoraConfig:
        if self.lora_alpha % self.r != 0:
            import warnings

            warnings.warn(
                f"lora_alpha ({self.lora_alpha}) is not a multiple of r ({self.r}). "
                f"Effective scaling will be {self.lora_alpha / self.r:.3f}. "
                f"This is valid but unusual — verify it is intentional.",
                stacklevel=2,
            )
        return self


class SFTConfig(_Strict):
    num_train_epochs: int = Field(default=3, ge=1, le=20)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=200, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    packing: bool = False
    completion_only: bool = True


class DPOConfig(_Strict):
    beta: float = Field(default=0.1, gt=0.0, le=1.0)
    num_train_epochs: int = Field(default=1, ge=1, le=5)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=5e-6, gt=0)
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=0.5)
    max_prompt_length: int = Field(default=512, ge=64)
    max_length: int = Field(default=1024, ge=128)
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)


class EvalConfig(_Strict):
    suites: list[str] = Field(default=["domain_v1", "mtbench_sub", "alpacaeval_sub"])
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=512, ge=32, le=2048)
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    bootstrap_n: int = Field(default=10000, ge=1000)


class ServeConfig(_Strict):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    default_engine: Literal["transformers", "ollama", "echo"] = "echo"
    ollama_host: str = "http://localhost:11434"
    concurrency_limit: int = Field(default=4, ge=1, le=32)


class UIConfig(_Strict):
    api_base: str = "http://localhost:8000"
    share: bool = False
    server_port: int = Field(default=7860, ge=1024, le=65535)


class DataConfig(_Strict):
    name: str = "dev_assistant_v1"
    kind: Literal["sft", "preference"] = "sft"
    sources: list[str] = Field(default_factory=list)
    max_seq_len: int = Field(default=1024, ge=128)
    val_ratio: float = Field(default=0.05, ge=0.01, le=0.3)
    dedup_jaccard: float = Field(default=0.85, ge=0.5, le=1.0)
    decontaminate: bool = True
    decontaminate_ngram: int = Field(default=13, ge=5, le=25)


# ── Root ────────────────────────────────────────────────────────────────
class ExportConfig(_Strict):
    quant_type: str = "Q4_K_M"
    llama_cpp_dir: str = "vendor/llama.cpp"
    ollama_tag_prefix: str = "alignforge"
    ollama_version: str = "1"
    smoke_test_prompt: str = "What is the difference between a list and a tuple in Python?"


class AlignForgeConfig(_Strict):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


# ── Composition engine ──────────────────────────────────────────────────


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Lists are replaced, not appended.
    None values in override do NOT wipe base values."""
    merged = base.copy()
    for key, value in override.items():
        if value is None:
            continue
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_dotted_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply --set key=value overrides. Keys use dots: 'sft.lr' → d['sft']['lr'].
    Values are YAML-parsed so '1e-4' becomes a float, 'true' becomes a bool."""
    from alignforge.core.errors import ConfigError

    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must be key=value, got: {item!r}", key=item)
        key_path, raw_value = item.split("=", 1)
        parts = key_path.strip().split(".")
        value = yaml.safe_load(raw_value)

        target = d
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    return d


def _canonical_json(d: dict[str, Any]) -> str:
    """Deterministic JSON: sorted keys, no whitespace, consistent floats."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: AlignForgeConfig) -> str:
    """SHA-256 of the canonical JSON, truncated to 12 hex chars."""
    blob = _canonical_json(cfg.model_dump())
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(
    component_path: Path | None = None,
    overrides: list[str] | None = None,
    base_path: Path | None = None,
) -> AlignForgeConfig:
    """Full composition pipeline: base → component → overrides → validate → freeze.

    Steps map to connector C1 in the master document:
      1. Load base.yaml
      2. Load the named component config (if any)
      3. Deep-merge component over base
      4. Apply --set dotted overrides
      5. Validate into the typed Pydantic tree
    """
    from alignforge.core.errors import ConfigError
    from alignforge.core.paths import get_paths

    paths = get_paths()

    # Step 1: base config.
    bp = base_path or paths.configs_dir / "base.yaml"
    if bp.exists():
        with bp.open() as f:
            base_dict: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        base_dict = {}

    # Step 2: component config.
    component_dict: dict[str, Any] = {}
    if component_path is not None:
        if not component_path.exists():
            raise ConfigError(
                f"Component config not found: {component_path}", key=str(component_path)
            )
        with component_path.open() as f:
            component_dict = yaml.safe_load(f) or {}

    # Step 3: deep merge.
    merged = _deep_merge(base_dict, component_dict)

    # Step 4: CLI overrides.
    if overrides:
        merged = _apply_dotted_overrides(merged, overrides)

    # Step 5: validate.
    try:
        return AlignForgeConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigError(f"Config validation failed: {exc}") from exc
