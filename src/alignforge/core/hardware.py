"""Hardware probe: device, VRAM, dtype support, batch size recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class HardwareInfo:
    device: str  # "cuda", "mps", "cpu"
    device_name: str  # e.g. "Tesla T4", "Apple M2", "cpu"
    compute_capability: tuple[int, int] | None  # e.g. (7, 5) for T4
    vram_total_gb: float  # 0.0 for CPU
    vram_free_gb: float  # 0.0 for CPU
    bf16_supported: bool  # Ampere+ (compute cap ≥ 8.0)
    recommended_dtype: str  # "float16" or "bfloat16"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_hardware() -> HardwareInfo:
    """Detect device capabilities. Safe to call without torch installed."""
    try:
        import torch
    except ImportError:
        info = HardwareInfo(
            device="cpu",
            device_name="cpu",
            compute_capability=None,
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            bf16_supported=False,
            recommended_dtype="float16",
        )
        log.info("hardware_probe", **info.to_dict())
        return info

    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(dev)
        total = torch.cuda.get_device_properties(dev).total_mem / (1024**3)
        free = total - torch.cuda.memory_reserved(dev) / (1024**3)
        bf16_ok = cc[0] >= 8  # Ampere = 8.0, Turing (T4) = 7.5
        info = HardwareInfo(
            device="cuda",
            device_name=torch.cuda.get_device_name(dev),
            compute_capability=cc,
            vram_total_gb=round(total, 2),
            vram_free_gb=round(free, 2),
            bf16_supported=bf16_ok,
            recommended_dtype="bfloat16" if bf16_ok else "float16",
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info = HardwareInfo(
            device="mps",
            device_name="Apple Silicon",
            compute_capability=None,
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            bf16_supported=False,
            recommended_dtype="float16",
        )
    else:
        info = HardwareInfo(
            device="cpu",
            device_name="cpu",
            compute_capability=None,
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            bf16_supported=False,
            recommended_dtype="float16",
        )

    log.info("hardware_probe", **info.to_dict())
    return info


def recommend_batch_config(
    model_params_b: float,
    seq_len: int,
    free_vram_gb: float,
    is_dpo: bool = False,
) -> tuple[int, int]:
    """Suggest (batch_size, grad_accum) to hit effective_batch ~ 16.

    Very conservative: leaves 30% headroom for activation peaks and optimizer states.
    """
    # Rough per-sample memory: 2 bytes/param (fp16 activations) x seq/1024 scaling.
    # DPO doubles it (chosen + rejected forward passes).
    multiplier = 2.0 if is_dpo else 1.0
    per_sample_gb = (model_params_b * 2 * (seq_len / 1024) * multiplier) / 1024
    usable = free_vram_gb * 0.70
    max_bs = max(1, int(usable / per_sample_gb)) if per_sample_gb > 0 else 1
    bs = min(max_bs, 4)
    accum = max(1, 16 // bs)
    return bs, accum
