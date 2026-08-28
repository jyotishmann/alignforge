"""Hardware probe tests — must pass on CPU-only CI runners."""

from __future__ import annotations

from alignforge.core.hardware import HardwareInfo, probe_hardware, recommend_batch_config


def test_probe_returns_hardware_info() -> None:
    """probe_hardware returns a valid HardwareInfo regardless of GPU presence."""
    info = probe_hardware()
    assert isinstance(info, HardwareInfo)
    assert info.device in ("cuda", "mps", "cpu")
    assert isinstance(info.recommended_dtype, str)


def test_probe_to_dict() -> None:
    """to_dict produces a serialisable dictionary."""
    info = probe_hardware()
    d = info.to_dict()
    assert "device" in d
    assert "bf16_supported" in d


def test_recommend_batch_config_conservative() -> None:
    """Batch recommendation stays within VRAM limits."""
    bs, accum = recommend_batch_config(
        model_params_b=1.5, seq_len=1024, free_vram_gb=14.0, is_dpo=False
    )
    assert bs >= 1
    assert accum >= 1
    assert bs * accum >= 4  # effective batch is at least 4

    # DPO needs more memory — batch should be smaller or equal.
    bs_dpo, _ = recommend_batch_config(
        model_params_b=1.5, seq_len=1024, free_vram_gb=14.0, is_dpo=True
    )
    assert bs_dpo <= bs
