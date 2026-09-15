"""Training stage: SFT and DPO. GPU-only; never imported by serve/ or ui/."""

from alignforge.train.run import run_sft

__all__ = ["run_sft"]
