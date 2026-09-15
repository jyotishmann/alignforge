"""Training stage: SFT and DPO. GPU-only; never imported by serve/ or ui/."""

from alignforge.train.dpo_run import run_dpo
from alignforge.train.run import run_sft

__all__ = ["run_dpo", "run_sft"]
