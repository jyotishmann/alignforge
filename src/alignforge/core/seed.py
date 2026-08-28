"""Deterministic seeding across Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import structlog

log = structlog.get_logger()


def seed_everything(seed: int) -> None:
    """Seed all known RNG sources. Call once at process start."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # CuBLAS workspace config — required for deterministic matmul on Ampere+.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some ops (flash attention) have no deterministic implementation.
            log.warning(
                "deterministic_algorithms_partial",
                msg="torch.use_deterministic_algorithms raised — "
                "some ops will remain nondeterministic.",
            )
            torch.use_deterministic_algorithms(False)
    except ImportError:
        pass

    log.info("seed_set", seed=seed)
