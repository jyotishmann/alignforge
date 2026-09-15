"""DPO-specific callbacks: implicit KL monitoring and divergence detection."""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


class DPOMetricsCallback:
    """Compute and log implicit KL from already-computed reward metrics.

    Implicit KL = E[(r_chosen + r_rejected) / (2 * beta)]
    This is the average absolute magnitude of the implicit reward, which
    measures how far the policy has drifted from the reference.

    KL formula derivation:
        KL(π_θ || π_ref) ≈ (1/β) · E_D[r̂(y_w) - r̂(y_l)] / 2
    We approximate per-batch using available reward scalars.
    """

    def __init__(self, beta: float, kl_warn_threshold: float = 8.0) -> None:
        self.beta = beta
        self.kl_warn_threshold = kl_warn_threshold
        self._step_klvals: list[float] = []

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        r_chosen = logs.get("rewards/chosen")
        r_rejected = logs.get("rewards/rejected")

        if r_chosen is None or r_rejected is None:
            return

        # Implicit KL: average magnitude of implicit rewards / beta.
        # Approximates E[log(π_θ/π_ref)] over the batch.
        implicit_kl = (abs(r_chosen) + abs(r_rejected)) / (2.0 * self.beta)
        margin = logs.get("rewards/margins", r_chosen - r_rejected)

        self._step_klvals.append(implicit_kl)
        log.info(
            "dpo_step_metrics",
            step=state.global_step,
            rewards_chosen=round(r_chosen, 4),
            rewards_rejected=round(r_rejected, 4),
            rewards_margin=round(margin, 4),
            rewards_accuracy=round(logs.get("rewards/accuracies", 0.0), 4),
            implicit_kl=round(implicit_kl, 4),
            beta=self.beta,
        )

        if implicit_kl > self.kl_warn_threshold:
            log.warning(
                "dpo_kl_high",
                step=state.global_step,
                implicit_kl=round(implicit_kl, 4),
                threshold=self.kl_warn_threshold,
                msg=(
                    f"Implicit KL ({implicit_kl:.2f}) exceeds threshold ({self.kl_warn_threshold}). "
                    f"Policy is drifting far from SFT reference. Consider:\n"
                    f"  - Reducing learning_rate (currently check config)\n"
                    f"  - Increasing beta (currently {self.beta})\n"
                    f"  - Stopping early at the current checkpoint."
                ),
            )

    def mean_kl(self) -> float:
        """Mean implicit KL over all logged steps."""
        return sum(self._step_klvals) / len(self._step_klvals) if self._step_klvals else 0.0


class DivergenceGuardCallback:
    """Stop training early if the implicit KL diverges beyond recovery.

    Uses a sliding window of KL values. If the window mean exceeds
    kl_stop_threshold, sets control.should_training_stop = True.

    This is a last-resort guard. You should first tune beta and LR so
    KL stays bounded without needing the guard.
    """

    def __init__(
        self,
        beta: float,
        kl_stop_threshold: float = 15.0,
        window_size: int = 20,
    ) -> None:
        self.beta = beta
        self.kl_stop_threshold = kl_stop_threshold
        self.window_size = window_size
        self._kl_window: list[float] = []

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        r_chosen = logs.get("rewards/chosen")
        r_rejected = logs.get("rewards/rejected")
        if r_chosen is None or r_rejected is None:
            return

        kl = (abs(r_chosen) + abs(r_rejected)) / (2.0 * self.beta)
        self._kl_window.append(kl)
        if len(self._kl_window) > self.window_size:
            self._kl_window.pop(0)

        window_mean = sum(self._kl_window) / len(self._kl_window)

        if len(self._kl_window) >= self.window_size and window_mean > self.kl_stop_threshold:
            log.error(
                "dpo_divergence_stopping",
                step=state.global_step,
                window_kl=round(window_mean, 4),
                threshold=self.kl_stop_threshold,
                msg=(
                    "DPO divergence detected. Training stopped early. "
                    "Increase beta or decrease learning_rate and retry."
                ),
            )
            control.should_training_stop = True
