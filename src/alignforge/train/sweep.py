"""Beta sweep: run DPO at multiple beta values and compare metrics."""

from __future__ import annotations

from typing import Any

import structlog

from alignforge.core.config import AlignForgeConfig, load_config

log = structlog.get_logger()


def run_beta_sweep(
    base_cfg: AlignForgeConfig,
    sft_run_id: str,
    preference_hash: str,
    betas: list[float],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Run DPO at each beta value and return a list of result dicts.

    Each result contains: beta, run_id, mean_kl, rewards_margin (final).
    The win-rate for each checkpoint is added later after eval (Part 08).
    """
    from alignforge.train.dpo_run import run_dpo

    results: list[dict[str, Any]] = []
    for beta in betas:
        log.info("sweep_run_starting", beta=beta)
        # Override beta in a fresh config.
        run_cfg = load_config(
            overrides=[
                f"dpo.beta={beta}",
                f"dpo.num_train_epochs={base_cfg.dpo.num_train_epochs}",
            ]
        )
        try:
            run_id = run_dpo(
                cfg=run_cfg,
                sft_run_id=sft_run_id,
                preference_dataset_hash=preference_hash,
                limit=limit,
            )
            results.append({"beta": beta, "run_id": run_id, "status": "completed"})
            log.info("sweep_run_done", beta=beta, run_id=run_id)
        except Exception as exc:
            log.error("sweep_run_failed", beta=beta, error=str(exc))
            results.append({"beta": beta, "run_id": None, "status": "failed", "error": str(exc)})

    _write_sweep_table(results)
    return results


def _write_sweep_table(results: list[dict[str, Any]]) -> None:
    """Write a Markdown table of sweep results to reports/beta_sweep.md."""
    from alignforge.core.paths import get_paths
    # import json

    paths = get_paths()
    out = paths.reports_dir / "beta_sweep.md"

    lines = [
        "# β Sweep Results\n",
        "| β | run_id | status | win_rate | mean_kl |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        win_rate = r.get("win_rate", "pending eval")
        mean_kl = r.get("mean_kl", "—")
        lines.append(
            f"| {r['beta']} | {r.get('run_id', '—')} | {r['status']} | {win_rate} | {mean_kl} |"
        )
    lines += [
        "\n`win_rate` is filled in after `alignforge eval all` runs on each checkpoint.",
        "\nSee `docs/dpo_math.md` for the β trade-off analysis.\n",
    ]

    with out.open("w") as f:
        f.write("\n".join(lines))

    log.info("sweep_table_written", path=str(out))
