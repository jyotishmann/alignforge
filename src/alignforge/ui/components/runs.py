"""Run browser tab — shows all training runs from the registry."""

from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd
import structlog

log = structlog.get_logger()


def build_runs_tab() -> dict[str, Any]:
    """Build the run browser tab."""
    with gr.Tab("🗂️ Runs"):
        gr.Markdown("### Training Run Registry\n*All runs from the AlignForge registry.*")
        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
        runs_table = gr.Dataframe(
            headers=["run_id", "kind", "status", "config_hash", "dataset_hash", "started_at"],
            datatype=["str", "str", "str", "str", "str", "str"],
            label="Runs",
            interactive=False,
        )
        gr.Markdown("---")
        gr.Markdown("### Run Detail")
        run_detail_md = gr.Markdown("*Click a run ID to see details.*")

    return {
        "refresh_btn": refresh_btn,
        "runs_table": runs_table,
        "run_detail_md": run_detail_md,
    }


def wire_runs_events(components: dict[str, Any]) -> None:
    """Attach events to the run browser."""

    def refresh_runs():
        from alignforge.core.registry import get_registry

        reg = get_registry()
        runs = reg.list_runs(limit=50)
        if not runs:
            return pd.DataFrame(
                columns=["run_id", "kind", "status", "config_hash", "dataset_hash", "started_at"]
            )
        rows = []
        for r in runs:
            rows.append(
                {
                    "run_id": r["run_id"],
                    "kind": r["kind"],
                    "status": r["status"],
                    "config_hash": (r.get("config_hash") or "")[:12],
                    "dataset_hash": (r.get("dataset_hash") or "")[:12],
                    "started_at": (r.get("started_at") or "")[:19],
                }
            )
        return pd.DataFrame(rows)

    components["refresh_btn"].click(fn=refresh_runs, outputs=[components["runs_table"]])

    def on_select(evt: gr.SelectData, df: pd.DataFrame) -> str:
        if evt.index is None or len(df) == 0:
            return "*Select a row to see run details.*"
        row_idx = evt.index[0]
        if row_idx >= len(df):
            return "*Row out of range.*"
        run_id = str(df.iloc[row_idx]["run_id"])
        from alignforge.core.registry import get_registry

        reg = get_registry()
        run = reg.get_run(run_id)
        if not run:
            return f"*Run {run_id!r} not found.*"
        arts = reg.get_artifacts(run_id)
        lines = [
            f"## Run: `{run_id}`",
            f"- **Kind:** {run['kind']}",
            f"- **Status:** {run['status']}",
            f"- **Config hash:** `{run.get('config_hash', '')}`",
            f"- **Dataset hash:** `{run.get('dataset_hash', '')}`",
            f"- **Git SHA:** `{run.get('git_sha', 'unknown')}`",
            f"- **Started:** {(run.get('started_at') or '')[:19]}",
            f"- **Finished:** {(run.get('finished_at') or 'running')[:19]}",
            "",
            "**Artifacts:**",
        ]
        for a in arts:
            lines.append(f"- `{a['kind']}`: `{a['path'][:60]}`")
        if run.get("metrics_json"):
            lines += ["", "**Final metrics:**", f"```json\n{run['metrics_json']}\n```"]
        return "\n".join(lines)

    components["runs_table"].select(
        fn=on_select,
        inputs=[components["runs_table"]],
        outputs=[components["run_detail_md"]],
    )
