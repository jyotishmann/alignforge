"""Evaluation results browser tab."""

from __future__ import annotations

from typing import Any

import gradio as gr
import structlog

from alignforge.ui.client import get_http_client

log = structlog.get_logger()


def build_evals_tab(api_base: str) -> dict[str, Any]:
    """Build the evaluation browser tab."""
    with gr.Tab("📊 Evaluations"):
        gr.Markdown(
            "### Evaluation Results\n*Win rates with 95% bootstrap CI. Pulled from the API.*"
        )

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            eval_dropdown = gr.Dropdown(choices=[], label="Select eval run", scale=4)

        # Win-rate chart.
        win_rate_plot = gr.Plot(label="Win Rate (with 95% CI)", visible=False)
        metrics_md = gr.Markdown("*Load an eval run to see results.*")

        gr.Markdown("---")
        gr.Markdown("### Case Browser")
        gr.Markdown("*Select a case to see all responses and the judge rationale.*")

        case_dropdown = gr.Dropdown(choices=[], label="Case ID", interactive=True)
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", lines=3, interactive=False)
            intent_box = gr.Textbox(label="Intent", lines=2, interactive=False)
        with gr.Row():
            base_box = gr.Textbox(label="Base response", lines=6, interactive=False)
            sft_box = gr.Textbox(label="SFT response", lines=6, interactive=False)
            dpo_box = gr.Textbox(label="DPO response", lines=6, interactive=False)
        rationale_box = gr.Textbox(label="Judge rationale", lines=4, interactive=False)

    return {
        "refresh_btn": refresh_btn,
        "eval_dropdown": eval_dropdown,
        "win_rate_plot": win_rate_plot,
        "metrics_md": metrics_md,
        "case_dropdown": case_dropdown,
        "prompt_box": prompt_box,
        "intent_box": intent_box,
        "base_box": base_box,
        "sft_box": sft_box,
        "dpo_box": dpo_box,
        "rationale_box": rationale_box,
    }


def wire_evals_events(components: dict[str, Any], api_base: str) -> None:
    """Attach events to the eval browser."""

    async def refresh_evals():
        client = get_http_client(api_base)
        try:
            r = await client.get("/v1/evals", timeout=5.0)
            evals = r.json()
            choices = [e["eval_id"] for e in evals]
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
        except Exception as exc:
            log.warning("evals_fetch_failed", error=str(exc))
            return gr.Dropdown(choices=[])

    async def load_eval(eval_id: str) -> tuple[Any, str, Any, bool]:
        if not eval_id:
            return None, "*No eval selected.*", gr.Dropdown(choices=[]), True

        client = get_http_client(api_base)
        try:
            r = await client.get(f"/v1/evals/{eval_id}", timeout=10.0)
            metrics = r.json()
        except Exception as exc:
            return None, f"*Error loading eval: {exc}*", gr.Dropdown(choices=[]), True

        # Build Plotly figure for win rates.
        try:
            import plotly.graph_objects as go

            pairwise = metrics.get("pairwise", {})
            pairs = list(pairwise.keys())
            win_rates = [pairwise[p]["win_rate"] for p in pairs]
            ci_lows = [pairwise[p]["win_rate"] - pairwise[p]["ci_low"] for p in pairs]
            ci_highs = [pairwise[p]["ci_high"] - pairwise[p]["win_rate"] for p in pairs]

            fig = go.Figure(
                go.Bar(
                    x=pairs,
                    y=win_rates,
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": ci_highs,
                        "arrayminus": ci_lows,
                    },
                    marker_color=["#2ecc71" if w > 0.5 else "#e74c3c" for w in win_rates],
                )
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(
                title=f"Win Rates — eval {eval_id}",
                yaxis_title="Win Rate",
                yaxis_range=[0, 1],
                height=350,
            )
        except ImportError:
            fig = None

        # Markdown summary.
        elo = metrics.get("elo", {})
        elo_str = " | ".join(f"`{m}` {s}" for m, s in sorted(elo.items(), key=lambda x: -x[1]))
        pb = metrics.get("position_bias_rate", {})
        pb_str = " | ".join(f"`{k}` {round(v * 100)}%" for k, v in pb.items())
        md = f"**Elo:** {elo_str}\n\n**Position bias rate:** {pb_str}"

        return fig, md, gr.Dropdown(choices=[]), False

    components["refresh_btn"].click(fn=refresh_evals, outputs=[components["eval_dropdown"]])
    components["eval_dropdown"].change(
        fn=load_eval,
        inputs=[components["eval_dropdown"]],
        outputs=[
            components["win_rate_plot"],
            components["metrics_md"],
            components["case_dropdown"],
            components["win_rate_plot"],
        ],
    )
