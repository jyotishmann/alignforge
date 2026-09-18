"""Gradio application factory — assembles all tabs."""

from __future__ import annotations

import gradio as gr

from alignforge.ui.components.arena import build_arena_tab, wire_arena_events
from alignforge.ui.components.chat import build_chat_tab, wire_chat_events
from alignforge.ui.components.evals import build_evals_tab, wire_evals_events
from alignforge.ui.components.runs import build_runs_tab, wire_runs_events
from alignforge.ui.state import ArenaSessionState


def create_ui(api_base: str = "http://localhost:8000") -> gr.Blocks:
    """Build and return the complete Gradio application."""

    with gr.Blocks(
        title="AlignForge",
        theme=gr.themes.Soft(),
        css="""
        .telemetry { font-size: 0.75em; color: #888; margin-top: 4px; }
        .gr-button.primary { background: #2ecc71; }
        """,
    ) as demo:
        gr.Markdown(
            f"# ⚡ AlignForge\n*Three-way comparison arena: Base · SFT · DPO*\n\nAPI: `{api_base}`"
        )

        # Per-session state.
        session_state = gr.State(ArenaSessionState())

        # Build all tabs.
        arena_components = build_arena_tab(session_state, api_base)
        chat_components = build_chat_tab(api_base)
        evals_components = build_evals_tab(api_base)
        runs_components = build_runs_tab()

        # Wire events.
        wire_arena_events(arena_components, session_state, api_base)
        wire_chat_events(chat_components, api_base)
        wire_evals_events(evals_components, api_base)
        wire_runs_events(runs_components)

        # On load: check API health and populate model list.
        async def on_load(sess: ArenaSessionState) -> tuple[ArenaSessionState, str]:
            import copy

            from alignforge.ui.client import check_health, get_models

            health = await check_health(api_base)
            if health.get("status") == "unavailable":
                return sess, f"⚠️ API unavailable at {api_base}. Start with: `alignforge serve api`"

            models = await get_models(api_base)
            if models:
                new_sess = copy.deepcopy(sess)
                new_sess.available_models = models
                new_sess.column_model_ids = [m["id"] for m in models[:3]]
                for i, m in enumerate(models[:3]):
                    new_sess.telemetry[i].model_id = m["id"]
                    new_sess.telemetry[i].display_name = m.get("display_name", m["id"])
                return new_sess, f"✓ Connected to API — {len(models)} model(s) available"
            return sess, "⚠️ No models registered. Run `alignforge registry publish` first."

        status_banner = gr.Markdown("")
        demo.load(fn=on_load, inputs=[session_state], outputs=[session_state, status_banner])

    return demo  # type: ignore[no-any-return]
