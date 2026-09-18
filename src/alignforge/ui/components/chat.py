"""Single-model chat tab with persistent conversation history."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import gradio as gr
import structlog

from alignforge.ui.client import get_models, stream_chat

log = structlog.get_logger()


def build_chat_tab(api_base: str) -> dict[str, Any]:
    """Build the single-chat tab."""
    with gr.Tab("💬 Chat"):
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["dpo"],
                value="dpo",
                label="Model",
                scale=2,
            )
            system_prompt = gr.Textbox(
                value="You are a helpful, direct, and concise developer assistant.",
                label="System prompt",
                scale=5,
            )
            clear_btn = gr.Button("Clear", scale=1)

        chatbot = gr.Chatbot(height=500)

        with gr.Row():
            msg_box = gr.Textbox(placeholder="Ask something...", show_label=False, scale=9)
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            temp_slider = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            maxtok_slider = gr.Slider(64, 1024, value=512, step=64, label="Max tokens")

    return {
        "chatbot": chatbot,
        "model_dropdown": model_dropdown,
        "system_prompt": system_prompt,
        "msg_box": msg_box,
        "send_btn": send_btn,
        "clear_btn": clear_btn,
        "temp_slider": temp_slider,
        "maxtok_slider": maxtok_slider,
    }


def wire_chat_events(components: dict[str, Any], api_base: str) -> None:
    """Attach event handlers to chat components."""

    async def on_chat(
        message: str,
        history: list[list[str | None]],
        model_id: str,
        system_p: str,
        temp: float,
        maxtok: int,
    ) -> AsyncIterator[tuple[list[list[str | None]], str]]:
        if not message.strip():
            yield history, ""
            return

        # Build messages list including history.
        messages: list[dict[str, str]] = []
        if system_p:
            messages.append({"role": "system", "content": system_p})
        for user_msg, asst_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if asst_msg:
                messages.append({"role": "assistant", "content": asst_msg})
        messages.append({"role": "user", "content": message})

        # Append partial response to history.
        history = [*history, [message, ""]]
        accumulated = ""

        async for delta in stream_chat(
            api_base=api_base,
            model_id=model_id,
            messages=messages,
            temperature=temp,
            max_tokens=int(maxtok),
        ):
            accumulated += delta
            history[-1][1] = accumulated
            yield history, ""  # clear input while streaming

    for trigger in [components["send_btn"].click, components["msg_box"].submit]:
        trigger(
            fn=on_chat,
            inputs=[
                components["msg_box"],
                components["chatbot"],
                components["model_dropdown"],
                components["system_prompt"],
                components["temp_slider"],
                components["maxtok_slider"],
            ],
            outputs=[components["chatbot"], components["msg_box"]],
        )

    components["clear_btn"].click(
        fn=lambda: ([], ""),
        outputs=[components["chatbot"], components["msg_box"]],
    )

    async def refresh_models() -> gr.Dropdown:
        models = await get_models(api_base)
        choices = [m["id"] for m in models] or ["dpo"]
        return gr.Dropdown(choices=choices, value=choices[0] if choices else "dpo")

    components["model_dropdown"].focus(fn=refresh_models, outputs=[components["model_dropdown"]])
