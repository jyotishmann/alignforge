"""Three-way concurrent streaming arena."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import gradio as gr
import structlog

from alignforge.ui.client import post_vote, stream_chat
from alignforge.ui.state import ArenaSessionState, ColumnTelemetry

log = structlog.get_logger()

_DONE = object()  # per-column sentinel


async def three_way_stream(
    prompt: str,
    state: ArenaSessionState,
    api_base: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> AsyncIterator[tuple[list[list[Any]], list[str], ArenaSessionState]]:
    """Concurrent three-way streaming generator.

    Yields (chatbot_histories, telemetry_lines, updated_state) after every
    token from any column.

    chatbot_histories: list of 3 Gradio chatbot histories
      [[[user_msg, assistant_partial]], [[...]], [[...]]]
    telemetry_lines: list of 3 telemetry strings for display below each column
    """
    # Queue carries (column_index: int, delta: str | None | _DONE_sentinel)
    queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()

    # Initialise accumulated state.
    histories: list[list[list[str | None]]] = [[[prompt, ""]] for _ in range(3)]
    telemetry: list[ColumnTelemetry] = [
        ColumnTelemetry(
            model_id=state.model_at(col),
            display_name=state.label_for(col),
            t_start=time.monotonic(),
        )
        for col in range(3)
    ]
    pending = set(range(3))  # columns not yet done

    async def pump(col_idx: int) -> None:
        """Open SSE stream for one column, push deltas to the shared queue."""
        model_id = state.model_at(col_idx)
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        try:
            async for delta in stream_chat(
                api_base=api_base,
                model_id=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                request_id=req_id,
            ):
                await queue.put((col_idx, delta))
        except Exception as exc:
            error_msg = f"[Error: {exc}]"
            await queue.put((col_idx, f"__ERROR__:{error_msg}"))
        finally:
            await queue.put((col_idx, _DONE))

    # Launch all three pumps concurrently.
    tasks = [asyncio.create_task(pump(col)) for col in range(3)]

    try:
        while pending:
            col_idx, value = await queue.get()

            if value is _DONE:
                pending.discard(col_idx)
                continue

            if isinstance(value, str) and value.startswith("__ERROR__:"):
                error_text = value[10:]
                telemetry[col_idx].error = error_text
                histories[col_idx][0][1] = f"*{error_text}*"
            else:
                # It's a content delta.
                delta = str(value)
                if telemetry[col_idx].t_first_token is None:
                    telemetry[col_idx].t_first_token = time.monotonic()
                telemetry[col_idx].n_tokens += 1
                histories[col_idx][0][1] = (histories[col_idx][0][1] or "") + delta

            # Yield updated state to Gradio.
            telem_lines = [t.telemetry_line() for t in telemetry]
            yield histories, telem_lines, state

    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    # Final yield with updated responses in state.
    import copy

    final_state = copy.deepcopy(state)
    final_state.last_responses = [str(h[0][1] or "") for h in histories]
    final_state.current_prompt = prompt
    final_state.telemetry = telemetry
    yield histories, [t.telemetry_line() for t in telemetry], final_state


def build_arena_tab(
    state: gr.State,
    api_base: str,
) -> dict[str, Any]:
    """Build the arena tab components. Returns a dict of component references."""

    with gr.Tab("⚔️ Arena"):
        with gr.Row():
            blind_toggle = gr.Checkbox(
                value=True,
                label="Blind mode (hide model identities until after voting)",
            )

        # Three-column chatbot display.
        with gr.Row():
            chatbots = []
            col_labels = []
            telem_boxes = []
            for i in range(3):
                with gr.Column():
                    lbl = gr.Markdown(f"**Column {['A', ' B', 'C'][i]}**")
                    col_labels.append(lbl)
                    cb = gr.Chatbot(height=400, show_label=False)
                    chatbots.append(cb)
                    tb = gr.Markdown("", elem_classes=["telemetry"])
                    telem_boxes.append(tb)

        # Prompt input.
        with gr.Row():
            prompt_box = gr.Textbox(
                placeholder="Ask the models something...",
                show_label=False,
                scale=9,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # Decode parameters.
        with gr.Accordion("⚙️ Decode parameters", open=False), gr.Row():
            temp_slider = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            topp_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            maxtok_slider = gr.Slider(64, 1024, value=512, step=64, label="Max tokens")

        # Voting row.
        gr.Markdown("### Which response is best?")
        with gr.Row():
            vote_btns = [
                gr.Button(f"◀ {['A', 'B', 'C'][i]}", variant="secondary") for i in range(3)
            ]
            vote_tie = gr.Button("Tie", variant="secondary")
        vote_status = gr.Markdown("")

    return {
        "chatbots": chatbots,
        "col_labels": col_labels,
        "telem_boxes": telem_boxes,
        "prompt_box": prompt_box,
        "send_btn": send_btn,
        "temp_slider": temp_slider,
        "topp_slider": topp_slider,
        "maxtok_slider": maxtok_slider,
        "vote_btns": vote_btns,
        "vote_tie": vote_tie,
        "vote_status": vote_status,
        "blind_toggle": blind_toggle,
    }


def wire_arena_events(
    components: dict[str, Any],
    state: gr.State,
    api_base: str,
) -> None:
    """Attach event handlers to arena components."""

    chatbots = components["chatbots"]
    telem_boxes = components["telem_boxes"]
    # col_labels = components["col_labels"]

    async def on_send(
        prompt: str,
        sess: ArenaSessionState,
        temp: float,
        topp: float,
        maxtok: int,
    ) -> AsyncGenerator[tuple[Any, ...], None]:
        """Stream from all three columns concurrently."""
        if not prompt.strip():
            return

        # In blind mode, shuffle columns on each new prompt.
        if sess.blind_mode:
            sess = sess.shuffle_columns()

        async for histories, telem_lines, updated_sess in three_way_stream(
            prompt=prompt,
            state=sess,
            api_base=api_base,
            temperature=temp,
            top_p=topp,
            max_tokens=int(maxtok),
        ):
            yield (
                histories[0],
                histories[1],
                histories[2],
                telem_lines[0],
                telem_lines[1],
                telem_lines[2],
                updated_sess,
                "",  # clear the prompt box after send
            )

    components["send_btn"].click(
        fn=on_send,
        inputs=[
            components["prompt_box"],
            state,
            components["temp_slider"],
            components["topp_slider"],
            components["maxtok_slider"],
        ],
        outputs=[
            *chatbots,
            *telem_boxes,
            state,
            components["prompt_box"],
        ],
    )
    components["prompt_box"].submit(
        fn=on_send,
        inputs=[
            components["prompt_box"],
            state,
            components["temp_slider"],
            components["topp_slider"],
            components["maxtok_slider"],
        ],
        outputs=[*chatbots, *telem_boxes, state, components["prompt_box"]],
    )

    async def on_vote(winner_col: int, sess: ArenaSessionState) -> tuple[ArenaSessionState, str]:
        """Record a vote and reveal model identities."""
        if not sess.current_prompt or not any(sess.last_responses):
            return sess, "⚠️ Generate responses first."

        # Map visual column to actual model.
        winner_model = sess.model_at(winner_col)
        losers = [(i, sess.model_at(i)) for i in range(3) if i != winner_col]

        for loser_col, loser_model in losers:
            try:
                await post_vote(
                    api_base=api_base,
                    prompt=sess.current_prompt,
                    response_a=sess.last_responses[winner_col],
                    response_b=sess.last_responses[loser_col],
                    model_a=winner_model,
                    model_b=loser_model,
                    winner="a",
                    session_id=sess.session_id,
                )
            except Exception as exc:
                log.warning("vote_failed", error=str(exc))

        # Reveal identities after vote.
        reveal_lines = [
            f"**Col {['A', 'B', 'C'][v_col]}: `{sess.model_at(v_col)}`**" for v_col in range(3)
        ]
        vote_msg = f"✓ Voted for **{winner_model}**. " + " | ".join(reveal_lines)
        return sess, vote_msg

    async def on_tie_vote(sess: ArenaSessionState) -> tuple[ArenaSessionState, str]:
        return await on_vote(-1, sess)  # -1 signals tie

    for col_i, btn in enumerate(components["vote_btns"]):
        col_capture = col_i
        btn.click(
            fn=lambda s, c=col_capture: on_vote(c, s),
            inputs=[state],
            outputs=[state, components["vote_status"]],
        )

    components["vote_tie"].click(
        fn=lambda s: asyncio.get_event_loop().run_until_complete(on_tie_vote(s)),
        inputs=[state],
        outputs=[state, components["vote_status"]],
    )

    async def on_blind_toggle(enabled: bool, sess: ArenaSessionState) -> ArenaSessionState:
        import copy

        new_sess = copy.deepcopy(sess)
        new_sess.blind_mode = enabled
        if enabled:
            new_sess = new_sess.shuffle_columns()
        return new_sess

    components["blind_toggle"].change(
        fn=on_blind_toggle,
        inputs=[components["blind_toggle"], state],
        outputs=[state],
    )
