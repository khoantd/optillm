"""Gradio chat UI for the OptiLLM proxy."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_ROUTING_HINT = (
    "Write a full math, logic, or code problem—not a technique label. "
    "Or pick a specific approach above instead of Auto."
)

# Gradio 5.15 Chatbot in Blocks triggers gradio_client schema bug (bool additionalProperties).
# ChatInterface avoids that; use OpenAI-style message history.
History = Union[List[Dict[str, str]], List[List[str]], None]


def approach_dropdown_choices(known_approaches: List[str]) -> List[str]:
    """Ordered approach slugs for the GUI selector."""
    primary = ["auto", "predict", "none"]
    rest = sorted(a for a in known_approaches if a not in primary)
    return primary + rest


def extract_optillm_routing(response: Any) -> Optional[Dict[str, Any]]:
    """Read optillm_routing from an OpenAI SDK chat completion response."""
    if response is None:
        return None
    if hasattr(response, "model_extra") and response.model_extra:
        meta = response.model_extra.get("optillm_routing")
        if meta:
            return meta
    if hasattr(response, "to_dict"):
        data = response.to_dict()
        if isinstance(data, dict):
            return data.get("optillm_routing")
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        if isinstance(data, dict):
            return data.get("optillm_routing")
    return None


def format_routing_markdown(meta: Optional[Dict[str, Any]]) -> str:
    """Human-readable routing summary for the GUI panel."""
    if not meta:
        return (
            "**Routing:** No metadata in the response "
            "(server may use a fixed `--approach`, or routing was not applied)."
        )

    resolved = meta.get("resolved_approach", "?")
    confidence = float(meta.get("confidence", 0) or 0)
    source = meta.get("source", "?")
    lines = [
        f"**Resolved:** `{resolved}` · **confidence:** {confidence:.1%} · **source:** `{source}`",
    ]
    if meta.get("requested_mode"):
        lines.append(f"**Requested mode:** `{meta['requested_mode']}`")
    if meta.get("override_reason"):
        lines.append(f"**Override:** `{meta['override_reason']}`")

    top = meta.get("top_scores") or []
    if top:
        parts = [
            f"`{item.get('approach', '?')}` {float(item.get('confidence', 0)):.1%}"
            for item in top[:5]
        ]
        lines.append("**Top scores:** " + ", ".join(parts))

    if resolved == "none" and source == "fallback":
        lines.append(f"\n*{_ROUTING_HINT}*")

    return "\n\n".join(lines)


def _content_from_turn(turn: Any) -> str:
    if turn is None:
        return ""
    if isinstance(turn, str):
        return turn
    if isinstance(turn, dict):
        content = turn.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
    return str(turn)


def _build_messages(
    message: str,
    history: History,
    system_prompt: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    sp = (system_prompt or "").strip()
    if sp:
        messages.append({"role": "system", "content": sp})

    for item in history or []:
        if isinstance(item, dict) and "role" in item:
            role = item.get("role")
            content = _content_from_turn(item)
            if role and content:
                messages.append({"role": str(role), "content": content})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_turn, assistant_turn = item[0], item[1]
            if user_turn:
                messages.append({"role": "user", "content": _content_from_turn(user_turn)})
            if assistant_turn:
                messages.append(
                    {"role": "assistant", "content": _content_from_turn(assistant_turn)}
                )

    messages.append({"role": "user", "content": message})
    return messages


def launch_gradio_gui(
    app,
    server_config: Dict[str, Any],
    port: int,
    host: str,
    known_approaches: List[str],
) -> None:
    """
    Start the Flask app in a background thread and launch the Gradio UI.

    ``app`` is the Flask application from server.py.
    """
    import gradio as gr
    import httpx
    from openai import OpenAI

    server_thread = threading.Thread(target=app.run, kwargs={"host": host, "port": port})
    server_thread.daemon = True
    server_thread.start()

    base_url = f"http://localhost:{port}/v1"
    gui_port = int(os.environ.get("OPTILLM_GUI_PORT", 7860))
    server_config["gui_url"] = f"http://{host}:{gui_port}"
    logger.info("Launching Gradio interface connected to %s", base_url)

    default_approach = server_config.get("approach", "auto")
    if default_approach not in approach_dropdown_choices(known_approaches):
        default_approach = "auto"
    approach_choices = approach_dropdown_choices(known_approaches)
    default_model = server_config.get("model", "gpt-4o-mini")

    def chat_with_optillm(
        message: str,
        history: History,
        system_prompt: str,
        approach_slug: str,
    ):
        if not (message or "").strip():
            return "", format_routing_markdown(None)

        custom_client = OpenAI(
            api_key="optillm",
            base_url=base_url,
            timeout=httpx.Timeout(1800.0, connect=5.0),
            max_retries=0,
        )

        messages = _build_messages(message, history, system_prompt)
        extra_body: Dict[str, Any] = {}
        slug = (approach_slug or "auto").strip()
        if slug:
            extra_body["optillm_approach"] = slug

        try:
            request_kwargs: Dict[str, Any] = {
                "model": default_model,
                "messages": messages,
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            response = custom_client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or ""
            routing_md = format_routing_markdown(extract_optillm_routing(response))
            return content, routing_md
        except Exception as exc:
            logger.exception("GUI chat request failed")
            return f"**Error:** {exc}", f"**Error:** {exc}"

    description = (
        f"Connected to OptiLLM at `{base_url}` · model `{default_model}` · "
        f"server default approach `{server_config.get('approach', 'auto')}`"
    )

    routing_panel = gr.Markdown(
        value="Send a message to see routing (resolved approach, confidence, top scores).",
    )
    approach_dd = gr.Dropdown(
        choices=approach_choices,
        value=default_approach,
        label="Approach",
        info="Auto uses the ML router when enabled on the server.",
    )
    system_prompt_tb = gr.Textbox(
        label="System prompt (optional)",
        placeholder="e.g. You are a helpful tutor. Think step by step.",
        lines=2,
    )

    demo = gr.ChatInterface(
        chat_with_optillm,
        type="messages",
        title="OptiLLM Chat",
        description=description,
        textbox=gr.Textbox(
            placeholder=(
                "Ask a full question, e.g. "
                "'In a family with two children, at least one is a boy. "
                "What is P(both boys)?'"
            ),
            container=False,
        ),
        additional_inputs=[system_prompt_tb, approach_dd],
        additional_inputs_accordion=gr.Accordion("OptiLLM settings", open=True),
        additional_outputs=[routing_panel],
    )

    demo.queue()
    demo.launch(server_name=host, server_port=gui_port, share=False)
