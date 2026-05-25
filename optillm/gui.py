"""Gradio chat UI for the OptiLLM proxy."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from optillm.reasoning_trace import (
    ReasoningSplit,
    format_chat_with_reasoning,
    format_thinking_markdown,
    iter_reveal_stream,
    merge_reasoning_stream,
    split_thinking_and_answer,
)

logger = logging.getLogger(__name__)

_GUI_TESTING_DOC = "tests/GUI_TESTING.md"

_ROUTING_HINT = (
    "Auto fell back to plain `none` (low router confidence). "
    "Load a sample below or set Approach to a specific slug."
)

# Slug -> test_cases.json ``name`` values (see tests/GUI_TESTING.md).
_APPROACH_SAMPLE_NAMES: Dict[str, List[str]] = {
    "moa": ["Arena Bench Hard"],
    "bon": ["GSM8K", "Simple Math Problem"],
    "cot_reflection": [
        "GenSelect Math",
        "Reasoning Token Test - Complex Logic",
        "Reasoning Token Test - Multi-Step Problem",
    ],
    "mcts": ["Reasoning Token Test - Strategic Thinking"],
    "plansearch": ["Maths Problem", "Reasoning Token Test - Algorithm Design"],
    "mars": ["GH"],
    "z3": ["r/LocalLLaMA"],
    "self_consistency": ["Reasoning Token Test - Counter-intuitive"],
}

_RE2_SAMPLE_QUERY = "How many r's are there in strawberry?"

# Gradio 5.15 Chatbot in Blocks triggers gradio_client schema bug (bool additionalProperties).
# ChatInterface avoids that; use OpenAI-style message history.
History = Union[List[Dict[str, str]], List[List[str]], None]

GuiSample = Dict[str, str]


def default_test_cases_path() -> Path:
    """Path to tests/test_cases.json relative to the repo root."""
    return Path(__file__).resolve().parent.parent / "tests" / "test_cases.json"


def load_test_cases(path: Optional[os.PathLike[str]] = None) -> List[Dict[str, Any]]:
    """Load test case rows from JSON."""
    json_path = Path(path) if path else default_test_cases_path()
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}")
    return data


def build_gui_samples(
    cases: Optional[List[Dict[str, Any]]] = None,
    *,
    known_approaches: Optional[List[str]] = None,
) -> List[GuiSample]:
    """
    Build labeled presets for the GUI sample dropdown.

    Each sample includes slug, query, system_prompt, and a stable ``id``.
    """
    rows = cases if cases is not None else load_test_cases()
    by_name = {str(row.get("name", "")): row for row in rows}
    samples: List[GuiSample] = []

    for slug, names in _APPROACH_SAMPLE_NAMES.items():
        if known_approaches is not None and slug not in known_approaches:
            continue
        for name in names:
            row = by_name.get(name)
            if not row:
                logger.warning("GUI sample missing test case: %s", name)
                continue
            samples.append(
                {
                    "id": f"{slug}::{name}",
                    "label": f"{slug} — {name}",
                    "slug": slug,
                    "name": name,
                    "query": str(row.get("query", "")),
                    "system_prompt": str(row.get("system_prompt", "") or ""),
                }
            )

    if known_approaches is None or "re2" in known_approaches:
        samples.append(
            {
                "id": "re2::strawberry",
                "label": "re2 — strawberry (README)",
                "slug": "re2",
                "name": "strawberry",
                "query": _RE2_SAMPLE_QUERY,
                "system_prompt": "",
            }
        )

    return samples


def samples_by_id(samples: List[GuiSample]) -> Dict[str, GuiSample]:
    return {s["id"]: s for s in samples}


def format_testing_guide_ui_markdown() -> str:
    """Condensed in-UI copy of tests/GUI_TESTING.md."""
    rows = [
        "| Goal | Set **Approach** to | Then |",
        "|------|---------------------|------|",
        "| Run a specific technique | slug (`moa`, `z3`, …) | Load a **sample** → copy message into chat |",
        "| Test ML routing | `auto` or `predict` | Load sample (full problem only; not meta labels) |",
    ]
    manual = [
        "| Technique | Slug | Sample in dropdown |",
        "|-----------|------|---------------------|",
    ]
    for slug, names in _APPROACH_SAMPLE_NAMES.items():
        manual.append(f"| {slug} | `{slug}` | {', '.join(names[:2])}{'…' if len(names) > 2 else ''} |")
    manual.append(f"| ReRead | `re2` | strawberry (README) |")

    return "\n".join(
        [
            "### How to test",
            *rows,
            "",
            "**Do not** type category lines like `Hard reasoning: prove √2…` — use **Load sample**.",
            "",
            "### Manual approach → sample",
            *manual,
            "",
            "### Auto mode",
            "Default router needs **≥30%** confidence or it uses plain `none`. "
            "Most samples still fall back; **Algorithm design** often routes to `z3`.",
            "",
            f"Full doc: `{_GUI_TESTING_DOC}`",
        ]
    )


def sample_dropdown_choices(
    samples: List[GuiSample],
    approach_slug: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Return (label, id) pairs for gr.Dropdown, optionally filtered by approach."""
    filtered = samples
    if approach_slug and approach_slug not in ("auto", "predict", "none"):
        filtered = [s for s in samples if s["slug"] == approach_slug]
    choices = [(s["label"], s["id"]) for s in filtered]
    return [("", "")] + choices if not choices else choices


def approach_dropdown_choices(known_approaches: List[str]) -> List[str]:
    """Ordered approach slugs for the GUI selector."""
    primary = ["auto", "predict", "none"]
    rest = sorted(a for a in known_approaches if a not in primary)
    return primary + rest


def extract_routing_from_stream_chunk(chunk: Any) -> Optional[Dict[str, Any]]:
    """Read optillm_routing from an OpenAI streaming chunk if present."""
    if chunk is None:
        return None
    if hasattr(chunk, "model_extra") and chunk.model_extra:
        meta = chunk.model_extra.get("optillm_routing")
        if meta:
            return meta
    if hasattr(chunk, "model_dump"):
        data = chunk.model_dump()
        if isinstance(data, dict) and data.get("optillm_routing"):
            return data["optillm_routing"]
    return None


def extract_optillm_routing(response: Any) -> Optional[Dict[str, Any]]:
    """Read optillm_routing from an OpenAI SDK chat completion response."""
    if response is None:
        return None
    if hasattr(response, "model_extra") and response.model_extra:
        meta = response.model_extra.get("optillm_routing")
        if meta:
            return meta
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        if isinstance(data, dict):
            return data.get("optillm_routing")
    if hasattr(response, "to_dict"):
        data = response.to_dict()
        if isinstance(data, dict):
            return data.get("optillm_routing")
    return None


def format_routing_markdown(meta: Optional[Dict[str, Any]]) -> str:
    """Human-readable routing summary for the GUI panel."""
    if not meta:
        return (
            "**Routing:** No metadata in the response "
            "(fixed approach, or routing was not applied)."
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


def _apply_gui_sample_plain(
    sample_id: Optional[str],
    catalog: Dict[str, GuiSample],
) -> Tuple[str, str, str, str]:
    """Non-Gradio apply for tests and callbacks."""
    if not sample_id or sample_id not in catalog:
        return "", "", "", "**Sample:** choose a row to fill fields."

    sample = catalog[sample_id]
    hint = (
        f"**Loaded:** `{sample['slug']}` · *{sample['name']}* — "
        "copy the **Sample message** into the chat box below, then send."
    )
    return sample["slug"], sample["system_prompt"], sample["query"], hint


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
    from gradio.chat_interface import ChatInterface
    from gradio.layouts import Column
    import httpx
    from openai import OpenAI

    _GUI_CSS = """
    #optillm-routing-panel, #optillm-thinking-panel {
        min-height: 4.5rem;
        padding: 0.75rem 1rem;
        margin: 0.35rem 0;
        border: 1px solid var(--border-color-primary, #e5e7eb);
        border-radius: 0.5rem;
        background: var(--background-fill-secondary, #f9fafb);
    }
    #optillm-status-accordion { margin-top: 0.5rem; margin-bottom: 0.5rem; }
    """

    class OptillmChatInterface(ChatInterface):
        """ChatInterface with routing/reasoning below the chatbot."""

        def __init__(self, *args, extra_header_components=None, **kwargs):
            self._extra_header_components = extra_header_components or []
            self._sample_dd = kwargs.pop("sample_dd", None)
            self._approach_dd = kwargs.pop("approach_dd", None)
            self._system_prompt_tb = kwargs.pop("system_prompt_tb", None)
            self._sample_query_tb = kwargs.pop("sample_query_tb", None)
            self._sample_hint_md = kwargs.pop("sample_hint_md", None)
            self._on_sample_change = kwargs.pop("on_sample_change", None)
            self._on_approach_change = kwargs.pop("on_approach_change", None)
            super().__init__(*args, **kwargs)

        def _render_header(self):
            if self._extra_header_components:
                with Column():
                    for component in self._extra_header_components:
                        if not component.is_rendered:
                            component.render()
            super()._render_header()

        def _render_footer(self):
            if self.additional_outputs:
                with gr.Accordion("Routing & reasoning (live)", open=True, elem_id="optillm-status-accordion"):
                    for component in self.additional_outputs:
                        if not component.is_rendered:
                            component.render()
            super()._render_footer()

        def _setup_events(self) -> None:
            super()._setup_events()
            if self._sample_dd is not None and self._on_sample_change is not None:
                self._sample_dd.change(
                    self._on_sample_change,
                    inputs=[self._sample_dd],
                    outputs=[
                        self._approach_dd,
                        self._system_prompt_tb,
                        self._sample_query_tb,
                        self._sample_hint_md,
                    ],
                )
            if self._approach_dd is not None and self._on_approach_change is not None:
                self._approach_dd.change(
                    self._on_approach_change,
                    inputs=[self._approach_dd],
                    outputs=[
                        self._sample_dd,
                        self._sample_query_tb,
                        self._sample_hint_md,
                    ],
                )

    if not server_config.get("return_full_response"):
        server_config["return_full_response"] = True
        logger.info(
            "GUI enabled return_full_response so reasoning tags are available for streaming"
        )

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

    gui_samples = build_gui_samples(known_approaches=known_approaches)
    sample_catalog = samples_by_id(gui_samples)
    initial_sample_choices = sample_dropdown_choices(gui_samples, default_approach)

    def chat_with_optillm(
        message: str,
        history: History,
        _sample_id: str,
        _sample_query: str,
        _sample_hint: str,
        approach_slug: str,
        system_prompt: str,
    ) -> Generator[Tuple[str, str, str], None, None]:
        del _sample_id, _sample_query, _sample_hint
        if not (message or "").strip():
            yield "", format_routing_markdown(None), format_thinking_markdown("", phase="idle")
            return

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

        routing_md = format_routing_markdown(None)
        waiting = ReasoningSplit("", "", False)
        yield (
            format_chat_with_reasoning(waiting, phase="waiting"),
            routing_md,
            format_thinking_markdown("", phase="waiting"),
        )

        try:
            request_kwargs: Dict[str, Any] = {
                "model": default_model,
                "messages": messages,
                "stream": True,
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            raw_accum = ""
            reasoning_accum = ""
            routing_meta: Optional[Dict[str, Any]] = None
            live_updates = 0

            stream = custom_client.chat.completions.create(**request_kwargs)
            for chunk in stream:
                routing_meta = extract_routing_from_stream_chunk(chunk) or routing_meta
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "reasoning_content", None):
                    reasoning_accum += delta.reasoning_content or ""
                if getattr(delta, "content", None):
                    raw_accum += delta.content or ""

                merged = merge_reasoning_stream(raw_accum, reasoning_accum)
                routing_md = format_routing_markdown(routing_meta)
                thinking_md = format_thinking_markdown(
                    merged.thinking,
                    phase="streaming",
                )
                chat = format_chat_with_reasoning(merged, phase="streaming")
                yield chat, routing_md, thinking_md
                live_updates += 1

            merged = merge_reasoning_stream(raw_accum, reasoning_accum)
            if not merged.answer.strip() and raw_accum.strip():
                merged = split_thinking_and_answer(raw_accum)
            routing_md = format_routing_markdown(routing_meta)

            # Server sends chunks after inference finishes — animate reveal in the UI.
            if live_updates <= 2 and (merged.thinking or merged.answer):
                yield from iter_reveal_stream(merged, routing_md)
            else:
                thinking_md = (
                    format_thinking_markdown(merged.thinking, phase="done")
                    if merged.thinking.strip()
                    else format_thinking_markdown("", phase="empty")
                )
                chat = format_chat_with_reasoning(merged, phase="done")
                yield chat, routing_md, thinking_md
        except Exception as exc:
            logger.exception("GUI chat request failed")
            err = f"**Error:** {exc}"
            yield err, err, format_thinking_markdown("", phase="empty")

    def on_sample_change(sample_id: Optional[str]):
        if not sample_id:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                "**Sample:** choose a row to fill approach, system prompt, and message text.",
            )
        slug, sp, query, hint = _apply_gui_sample_plain(sample_id, sample_catalog)
        return slug, sp, query, hint

    def on_approach_change(approach_slug: str):
        import gradio as gr

        choices = sample_dropdown_choices(gui_samples, approach_slug or None)
        if approach_slug in ("auto", "predict", "none", ""):
            info = "All benchmark samples (for auto, use full query text only)."
        else:
            info = f"Samples recommended for `{approach_slug}`."
        return (
            gr.Dropdown(choices=choices, value=None, label="Load sample", info=info),
            "",
            f"**Approach:** `{approach_slug or 'auto'}` — pick a sample or paste your own question.",
        )

    description = (
        f"Connected to OptiLLM at `{base_url}` · model `{default_model}` · "
        f"default approach `{server_config.get('approach', 'auto')}`. "
        "Reasoning streams in the chat and in **Routing & reasoning** below the chat. "
        "Use **Testing guide** (accordion) to load benchmark prompts."
    )

    routing_panel = gr.Markdown(
        value="Send a message to see routing (resolved approach, confidence, top scores).",
        elem_id="optillm-routing-panel",
        render=False,
    )
    thinking_panel = gr.Markdown(
        value=format_thinking_markdown("", phase="idle"),
        elem_id="optillm-thinking-panel",
        render=False,
    )
    guide_panel = gr.Markdown(value=format_testing_guide_ui_markdown(), render=False)
    sample_hint_md = gr.Markdown(
        value="**Sample:** choose a preset to fill approach, system prompt, and message text.",
        render=False,
    )
    sample_dd = gr.Dropdown(
        choices=initial_sample_choices,
        value=None,
        label="Load sample",
        info="Benchmark prompts from tests/test_cases.json (filtered when you pick an approach).",
        render=False,
    )
    sample_query_tb = gr.Textbox(
        label="Sample message (copy into chat)",
        lines=4,
        max_lines=12,
        interactive=True,
        placeholder="Select a sample above, then copy this text into the message box and send.",
        render=False,
    )
    approach_dd = gr.Dropdown(
        choices=approach_choices,
        value=default_approach,
        label="Approach",
        info="Slug = run that technique. Auto/predict = ML router (needs a full problem).",
        render=False,
    )
    system_prompt_tb = gr.Textbox(
        label="System prompt (optional)",
        placeholder="Filled automatically when you load a sample.",
        lines=2,
        render=False,
    )

    demo = OptillmChatInterface(
        chat_with_optillm,
        type="messages",
        title="OptiLLM Chat",
        description=description,
        css=_GUI_CSS,
        textbox=gr.Textbox(
            placeholder="Copy text from **Sample message** in the accordion, or type your own full question.",
            container=False,
        ),
        additional_inputs=[
            sample_dd,
            sample_query_tb,
            sample_hint_md,
            approach_dd,
            system_prompt_tb,
        ],
        additional_inputs_accordion=gr.Accordion("Testing guide & settings", open=True),
        additional_outputs=[routing_panel, thinking_panel],
        extra_header_components=[guide_panel],
        sample_dd=sample_dd,
        approach_dd=approach_dd,
        system_prompt_tb=system_prompt_tb,
        sample_query_tb=sample_query_tb,
        sample_hint_md=sample_hint_md,
        on_sample_change=on_sample_change,
        on_approach_change=on_approach_change,
    )

    demo.queue()
    demo.launch(server_name=host, server_port=gui_port, share=False)
