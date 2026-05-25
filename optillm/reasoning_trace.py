"""Parse and format model reasoning traces (think tags, CoT blocks) for UI and streaming."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple

# Build tag names without "\\r" in "\\redacted" inside string literals.
_REDACTED_THINKING = "redacted" + "_thinking"

# Common think-tag aliases (avoid "\\r" in "\\redacted" string literals).
_THINK_TAG_PAIRS: Tuple[Tuple[str, str], ...] = (
    (_REDACTED_THINKING, _REDACTED_THINKING),
    ("thinking", "thinking"),
    ("think", "think"),
    ("reflection", "reflection"),
)

def _complete_block_pattern(open_tag: str, close_tag: str) -> re.Pattern[str]:
    # Concatenate tags — never use rf"...</{tag}>" (\\r in \\redacted becomes carriage return).
    return re.compile(
        "<" + re.escape(open_tag) + ">(.*?)</" + re.escape(close_tag) + ">",
        re.DOTALL,
    )


def _open_tail_pattern(open_tag: str, close_tag: str) -> re.Pattern[str]:
    return re.compile(
        "<"
        + re.escape(open_tag)
        + ">(?!.*</"
        + re.escape(close_tag)
        + ">)(.*)$",
        re.DOTALL,
    )


_COMPLETE_BLOCK_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    _complete_block_pattern(open_tag, close_tag)
    for open_tag, close_tag in _THINK_TAG_PAIRS
)

_OPEN_TAIL_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    _open_tail_pattern(open_tag, close_tag)
    for open_tag, close_tag in _THINK_TAG_PAIRS
)

_OUTPUT_PATTERN = re.compile(r"<output>(.*?)(?:</output>|$)", re.DOTALL)
_TAG_NAMES = (_REDACTED_THINKING, "thinking", "think", "reflection", "output")
_TAG_STRIP_PATTERN = re.compile(
    "</?(?:" + "|".join(re.escape(name) for name in _TAG_NAMES) + ")>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReasoningSplit:
    """Thinking text, visible answer, and whether a think block is still open."""

    thinking: str
    answer: str
    in_progress: bool


def split_thinking_and_answer(text: str) -> ReasoningSplit:
    """
    Split a model response into reasoning trace and user-visible answer.

    Supports ``<think>``, ``<thinking>``, ``<reflection>``, and ``<output>``.
    """
    if not text or not isinstance(text, str):
        return ReasoningSplit("", "", False)

    thinking_parts: List[str] = []
    for pattern in _COMPLETE_BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            block = match.group(1).strip()
            if block:
                thinking_parts.append(block)

    in_progress = False
    for pattern in _OPEN_TAIL_PATTERNS:
        tail = pattern.search(text)
        if tail:
            fragment = tail.group(1).strip()
            if fragment:
                thinking_parts.append(fragment)
                in_progress = True
            break

    output_match = _OUTPUT_PATTERN.search(text)
    if output_match:
        answer = output_match.group(1).strip()
    else:
        answer = text
        for pattern in _COMPLETE_BLOCK_PATTERNS:
            answer = pattern.sub("", answer)
        for pattern in _OPEN_TAIL_PATTERNS:
            answer = pattern.sub("", answer)
        answer = _TAG_STRIP_PATTERN.sub("", answer).strip()

    thinking = "\n\n---\n\n".join(thinking_parts).strip()
    if not thinking and not in_progress and "<" in text and "thinking" in text.lower():
        in_progress = True

    return ReasoningSplit(thinking, answer, in_progress)


def merge_reasoning_stream(raw_text: str, reasoning_delta: str) -> ReasoningSplit:
    """Combine SSE reasoning_content deltas with tag-based parsing of content."""
    split = split_thinking_and_answer(raw_text)
    thinking = (reasoning_delta or "").strip() or split.thinking
    answer = split.answer
    in_progress = split.in_progress and not answer
    if thinking and not answer:
        in_progress = True
    return ReasoningSplit(thinking, answer, in_progress)


def chunk_text(text: str, *, size: int = 64) -> Iterator[str]:
    """Yield successive slices of ``text`` for simulated progressive reveal."""
    if not text:
        return
    for index in range(0, len(text), size):
        yield text[index : index + size]


def format_thinking_markdown(
    thinking: str,
    *,
    phase: str = "idle",
    char_count: int | None = None,
) -> str:
    """
    Markdown for the reasoning panel.

    ``phase`` is one of: idle, waiting, streaming, done, empty.
    """
    count = char_count if char_count is not None else len(thinking or "")

    if phase == "idle":
        tag = _REDACTED_THINKING
        return (
            "### Reasoning\n\n"
            "*Send a message to stream the model's reasoning trace here. "
            f"Works best with approaches that emit ``<{tag}>`` / "
            "``<thinking>`` tags (GUI enables full-response mode automatically).*"
        )
    if phase == "waiting":
        return (
            "### Reasoning\n\n"
            "⏳ **Running OptiLLM…** Waiting for the first reasoning tokens."
        )
    if phase == "empty":
        return (
            "### Reasoning\n\n"
            "*No separate reasoning trace in this response. "
            "Try ``cot_reflection``, ``re2``, or ThinkDeeper-style models, "
            "or check that the approach returns thinking tags.*"
        )
    if phase == "streaming":
        body = thinking.strip() if thinking else "…"
        return (
            "### Reasoning\n\n"
            f"⟳ **In progress** · {count:,} characters\n\n"
            f"```\n{body}\n```"
        )
    # done
    body = thinking.strip() if thinking else "_No reasoning content extracted._"
    return (
        "### Reasoning\n\n"
        f"✓ **Complete** · {count:,} characters\n\n"
        f"```\n{body}\n```"
    )


def format_chat_with_reasoning(
    split: ReasoningSplit,
    *,
    phase: str = "streaming",
) -> str:
    """
    Build the assistant chat bubble so reasoning is visible while the answer streams.

    Shown above the final answer in the main chatbot (not only the side panel).
    """
    if phase == "waiting":
        return (
            "⏳ **Generating…** OptiLLM is running the selected approach. "
            "Reasoning will stream here first, then the answer."
        )

    parts: List[str] = []
    if split.thinking.strip():
        label = "⟳ Reasoning" if phase == "streaming" else "✓ Reasoning"
        parts.append(f"**{label}**\n\n```\n{split.thinking.strip()}\n```")

    if split.answer.strip():
        parts.append(f"**Answer**\n\n{split.answer.strip()}")
    elif phase == "streaming" and split.thinking.strip():
        parts.append("*Answer will appear below when reasoning finishes…*")

    if parts:
        return "\n\n---\n\n".join(parts)

    if phase == "empty":
        return (
            "*No reasoning tags were found in this response. "
            "Restart with ``python optillm.py --gui`` (enables full-response mode) "
            "or try ``cot_reflection`` / ``re2``.*"
        )
    return "⏳ **Working…**"


def iter_reveal_stream(
    split: ReasoningSplit,
    routing_md: str,
    *,
    chunk_size: int = 48,
) -> Iterator[Tuple[str, str, str]]:
    """
    Yield Gradio updates that reveal thinking then answer in small steps.

    Used when the server returns one burst after inference (no live token stream).
    """
    thinking = split.thinking.strip()
    answer = split.answer.strip()

    if not thinking and not answer:
        thinking_md = format_thinking_markdown("", phase="empty")
        chat = format_chat_with_reasoning(split, phase="empty")
        yield chat, routing_md, thinking_md
        return

    if thinking:
        acc = ""
        for piece in chunk_text(thinking, size=chunk_size):
            acc += piece
            partial = ReasoningSplit(acc, "", in_progress=True)
            thinking_md = format_thinking_markdown(acc, phase="streaming")
            chat = format_chat_with_reasoning(partial, phase="streaming")
            yield chat, routing_md, thinking_md

    if answer:
        acc = ""
        for piece in chunk_text(answer, size=chunk_size):
            acc += piece
            partial = ReasoningSplit(thinking, acc, in_progress=False)
            thinking_md = format_thinking_markdown(
                thinking,
                phase="done" if acc == answer else "streaming",
            )
            chat = format_chat_with_reasoning(partial, phase="streaming")
            yield chat, routing_md, thinking_md

    final = ReasoningSplit(thinking, answer, in_progress=False)
    thinking_md = (
        format_thinking_markdown(thinking, phase="done")
        if thinking
        else format_thinking_markdown("", phase="empty")
    )
    chat = format_chat_with_reasoning(final, phase="done")
    yield chat, routing_md, thinking_md
