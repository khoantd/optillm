#!/usr/bin/env python3
"""Tests for reasoning trace parsing used by the GUI stream."""

import unittest

from optillm.reasoning_trace import (
    format_chat_with_reasoning,
    format_thinking_markdown,
    split_thinking_and_answer,
)

_REDACTED = "redacted" + "_thinking"


class TestReasoningTrace(unittest.TestCase):
    def test_split_redacted_thinking(self):
        text = (
            f"<{_REDACTED}>Step one\nStep two</{_REDACTED}>\n"
            "Final answer here."
        )
        split = split_thinking_and_answer(text)
        self.assertIn("Step one", split.thinking)
        self.assertEqual(split.answer, "Final answer here.")
        self.assertFalse(split.in_progress)

    def test_split_cot_output(self):
        text = (
            "<thinking>internal</thinking>\n"
            "<output>42</output>"
        )
        split = split_thinking_and_answer(text)
        self.assertIn("internal", split.thinking)
        self.assertEqual(split.answer, "42")

    def test_open_thinking_tag_in_progress(self):
        text = f"<{_REDACTED}>still working"
        split = split_thinking_and_answer(text)
        self.assertIn("still working", split.thinking)
        self.assertTrue(split.in_progress)

    def test_format_streaming_phase(self):
        md = format_thinking_markdown("abc", phase="streaming")
        self.assertIn("In progress", md)
        self.assertIn("abc", md)

    def test_format_empty_phase(self):
        md = format_thinking_markdown("", phase="empty")
        self.assertIn("No separate reasoning", md)

    def test_format_chat_shows_reasoning_and_answer(self):
        split = split_thinking_and_answer(
            f"<{'redacted' + '_thinking'}>step 1</{'redacted' + '_thinking'}>\n"
            "<output>42</output>"
        )
        chat = format_chat_with_reasoning(split, phase="done")
        self.assertIn("Reasoning", chat)
        self.assertIn("step 1", chat)
        self.assertIn("42", chat)


if __name__ == "__main__":
    unittest.main()
