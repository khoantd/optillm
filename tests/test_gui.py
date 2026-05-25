#!/usr/bin/env python3
"""Tests for Gradio GUI helpers (no Gradio runtime required)."""

import unittest

from optillm.gui import (
    _build_messages,
    approach_dropdown_choices,
    format_routing_markdown,
)


class TestGuiHelpers(unittest.TestCase):
    def test_approach_dropdown_order(self):
        choices = approach_dropdown_choices(
            ["none", "moa", "bon", "cot_reflection", "self_consistency"]
        )
        self.assertEqual(choices[:3], ["auto", "predict", "none"])
        self.assertIn("moa", choices)
        self.assertIn("self_consistency", choices)

    def test_format_routing_with_top_scores(self):
        md = format_routing_markdown(
            {
                "resolved_approach": "none",
                "confidence": 0.2327,
                "source": "fallback",
                "requested_mode": "auto",
                "top_scores": [
                    {"approach": "none", "confidence": 0.2327},
                    {"approach": "z3", "confidence": 0.1534},
                ],
            }
        )
        self.assertIn("`none`", md)
        self.assertIn("23.3%", md)
        self.assertIn("fallback", md)
        self.assertIn("Top scores", md)

    def test_format_routing_empty(self):
        md = format_routing_markdown(None)
        self.assertIn("No metadata", md)

    def test_build_messages_openai_history(self):
        msgs = _build_messages(
            "new",
            [{"role": "user", "content": "old"}],
            "Be helpful",
        )
        self.assertEqual(msgs[0], {"role": "system", "content": "Be helpful"})
        self.assertEqual(msgs[-1], {"role": "user", "content": "new"})


if __name__ == "__main__":
    unittest.main()
