#!/usr/bin/env python3
"""Tests for Gradio GUI helpers (no Gradio runtime required)."""

import unittest
from pathlib import Path

from optillm.gui import (
    _apply_gui_sample_plain,
    _build_messages,
    approach_dropdown_choices,
    build_gui_samples,
    format_routing_markdown,
    format_testing_guide_ui_markdown,
    load_test_cases,
    sample_dropdown_choices,
    samples_by_id,
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
        self.assertIn("Load a sample", md)

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

    def test_build_gui_samples_from_repo_json(self):
        path = Path(__file__).resolve().parent / "test_cases.json"
        cases = load_test_cases(path)
        samples = build_gui_samples(cases, known_approaches=["moa", "bon", "z3", "re2"])
        catalog = samples_by_id(samples)
        self.assertIn("moa::Arena Bench Hard", catalog)
        self.assertIn("re2::strawberry", catalog)
        moa = catalog["moa::Arena Bench Hard"]
        self.assertIn("numpy", moa["query"].lower())

    def test_apply_gui_sample(self):
        path = Path(__file__).resolve().parent / "test_cases.json"
        catalog = samples_by_id(build_gui_samples(load_test_cases(path)))
        slug, sp, query, hint = _apply_gui_sample_plain("bon::GSM8K", catalog)
        self.assertEqual(slug, "bon")
        self.assertIn("parking lot", query.lower())
        self.assertIn("Loaded", hint)

    def test_sample_dropdown_filter_by_approach(self):
        path = Path(__file__).resolve().parent / "test_cases.json"
        samples = build_gui_samples(load_test_cases(path))
        all_ids = {c[1] for c in sample_dropdown_choices(samples)}
        moa_ids = {c[1] for c in sample_dropdown_choices(samples, "moa") if c[1]}
        self.assertTrue(len(moa_ids) < len(all_ids))
        self.assertTrue(all(i.startswith("moa::") for i in moa_ids))

    def test_testing_guide_ui_mentions_manual_vs_auto(self):
        md = format_testing_guide_ui_markdown()
        self.assertIn("auto", md.lower())
        self.assertIn("moa", md)


if __name__ == "__main__":
    unittest.main()
