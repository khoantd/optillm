#!/usr/bin/env python3
"""Tests for OptiLLM proxy request key handling."""

import unittest

from optillm.server import (
    normalize_request_data,
    strip_optillm_proxy_keys,
    OPTILLM_PROXY_ONLY_KEYS,
)


class TestServerRequestConfig(unittest.TestCase):
    def test_normalize_request_data_flattens_extra_body(self):
        data = normalize_request_data(
            {
                "model": "gpt-4o-mini",
                "messages": [],
                "extra_body": {"optillm_approach": "auto", "temperature": 0.2},
            }
        )
        self.assertEqual(data["optillm_approach"], "auto")
        self.assertEqual(data["temperature"], 0.2)
        self.assertNotIn("extra_body", data)

    def test_strip_optillm_proxy_keys(self):
        config = {
            "temperature": 0.5,
            "optillm_approach": "none",
            "mcts_depth": 3,
        }
        strip_optillm_proxy_keys(config)
        self.assertEqual(config, {"temperature": 0.5})

    def test_optillm_approach_not_in_provider_keys(self):
        self.assertIn("optillm_approach", OPTILLM_PROXY_ONLY_KEYS)


if __name__ == "__main__":
    unittest.main()
