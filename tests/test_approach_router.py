#!/usr/bin/env python3
"""Tests for predictive optillm_approach auto-routing."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.approach_router import (
    ApproachPredictor,
    RoutingDecision,
    ROUTING_MODES,
    attach_optillm_routing,
)
from optillm.approach_router.resolver import (
    model_has_approach_prefix,
    resolve_optillm_approach,
)
from optillm.approach_router.ml_router import reset_model_cache_for_testing

# Mirror server known_approaches for prefix parsing (no heavy server import)
KNOWN_APPROACHES = [
    "none", "mcts", "bon", "moa", "rto", "z3", "self_consistency",
    "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2", "cepo", "mars",
]
PLUGIN_APPROACHES = {}


def _parse_combined_approach(model, known_approaches, plugin_approaches):
    """Minimal copy of server.parse_combined_approach for isolated tests."""
    if model == "auto":
        return "SINGLE", ["none"], model

    parts = model.split("-")
    approaches = []
    operation = "SINGLE"
    model_parts = []
    parsing_approaches = True

    for part in parts:
        if parsing_approaches:
            if part in known_approaches or part in plugin_approaches:
                approaches.append(part)
            elif "&" in part:
                operation = "AND"
                approaches.extend(part.split("&"))
            elif "|" in part:
                operation = "OR"
                approaches.extend(part.split("|"))
            else:
                parsing_approaches = False
                model_parts.append(part)
        else:
            model_parts.append(part)

    if not approaches:
        approaches = ["none"]
        operation = "SINGLE"

    actual_model = "-".join(model_parts)
    return operation, approaches, actual_model


class TestAttachOptillmRouting(unittest.TestCase):
    def test_attaches_metadata(self):
        resp = {"model": "gpt-4", "choices": []}
        meta = {"resolved_approach": "moa", "confidence": 0.9, "source": "ml"}
        out = attach_optillm_routing(resp, meta)
        self.assertEqual(out["optillm_routing"], meta)
        self.assertNotIn("optillm_routing", resp)

    def test_skips_when_none(self):
        resp = {"model": "gpt-4"}
        self.assertIs(attach_optillm_routing(resp, None), resp)


class TestModelHasApproachPrefix(unittest.TestCase):
    def test_detects_prefix(self):
        self.assertTrue(
            model_has_approach_prefix(
                "moa-gpt-4o-mini",
                KNOWN_APPROACHES,
                PLUGIN_APPROACHES,
                _parse_combined_approach,
            )
        )

    def test_plain_model(self):
        self.assertFalse(
            model_has_approach_prefix(
                "gpt-4o-mini",
                KNOWN_APPROACHES,
                PLUGIN_APPROACHES,
                _parse_combined_approach,
            )
        )


class TestResolveOptillmApproach(unittest.TestCase):
    def _mock_predictor(self, approach="moa", confidence=0.85):
        predictor = MagicMock(spec=ApproachPredictor)
        predictor.predict.return_value = RoutingDecision(
            approach=approach,
            confidence=confidence,
            source="ml",
            model_available=True,
            requested_mode="auto",
        )
        return predictor

    def test_auto_predicts_when_no_override(self):
        predictor = self._mock_predictor()
        resolved, model, meta = resolve_optillm_approach(
            optillm_approach="auto",
            raw_model="gpt-4o-mini",
            system_prompt="You are helpful.",
            initial_query="Solve 2+2",
            message_optillm_approach=None,
            request_has_explicit_approach=False,
            request_config={},
            auto_router_enabled=True,
            default_router_effort=0.7,
            known_approaches=KNOWN_APPROACHES,
            plugin_approaches=PLUGIN_APPROACHES,
            parse_combined_approach=_parse_combined_approach,
            predictor=predictor,
        )
        self.assertEqual(resolved, "moa")
        self.assertEqual(model, "gpt-4o-mini")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["resolved_approach"], "moa")
        predictor.predict.assert_called_once()

    def test_auto_skips_when_model_prefix(self):
        predictor = self._mock_predictor()
        resolved, model, meta = resolve_optillm_approach(
            optillm_approach="auto",
            raw_model="bon-gpt-4o-mini",
            system_prompt="",
            initial_query="hi",
            message_optillm_approach=None,
            request_has_explicit_approach=False,
            request_config={},
            auto_router_enabled=True,
            default_router_effort=0.7,
            known_approaches=KNOWN_APPROACHES,
            plugin_approaches=PLUGIN_APPROACHES,
            parse_combined_approach=_parse_combined_approach,
            predictor=predictor,
        )
        self.assertEqual(resolved, "auto")
        self.assertEqual(model, "bon-gpt-4o-mini")
        self.assertIsNone(meta)
        predictor.predict.assert_not_called()

    def test_predict_forces_ml_even_with_prefix(self):
        predictor = self._mock_predictor(approach="bon")
        resolved, model, meta = resolve_optillm_approach(
            optillm_approach="predict",
            raw_model="moa-gpt-4o-mini",
            system_prompt="",
            initial_query="hi",
            message_optillm_approach=None,
            request_has_explicit_approach=True,
            request_config={},
            auto_router_enabled=True,
            default_router_effort=0.7,
            known_approaches=KNOWN_APPROACHES,
            plugin_approaches=PLUGIN_APPROACHES,
            parse_combined_approach=_parse_combined_approach,
            predictor=predictor,
        )
        self.assertEqual(resolved, "bon")
        self.assertEqual(model, "gpt-4o-mini")
        predictor.predict.assert_called_once()

    def test_explicit_extra_body_override(self):
        predictor = self._mock_predictor()
        resolved, model, meta = resolve_optillm_approach(
            optillm_approach="re2",
            raw_model="gpt-4o-mini",
            system_prompt="",
            initial_query="hi",
            message_optillm_approach=None,
            request_has_explicit_approach=True,
            request_config={},
            auto_router_enabled=True,
            default_router_effort=0.7,
            known_approaches=KNOWN_APPROACHES,
            plugin_approaches=PLUGIN_APPROACHES,
            parse_combined_approach=_parse_combined_approach,
            predictor=predictor,
        )
        self.assertEqual(resolved, "re2")
        self.assertIsNone(meta)
        predictor.predict.assert_not_called()

    def test_message_tag_override(self):
        predictor = self._mock_predictor()
        resolved, model, meta = resolve_optillm_approach(
            optillm_approach="mcts",
            raw_model="gpt-4o-mini",
            system_prompt="",
            initial_query="hi",
            message_optillm_approach="mcts",
            request_has_explicit_approach=False,
            request_config={},
            auto_router_enabled=True,
            default_router_effort=0.7,
            known_approaches=KNOWN_APPROACHES,
            plugin_approaches=PLUGIN_APPROACHES,
            parse_combined_approach=_parse_combined_approach,
            predictor=predictor,
        )
        self.assertEqual(resolved, "mcts")
        self.assertIsNone(meta)
        predictor.predict.assert_not_called()

    def test_auto_disabled_falls_back_to_none(self):
        predictor = self._mock_predictor()
        resolved, model, meta = resolve_optillm_approach(
            optillm_approach="auto",
            raw_model="gpt-4o-mini",
            system_prompt="",
            initial_query="hi",
            message_optillm_approach=None,
            request_has_explicit_approach=False,
            request_config={},
            auto_router_enabled=False,
            default_router_effort=0.7,
            known_approaches=KNOWN_APPROACHES,
            plugin_approaches=PLUGIN_APPROACHES,
            parse_combined_approach=_parse_combined_approach,
            predictor=predictor,
        )
        self.assertEqual(resolved, "none")
        self.assertIsNone(meta)
        predictor.predict.assert_not_called()


class TestApproachPredictorFallback(unittest.TestCase):
    @patch("optillm.approach_router.predictor.load_optillm_model")
    def test_fallback_on_load_failure(self, mock_load):
        mock_load.side_effect = RuntimeError("model unavailable")
        decision = ApproachPredictor().predict("sys", "query", requested_mode="auto")
        self.assertEqual(decision.approach, "none")
        self.assertEqual(decision.source, "fallback")
        self.assertFalse(decision.model_available)


class TestMlRouterSingleton(unittest.TestCase):
    @patch("transformers.AutoTokenizer")
    @patch("transformers.AutoModel")
    @patch("huggingface_hub.hf_hub_download")
    @patch("safetensors.torch.load_model")
    def test_load_called_once(self, mock_load_model, mock_hf, mock_auto_model, mock_tokenizer):
        reset_model_cache_for_testing()
        mock_hf.return_value = "/fake/model.safetensors"
        mock_base = MagicMock()
        mock_base.config.hidden_size = 64
        mock_auto_model.from_pretrained.return_value = mock_base
        mock_tok = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        from optillm.approach_router.ml_router import load_optillm_model

        load_optillm_model()
        load_optillm_model()
        mock_auto_model.from_pretrained.assert_called_once()
        reset_model_cache_for_testing()


class TestRoutingModes(unittest.TestCase):
    def test_modes_defined(self):
        self.assertIn("auto", ROUTING_MODES)
        self.assertIn("predict", ROUTING_MODES)


if __name__ == "__main__":
    unittest.main()
