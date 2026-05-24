"""Tests for remote LiteLLM proxy configuration helpers."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_litellm_proxy_module():
    path = os.path.join(ROOT, "optillm", "litellm_proxy.py")
    spec = importlib.util.spec_from_file_location("litellm_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lp = _load_litellm_proxy_module()
normalize_openai_base_url = _lp.normalize_openai_base_url
resolve_litellm_proxy_url = _lp.resolve_litellm_proxy_url
resolve_litellm_proxy_api_key = _lp.resolve_litellm_proxy_api_key
should_route_via_litellm_proxy = _lp.should_route_via_litellm_proxy
litellm_proxy_explicitly_configured = _lp.litellm_proxy_explicitly_configured

server_config = {"base_url": ""}


class TestLiteLLMProxyHelpers(unittest.TestCase):
    def setUp(self):
        self._saved_env = os.environ.copy()
        for key in (
            "LITELLM_PROXY_URL",
            "LITELLM_API_BASE",
            "LITELLM_API_KEY",
            "LITELLM_MASTER_KEY",
            "OPENAI_API_KEY",
            "CEREBRAS_API_KEY",
            "OPTILLM_BASE_URL",
        ):
            os.environ.pop(key, None)
        self._saved_base_url = server_config.get("base_url", "")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        server_config["base_url"] = self._saved_base_url

    def test_normalize_appends_v1(self):
        self.assertEqual(
            normalize_openai_base_url("http://litellm:4000"),
            "http://litellm:4000/v1",
        )
        self.assertEqual(
            normalize_openai_base_url("http://litellm:4000/v1"),
            "http://litellm:4000/v1",
        )

    def test_resolve_url_from_litellm_proxy_url(self):
        os.environ["LITELLM_PROXY_URL"] = "http://proxy.example:4000"
        self.assertEqual(
            resolve_litellm_proxy_url({}),
            "http://proxy.example:4000/v1",
        )

    def test_resolve_api_key_priority(self):
        os.environ["LITELLM_API_KEY"] = "sk-proxy"
        os.environ["OPENAI_API_KEY"] = "sk-openai"
        self.assertEqual(resolve_litellm_proxy_api_key(), "sk-proxy")

    def test_should_route_when_litellm_proxy_url_set(self):
        os.environ["LITELLM_PROXY_URL"] = "http://litellm:4000"
        self.assertTrue(should_route_via_litellm_proxy(server_config))

    def test_should_route_base_url_only_without_provider_keys(self):
        server_config["base_url"] = "http://litellm:4000"
        self.assertTrue(should_route_via_litellm_proxy(server_config))

    def test_should_not_route_when_openai_key_without_explicit_litellm_env(self):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        server_config["base_url"] = ""
        self.assertFalse(should_route_via_litellm_proxy(server_config))

    def test_explicit_litellm_env_overrides_openai_precedence_for_routing(self):
        os.environ["LITELLM_PROXY_URL"] = "http://litellm:4000"
        os.environ["OPENAI_API_KEY"] = "sk-test"
        self.assertTrue(litellm_proxy_explicitly_configured())
        self.assertTrue(should_route_via_litellm_proxy(server_config))


if __name__ == "__main__":
    unittest.main()
