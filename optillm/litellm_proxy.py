"""
Helpers for routing OptiLLM to a remote LiteLLM proxy (OpenAI-compatible API).

LiteLLM proxy: https://docs.litellm.ai/docs/proxy/quick_start
Typical base URL: http://localhost:4000/v1
"""

import os
from typing import Any, Dict, Optional


def normalize_openai_base_url(url: str) -> str:
    """Ensure URL ends with /v1 for OpenAI SDK clients."""
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def resolve_litellm_proxy_url(server_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve upstream LiteLLM (or any OpenAI-compatible) proxy URL.

    Priority: LITELLM_PROXY_URL > LITELLM_API_BASE > server base_url > OPTILLM_BASE_URL
    """
    server_config = server_config or {}

    for env_key in ("LITELLM_PROXY_URL", "LITELLM_API_BASE"):
        env_url = os.environ.get(env_key, "").strip()
        if env_url:
            return normalize_openai_base_url(env_url)

    base_url = (server_config.get("base_url") or os.environ.get("OPTILLM_BASE_URL") or "").strip()
    if base_url:
        return normalize_openai_base_url(base_url)
    return ""


def litellm_proxy_explicitly_configured() -> bool:
    """True when dedicated LiteLLM proxy env vars are set."""
    return bool(
        os.environ.get("LITELLM_PROXY_URL", "").strip()
        or os.environ.get("LITELLM_API_BASE", "").strip()
    )


def resolve_litellm_proxy_api_key() -> str:
    """
    API key sent to the LiteLLM proxy (master key or virtual key).

    LiteLLM accepts LITELLM_MASTER_KEY; many deployments also use OPENAI_API_KEY.
    """
    return (
        os.environ.get("LITELLM_API_KEY")
        or os.environ.get("LITELLM_MASTER_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "litellm"
    )


def should_route_via_litellm_proxy(server_config: Dict[str, Any]) -> bool:
    """
    Whether get_config() should use an OpenAI client pointed at the proxy URL.

    - Always when LITELLM_PROXY_URL / LITELLM_API_BASE is set (unless local OPTILLM inference).
    - Also when only OPTILLM_BASE_URL is set and no direct provider API keys are present.
    """
    if not resolve_litellm_proxy_url(server_config):
        return False
    if litellm_proxy_explicitly_configured():
        return True
    return not any(
        os.environ.get(k)
        for k in ("CEREBRAS_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
    )
