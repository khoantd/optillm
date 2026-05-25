"""Resolve optillm_approach with override priority and ML prediction."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .predictor import ApproachPredictor, ROUTING_MODES
from .types import RoutingDecision


def attach_optillm_routing(response_data: dict, routing_meta: Optional[dict]) -> dict:
    """Add optillm_routing extension to an OpenAI-style completion response."""
    if routing_meta and isinstance(response_data, dict):
        response_data = dict(response_data)
        response_data["optillm_routing"] = routing_meta
    return response_data


def model_has_approach_prefix(
    model_name: str,
    known_approaches: List[str],
    plugin_approaches: Dict[str, Any],
    parse_combined_approach: Callable,
) -> bool:
    """True if the model string already includes an approach prefix."""
    operation, approaches, _ = parse_combined_approach(
        model_name, known_approaches, plugin_approaches
    )
    if operation != "SINGLE":
        return True
    return approaches != ["none"]


def strip_approach_prefix(
    model_name: str,
    known_approaches: List[str],
    plugin_approaches: Dict[str, Any],
    parse_combined_approach: Callable,
) -> str:
    """Return base model name without approach prefix."""
    _, _, actual_model = parse_combined_approach(
        model_name, known_approaches, plugin_approaches
    )
    return actual_model or model_name


def resolve_optillm_approach(
    *,
    optillm_approach: str,
    raw_model: str,
    system_prompt: str,
    initial_query: str,
    message_optillm_approach: Optional[str],
    request_has_explicit_approach: bool,
    request_config: Optional[Dict[str, Any]],
    auto_router_enabled: bool,
    default_router_effort: float,
    known_approaches: List[str],
    plugin_approaches: Dict[str, Any],
    parse_combined_approach: Callable,
    predictor: Optional[ApproachPredictor] = None,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Resolve the effective approach and model for routing.

    Returns:
        (resolved_approach, model_for_routing, optillm_routing_metadata or None)
    """
    routing_meta: Optional[Dict[str, Any]] = None
    model = raw_model

    # Priority 1 & 2: prompt tag (already merged into optillm_approach by caller) or model prefix
    if message_optillm_approach:
        return optillm_approach, model, None

    has_model_prefix = model_has_approach_prefix(
        raw_model, known_approaches, plugin_approaches, parse_combined_approach
    )

    # Priority 3: explicit extra_body approach (not auto/predict)
    has_explicit_extra = (
        request_has_explicit_approach
        and optillm_approach not in ROUTING_MODES
    )

    if has_model_prefix and optillm_approach != "predict":
        return optillm_approach, model, None

    if has_explicit_extra:
        return optillm_approach, model, None

    # Priority 4: ML predictor
    should_predict = False
    if optillm_approach == "predict":
        should_predict = True
        if has_model_prefix:
            model = strip_approach_prefix(
                raw_model, known_approaches, plugin_approaches, parse_combined_approach
            )
    elif optillm_approach == "auto" and auto_router_enabled:
        should_predict = True

    if not should_predict:
        if optillm_approach in ROUTING_MODES:
            return "none", model, None
        return optillm_approach, model, None

    router_config = {}
    if request_config:
        raw_cfg = request_config.get("optillm_auto_router")
        if isinstance(raw_cfg, dict):
            router_config = raw_cfg
        elif request_config.get("optillm_auto_router_effort") is not None:
            router_config = {"effort": request_config["optillm_auto_router_effort"]}
    if "effort" not in router_config:
        router_config["effort"] = default_router_effort

    pred = predictor or ApproachPredictor()
    decision: RoutingDecision = pred.predict(
        system_prompt,
        initial_query,
        requested_mode=optillm_approach,
        router_config=router_config,
    )
    routing_meta = decision.to_metadata()
    return decision.approach, model, routing_meta
