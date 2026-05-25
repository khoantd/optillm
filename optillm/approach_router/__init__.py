"""Predictive routing for optillm_approach selection."""

from .predictor import ApproachPredictor, ROUTING_MODES, DEFAULT_ROUTER_CONFIG
from .types import RoutingDecision
from .ml_router import ML_APPROACHES, load_optillm_model, reset_model_cache_for_testing
from .resolver import resolve_optillm_approach, attach_optillm_routing

__all__ = [
    "ApproachPredictor",
    "RoutingDecision",
    "ROUTING_MODES",
    "DEFAULT_ROUTER_CONFIG",
    "ML_APPROACHES",
    "load_optillm_model",
    "reset_model_cache_for_testing",
    "resolve_optillm_approach",
    "attach_optillm_routing",
]
