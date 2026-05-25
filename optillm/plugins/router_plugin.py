"""Router plugin: ML-based approach selection via shared approach_router."""

import logging

from optillm.approach_router import ApproachPredictor

SLUG = "router"
logger = logging.getLogger(__name__)


def run(system_prompt, initial_query, client, model, **kwargs):
    request_config = kwargs.get("request_config") or {}
    request_id = kwargs.get("request_id")
    router_config = request_config.get("optillm_auto_router")
    if isinstance(router_config, dict):
        pass
    else:
        router_config = None

    try:
        from optillm.server import execute_single_approach, server_config

        predictor = ApproachPredictor()
        decision = predictor.predict(
            system_prompt,
            initial_query,
            requested_mode="router",
            router_config=router_config,
        )
        effort = (
            router_config.get("effort", server_config.get("auto_router_effort", 0.7))
            if router_config
            else server_config.get("auto_router_effort", 0.7)
        )
        logger.info(
            "Router plugin predicted %s (confidence=%.3f, effort=%.2f)",
            decision.approach,
            decision.confidence,
            effort,
        )
        return execute_single_approach(
            decision.approach,
            system_prompt,
            initial_query,
            client,
            model,
            request_config,
            request_id,
        )
    except Exception as exc:
        logger.error("Router plugin failed: %s. Falling back to none.", exc)
        from optillm.server import execute_single_approach

        return execute_single_approach(
            "none",
            system_prompt,
            initial_query,
            client,
            model,
            request_config,
            request_id,
        )
