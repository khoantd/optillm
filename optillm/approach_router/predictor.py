"""High-level approach predictor (ML-first with configurable fallback)."""

import logging
from typing import Any, Dict, Optional

from .ml_router import (
    OPTILLM_MODEL_NAME,
    load_optillm_model,
    predict_approach,
    preprocess_input,
)
from .types import RoutingDecision

logger = logging.getLogger(__name__)

ROUTING_MODES = frozenset({"auto", "predict"})
DEFAULT_ROUTER_CONFIG: Dict[str, Any] = {
    "effort": 0.7,
    "min_confidence": 0.0,
    "fallback": "none",
}


def _normalize_router_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_ROUTER_CONFIG)
    if config:
        merged.update({k: v for k, v in config.items() if v is not None})
    return merged


class ApproachPredictor:
    """ML-first predictor for optillm_approach selection."""

    def predict(
        self,
        system_prompt: str,
        initial_query: str,
        *,
        requested_mode: Optional[str] = None,
        router_config: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        cfg = _normalize_router_config(router_config)
        effort = float(cfg.get("effort", 0.7))
        min_confidence = float(cfg.get("min_confidence", 0.0))
        fallback = str(cfg.get("fallback", "none"))

        try:
            model, tokenizer, device = load_optillm_model()
            input_ids, attention_mask = preprocess_input(
                tokenizer, system_prompt or "", initial_query or ""
            )
            approach, confidence = predict_approach(
                model, input_ids, attention_mask, device, effort=effort
            )
            if confidence < min_confidence:
                logger.info(
                    "Approach router confidence %.3f below min %.3f, using fallback %s",
                    confidence,
                    min_confidence,
                    fallback,
                )
                return RoutingDecision(
                    approach=fallback,
                    confidence=confidence,
                    source="fallback",
                    model_available=True,
                    requested_mode=requested_mode,
                    model_name=OPTILLM_MODEL_NAME,
                )
            logger.info(
                "Approach router predicted %s (confidence=%.3f, mode=%s)",
                approach,
                confidence,
                requested_mode or "unknown",
            )
            return RoutingDecision(
                approach=approach,
                confidence=confidence,
                source="ml",
                model_available=True,
                requested_mode=requested_mode,
                model_name=OPTILLM_MODEL_NAME,
            )
        except Exception as exc:
            logger.error("Approach router prediction failed: %s", exc)
            return RoutingDecision(
                approach=fallback,
                confidence=0.0,
                source="fallback",
                model_available=False,
                requested_mode=requested_mode,
                model_name=OPTILLM_MODEL_NAME,
            )
