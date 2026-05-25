"""High-level approach predictor (ML-first with configurable fallback)."""

import logging
from typing import Any, Dict, Optional

from .ml_router import (
    OPTILLM_MODEL_NAME,
    format_router_input,
    load_optillm_model,
    predict_approach_topk,
    preprocess_input,
)
from .heuristics import apply_z3_algorithm_override
from .types import RoutingDecision

logger = logging.getLogger(__name__)

ROUTING_MODES = frozenset({"auto", "predict"})
DEFAULT_ROUTER_CONFIG: Dict[str, Any] = {
    "effort": 0.7,
    "min_confidence": 0.30,
    "fallback": "none",
    "top_k": 5,
}


def _router_input_stats(system_prompt: str, initial_query: str) -> Dict[str, int]:
    iq = initial_query or ""
    return {
        "system_chars": len(system_prompt or ""),
        "query_chars": len(iq),
        "user_turns": iq.count("User:"),
        "assistant_turns": iq.count("Assistant:"),
        "router_chars": len(format_router_input(system_prompt, initial_query)),
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
        top_k = max(1, int(cfg.get("top_k", 3)))

        stats = _router_input_stats(system_prompt or "", initial_query or "")
        logger.debug("Approach router input stats: %s", stats)

        try:
            model, tokenizer, device = load_optillm_model()
            input_ids, attention_mask = preprocess_input(
                tokenizer, system_prompt or "", initial_query or ""
            )
            approach, confidence, top_scores = predict_approach_topk(
                model, input_ids, attention_mask, device, effort=effort, top_k=top_k
            )
            top_summary = ", ".join(
                f"{label}={score:.3f}" for label, score in top_scores
            )
            source = "ml"
            override_reason = None
            approach, confidence, source, override_reason = apply_z3_algorithm_override(
                approach,
                confidence,
                top_scores,
                system_prompt or "",
                initial_query or "",
            )
            if override_reason:
                logger.info(
                    "Approach router heuristic override: z3 -> %s (%s); top=%s",
                    approach,
                    override_reason,
                    top_summary,
                )
                return RoutingDecision(
                    approach=approach,
                    confidence=confidence,
                    source=source,
                    model_available=True,
                    requested_mode=requested_mode,
                    model_name=OPTILLM_MODEL_NAME,
                    top_scores=top_scores,
                    override_reason=override_reason,
                )
            if confidence < min_confidence:
                logger.info(
                    "Approach router confidence %.3f below min %.3f, using fallback %s; top=%s",
                    confidence,
                    min_confidence,
                    fallback,
                    top_summary,
                )
                return RoutingDecision(
                    approach=fallback,
                    confidence=confidence,
                    source="fallback",
                    model_available=True,
                    requested_mode=requested_mode,
                    model_name=OPTILLM_MODEL_NAME,
                    top_scores=top_scores,
                )
            logger.info(
                "Approach router predicted %s (confidence=%.3f, mode=%s); top=%s; input=%s",
                approach,
                confidence,
                requested_mode or "unknown",
                top_summary,
                stats,
            )
            return RoutingDecision(
                approach=approach,
                confidence=confidence,
                source=source,
                model_available=True,
                requested_mode=requested_mode,
                model_name=OPTILLM_MODEL_NAME,
                top_scores=top_scores,
                override_reason=override_reason,
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
