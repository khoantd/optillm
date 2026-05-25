"""Types for approach routing decisions."""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RoutingDecision:
    """Result of ML approach prediction."""

    approach: str
    confidence: float
    source: str  # "ml" | "fallback" | "heuristic"
    model_available: bool
    requested_mode: Optional[str] = None
    model_name: str = "codelion/optillm-modernbert-large"
    top_scores: List[Tuple[str, float]] = field(default_factory=list)
    override_reason: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """OpenAI-compatible extension for API responses."""
        meta = {
            "resolved_approach": self.approach,
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "model": self.model_name,
        }
        if self.requested_mode is not None:
            meta["requested_mode"] = self.requested_mode
        if self.top_scores:
            meta["top_scores"] = [
                {"approach": label, "confidence": round(score, 4)}
                for label, score in self.top_scores
            ]
        if self.override_reason:
            meta["override_reason"] = self.override_reason
        return meta

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
