"""Types for approach routing decisions."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class RoutingDecision:
    """Result of ML approach prediction."""

    approach: str
    confidence: float
    source: str  # "ml" | "fallback"
    model_available: bool
    requested_mode: Optional[str] = None
    model_name: str = "codelion/optillm-modernbert-large"

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
        return meta

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
