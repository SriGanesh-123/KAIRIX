"""Central configuration for the Investigation Agent."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvestigationConfig:
    """Configurable boundaries and thresholds for multi-stage investigations."""

    max_investigation_rounds: int = 2
    max_follow_up_queries: int = 5
    max_repair_retries: int = 1
    max_llm_calls: int = 12
    max_graph_hops: int = 3
    initial_graph_hops: int = 1
    retrieval_limit: int = 5
    max_compact_evidence_items: int = 15
    confidence_thresholds: tuple[float, float] = (0.45, 0.75)
    provider_max_retries: int = 3
    provider_base_delay: float = 1.0
    provider_max_delay: float = 10.0
    provider_timeout: float = 30.0

    def __post_init__(self) -> None:
        low, high = self.confidence_thresholds
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("confidence thresholds must satisfy 0 <= low < high <= 1")
        if self.max_investigation_rounds < 1:
            raise ValueError("max_investigation_rounds must be at least 1")
        if self.max_follow_up_queries < 1:
            raise ValueError("max_follow_up_queries must be at least 1")
        if not 1 <= self.initial_graph_hops <= self.max_graph_hops <= 4:
            raise ValueError("graph hop range must satisfy 1 <= initial <= max <= 4")
