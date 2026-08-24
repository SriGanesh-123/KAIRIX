"""Investigation Agent package for evidence-driven deeper retrieval."""

from .agent import InvestigationAgent, InvestigationPlanner, InvestigationRetriever
from .config import InvestigationConfig
from .contracts import (
    ClaimVerification,
    GroundedAnswer,
    InvestigationDiagnostic,
    InvestigationErrorCode,
    InvestigationPlan,
    SufficiencyAssessment,
    VerificationResult,
)
from .formats import FormatDefinition, FormatRegistry

__all__ = [
    "InvestigationAgent",
    "InvestigationPlanner",
    "InvestigationRetriever",
    "InvestigationConfig",
    "InvestigationPlan",
    "SufficiencyAssessment",
    "GroundedAnswer",
    "VerificationResult",
    "ClaimVerification",
    "InvestigationDiagnostic",
    "InvestigationErrorCode",
    "FormatDefinition",
    "FormatRegistry",
]
