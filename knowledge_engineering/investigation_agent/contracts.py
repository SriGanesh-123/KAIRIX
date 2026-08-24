"""Stage contracts and robust schema validation for the investigation agent."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class InvestigationErrorCode(str, Enum):
    """Classified internal failure codes for observability and diagnostics."""
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    GRAPH_ERROR = "GRAPH_ERROR"
    LLM_ERROR = "LLM_ERROR"
    LLM_SCHEMA_ERROR = "LLM_SCHEMA_ERROR"
    EVIDENCE_VALIDATION_ERROR = "EVIDENCE_VALIDATION_ERROR"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"


@dataclass
class InvestigationDiagnostic:
    """Structured internal diagnostic record."""
    stage: str
    code: InvestigationErrorCode
    message: str
    recoverable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["code"] = self.code.value
        return result


def calculate_grounded_confidence(
    *,
    evidence: list[dict[str, Any]],
    cited_ids: list[str],
    verified: bool,
    knowledge_gaps: list[str],
    llm_confidence: float | None = None,
) -> float:
    """Compute an explainable, evidence-grounded confidence score.

    Calculated dynamically from:
    1. Presence of verified citations from retrieved evidence
    2. Average relevance score of cited evidence items
    3. Structural graph support in cited evidence
    4. Verification agreement
    5. Deduplicated knowledge gaps penalty
    """
    if not cited_ids or not evidence:
        return 0.10 if (evidence and not cited_ids) else 0.0

    evidence_by_id = {str(item.get("source_id") or item.get("id")): item for item in evidence}
    valid_cited = [evidence_by_id[cid] for cid in cited_ids if cid in evidence_by_id]
    if not valid_cited:
        return 0.0

    # 1. Base relevance from cited items
    scores = [float(item.get("score", 0.0) or 0.0) for item in valid_cited]
    avg_score = sum(scores) / len(scores) if scores else 0.5
    # Normalize typical vector score range [0.4 - 0.9] to [0.5 - 1.0]
    base_confidence = max(0.50, min(0.95, avg_score))

    # 2. Graph grounding bonus
    has_graph = any(int(item.get("graph_evidence_count", 0) or 0) > 0 for item in valid_cited)
    if has_graph:
        base_confidence = min(0.95, base_confidence + 0.08)

    # 3. LLM estimated confidence blending (if validly provided)
    if llm_confidence is not None and 0.0 <= llm_confidence <= 1.0:
        base_confidence = 0.6 * base_confidence + 0.4 * llm_confidence

    # 4. Knowledge gap penalty (-0.05 per gap, max penalty 0.30)
    gap_penalty = min(0.30, 0.05 * len(knowledge_gaps))
    base_confidence = max(0.20, base_confidence - gap_penalty)

    # 5. Verification status
    if verified:
        base_confidence = min(0.95, base_confidence + 0.05)
    else:
        base_confidence = min(0.40, base_confidence * 0.5)

    return round(max(0.0, min(1.0, base_confidence)), 2)


@dataclass
class InvestigationPlan:
    """Explicit contract for the investigation planning stage."""
    intent: str
    objectives: list[str] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    evidence_requirements: list[str] = field(default_factory=list)
    requested_output_format: str = ""
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SufficiencyAssessment:
    """Explicit contract for the evidence sufficiency assessment stage."""
    sufficient: bool
    knowledge_gaps: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundedAnswer:
    """Explicit contract for the grounded answer generation stage."""
    answer: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    knowledge_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Explicit contract for the answer verification stage."""
    verified: bool
    answer: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    knowledge_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_payload(content: str) -> dict[str, Any]:
    """Extract a JSON object from arbitrary LLM text output.

    Handles markdown fences (```json ... ```), preamble/postscript text,
    and common unicode quotation variations.
    """
    if not isinstance(content, str):
        raise ValueError("LLM response content must be a string")

    text = content.strip()
    if not text:
        raise ValueError("LLM returned an empty response")

    # Strip markdown code blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Find the outermost balanced JSON object {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            raise ValueError(f"Extracted JSON substring is malformed: {exc}") from exc

    raise ValueError(f"No valid JSON object found in response: {text[:200]!r}")


def _normalize_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        cleaned = raw.replace("\u2011", "-").strip()
        return [cleaned] if cleaned else []
    if isinstance(raw, (list, tuple, set)):
        result: list[str] = []
        seen_keys: set[str] = set()
        for item in raw:
            s = str(item).replace("\u2011", "-").strip()
            key = " ".join(s.lower().split())
            if s and key not in seen_keys:
                seen_keys.add(key)
                result.append(s)
        return result
    return []


def _clamp_confidence(raw: Any, default: float = 0.0) -> float:
    try:
        val = float(raw)
        if val != val:  # NaN check
            return default
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return default


def parse_and_validate_plan(content: str, default_query: str) -> InvestigationPlan:
    """Parse and validate the InvestigationPlan contract."""
    data = extract_json_payload(content)
    intent = str(data.get("intent", "")).strip() or "Investigate the supplied request using available evidence."
    objectives = _normalize_string_list(data.get("objectives"))
    queries = _normalize_string_list(data.get("retrieval_queries"))
    if not queries:
        queries = [default_query] if default_query else []
    evidence_reqs = _normalize_string_list(data.get("evidence_requirements"))
    output_format = str(data.get("requested_output_format", "")).strip()
    constraints = _normalize_string_list(data.get("constraints"))

    return InvestigationPlan(
        intent=intent,
        objectives=objectives,
        retrieval_queries=queries,
        evidence_requirements=evidence_reqs,
        requested_output_format=output_format,
        constraints=constraints,
    )


def parse_and_validate_sufficiency(content: str) -> SufficiencyAssessment:
    """Parse and validate the SufficiencyAssessment contract."""
    data = extract_json_payload(content)
    if "sufficient" not in data:
        raise ValueError("Missing required boolean field 'sufficient' in sufficiency assessment")
    raw_suff = data["sufficient"]
    if isinstance(raw_suff, bool):
        sufficient = raw_suff
    elif isinstance(raw_suff, str) and raw_suff.strip().lower() in ("true", "yes", "1"):
        sufficient = True
    elif isinstance(raw_suff, str) and raw_suff.strip().lower() in ("false", "no", "0"):
        sufficient = False
    else:
        raise ValueError(f"Invalid boolean value for 'sufficient': {raw_suff!r}")

    gaps = _normalize_string_list(data.get("knowledge_gaps"))
    follow_ups = _normalize_string_list(data.get("follow_up_queries"))

    return SufficiencyAssessment(
        sufficient=sufficient,
        knowledge_gaps=gaps,
        follow_up_queries=follow_ups,
    )


def parse_and_validate_answer(content: str, allowed_evidence_ids: set[str] | None = None) -> GroundedAnswer:
    """Parse and validate the GroundedAnswer contract, filtering evidence IDs to retrieved evidence."""
    data = extract_json_payload(content)
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer generation must contain a non-empty 'answer' string")

    raw_ids = _normalize_string_list(data.get("evidence_ids"))
    if allowed_evidence_ids is not None:
        valid_ids = [eid for eid in raw_ids if eid in allowed_evidence_ids]
    else:
        valid_ids = raw_ids

    confidence = _clamp_confidence(data.get("confidence"), default=0.0)
    gaps = _normalize_string_list(data.get("knowledge_gaps"))

    return GroundedAnswer(
        answer=answer.strip(),
        evidence_ids=list(dict.fromkeys(valid_ids)),
        confidence=confidence,
        knowledge_gaps=gaps,
    )


def parse_and_validate_verification(
    content: str,
    draft: GroundedAnswer,
    allowed_evidence_ids: set[str] | None = None,
) -> VerificationResult:
    """Parse and validate the VerificationResult contract."""
    data = extract_json_payload(content)
    raw_verified = data.get("verified")
    if isinstance(raw_verified, bool):
        verified = raw_verified
    elif isinstance(raw_verified, str) and raw_verified.strip().lower() in ("true", "yes", "1"):
        verified = True
    elif isinstance(raw_verified, str) and raw_verified.strip().lower() in ("false", "no", "0"):
        verified = False
    else:
        # Default to False if verifier does not clearly state True
        verified = False

    raw_answer = data.get("answer")
    answer = str(raw_answer).strip() if isinstance(raw_answer, str) and raw_answer.strip() else draft.answer

    raw_ids = _normalize_string_list(data.get("evidence_ids", draft.evidence_ids))
    if allowed_evidence_ids is not None:
        valid_ids = [eid for eid in raw_ids if eid in allowed_evidence_ids]
    else:
        valid_ids = raw_ids

    confidence = _clamp_confidence(data.get("confidence"), default=draft.confidence)
    gaps = _normalize_string_list(data.get("knowledge_gaps", draft.knowledge_gaps))

    return VerificationResult(
        verified=verified,
        answer=answer,
        evidence_ids=list(dict.fromkeys(valid_ids)),
        confidence=confidence,
        knowledge_gaps=gaps,
    )
