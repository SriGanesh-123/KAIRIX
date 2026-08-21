"""Provider-neutral LLM reviewer contract and shared structured schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator


class CodeFixFinding(BaseModel):
    """A potential source-code fix that must be handled by a developer."""

    code_fix_required: bool = False
    source_file: str = ""
    location: str = ""
    issue: str = ""
    evidence: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    action_owner: str = "DEVELOPER"
    source_modified: bool = False

    @field_validator("action_owner", mode="before")
    @classmethod
    def normalize_owner(cls, value: Any) -> str:
        return str(value or "DEVELOPER").upper()


class ArtifactReview(BaseModel):
    purpose: str = Field(description="What the artifact does in business/technical terms.")
    summary: str = Field(description="Concise explanation of the artifact's behavior.")
    key_findings: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    business_rules: list[str] = Field(default_factory=list)
    semantic_gaps: list[str] = Field(default_factory=list)
    evidence_candidates: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_deeper_analysis: bool = False
    reason: str = ""
    code_fix: CodeFixFinding = Field(default_factory=CodeFixFinding)

    @field_validator(
        "key_findings",
        "dependencies",
        "business_rules",
        "semantic_gaps",
        "evidence_candidates",
        mode="before",
    )
    @classmethod
    def normalize_review_items(cls, value: Any) -> list[str]:
        """Accept strings or structured objects returned by different LLMs.

        The canonical enrichment schema remains list[str], but providers may
        return richer JSON objects. Convert those objects to stable JSON text
        instead of rejecting an otherwise valid review.
        """
        if value is None:
            return []
        if isinstance(value, (str, int, float, bool)):
            return [str(value)]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return [str(value)]

        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                normalized.append(str(item))
        return normalized


class LLMReviewer(Protocol):
    provider: str
    model: str

    def review(self, artifact: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        ...


def build_prompt(artifact: dict[str, Any], profile: dict[str, Any], source_text: str) -> str:
    return f"""You are the Artifact Review component of a legacy reverse-engineering Knowledge Engineering Agent.

Review the supplied legacy artifact using ONLY the supplied artifact metadata, deterministic profile, and source text.
Do not invent dependencies, business rules, or evidence. If something cannot be established, put it in semantic_gaps.
Distinguish extracted facts from reasonable semantic interpretation. Keep confidence conservative.

SOURCE INTEGRITY RULE:
You are READ-ONLY. Never modify, rewrite, or claim to have modified the original source artifact.
If you identify a possible code defect or code change opportunity, report it only as a developer-owned code_fix finding.
Set source_modified to false. The developer is responsible for deciding and implementing any source-code change.
If no code fix is supported by evidence, set code_fix_required to false and leave the other code_fix fields empty/default.

ARTIFACT METADATA:
{json.dumps(artifact, ensure_ascii=False, indent=2)}

DETERMINISTIC ARTIFACT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

ORIGINAL SOURCE (may be truncated):
{source_text}

Return JSON with: purpose, summary, key_findings, dependencies, business_rules, semantic_gaps,
evidence_candidates, confidence (0.0-1.0), needs_deeper_analysis, reason, and code_fix.
The code_fix object must contain: code_fix_required, source_file, location, issue, evidence,
recommended_action, action_owner, source_modified.
"""


def load_source(artifact: dict[str, Any]) -> str:
    path_value = artifact.get("path")
    if not path_value:
        return "Original source path is not available in canonical metadata."
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.exists() or not path.is_file():
        return f"Original source could not be loaded from: {path_value}"
    return path.read_text(encoding="utf-8", errors="replace")[:40000]
