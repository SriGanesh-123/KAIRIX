"""Provider-neutral LLM reviewer contract and shared structured schema."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field


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

ARTIFACT METADATA:
{json.dumps(artifact, ensure_ascii=False, indent=2)}

DETERMINISTIC ARTIFACT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

ORIGINAL SOURCE (may be truncated):
{source_text}

Return JSON with: purpose, summary, key_findings, dependencies, business_rules, semantic_gaps,
evidence_candidates, confidence (0.0-1.0), needs_deeper_analysis, and reason.
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
