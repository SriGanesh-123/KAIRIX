"""Gemini-backed artifact reviewer for the Knowledge Engineering Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class ArtifactReview(BaseModel):
    """Structured semantic review returned by Gemini."""

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


class GeminiArtifactReviewer:
    """Review canonical artifacts with Gemini structured output."""

    # Stable Gemini 3.5 Flash-Lite model. It is intended for high-throughput,
    # low-cost structured extraction and subagent workloads.
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiArtifactReviewer")
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def review(self, artifact: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        source_text = self._load_source(artifact)
        prompt = self._build_prompt(artifact, profile, source_text)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ArtifactReview,
            ),
        )

        if not response.text:
            raise RuntimeError(f"Gemini returned an empty response for {artifact.get('file_name')}")

        result = ArtifactReview.model_validate_json(response.text)
        return result.model_dump()

    @staticmethod
    def _load_source(artifact: Dict[str, Any]) -> str:
        """Load the original artifact when its canonical path is available."""
        path_value = artifact.get("path")
        if not path_value:
            return "Original source path is not available in canonical metadata."

        path = Path(path_value)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            path = project_root / path

        if not path.exists() or not path.is_file():
            return f"Original source could not be loaded from: {path_value}"

        # Keep prompts bounded while retaining enough source for semantic review.
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[:40000]

    @staticmethod
    def _build_prompt(
        artifact: Dict[str, Any], profile: Dict[str, Any], source_text: str
    ) -> str:
        return f"""You are the Artifact Review component of a legacy reverse-engineering Knowledge Engineering Agent.

Review the supplied legacy artifact using ONLY the supplied artifact metadata, deterministic profile, and source text.
Do not invent dependencies, business rules, or evidence. If something cannot be established, put it in semantic_gaps.
Distinguish extracted facts from reasonable semantic interpretation. Keep confidence conservative.

Your job is to produce a structured review that will be reconciled with deterministic parser output later.

ARTIFACT METADATA:
{json.dumps(artifact, ensure_ascii=False, indent=2)}

DETERMINISTIC ARTIFACT PROFILE:
{json.dumps(profile, ensure_ascii=False, indent=2)}

ORIGINAL SOURCE (may be truncated):
{source_text}

Return:
- purpose
- concise summary
- key findings
- dependencies
- business rules
- semantic gaps
- candidate evidence locations/text descriptions
- confidence from 0.0 to 1.0
- whether deeper analysis is required
- reason when deeper analysis is required
"""
