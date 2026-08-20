"""Gemini-backed artifact reviewer for the Knowledge Engineering Agent."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field
from google import genai
from google.genai import errors, types


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

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for GeminiArtifactReviewer")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.client = genai.Client(api_key=api_key)

    def review(self, artifact: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        source_text = self._load_source(artifact)
        prompt = self._build_prompt(artifact, profile, source_text)

        response = self._generate_with_retry(prompt, artifact.get("file_name", "unknown"))

        if not response.text:
            raise RuntimeError(f"Gemini returned an empty response for {artifact.get('file_name')}")

        result = ArtifactReview.model_validate_json(response.text)
        return result.model_dump()

    def _generate_with_retry(self, prompt: str, artifact_name: str):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ArtifactReview,
                    ),
                )
            except errors.ServerError as exc:
                if getattr(exc, "status_code", None) != 503 or attempt >= self.max_retries:
                    raise

                wait_seconds = self.retry_delay_seconds * (2**attempt)
                print(
                    f"Gemini temporarily unavailable for {artifact_name}; "
                    f"retry {attempt + 1}/{self.max_retries} in {wait_seconds:.1f}s...",
                    flush=True,
                )
                time.sleep(wait_seconds)

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
