"""Groq SDK adapter with compact context and no provider fallback."""
from __future__ import annotations

import time
from typing import Any

from groq import Groq

from .reviewer import ArtifactReview, LLMReviewer, load_source


class GroqReviewer(LLMReviewer):
    provider = "groq"
    # Target a conservative request size below the 8K TPM limit.
    max_prompt_chars = 12000

    def __init__(self, api_key: str, model: str, max_retries: int = 1) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = Groq(api_key=api_key)

    @staticmethod
    def _compact_json(value: Any, max_chars: int) -> str:
        import json
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return text if len(text) <= max_chars else text[:max_chars] + "...[truncated]"

    def _build_bounded_prompt(self, artifact: dict[str, Any], profile: dict[str, Any]) -> str:
        source = load_source(artifact)
        artifact_text = self._compact_json(artifact, 1800)
        profile_text = self._compact_json(profile, 3000)
        remaining = max(3000, self.max_prompt_chars - len(artifact_text) - len(profile_text) - 1800)
        head = int(remaining * 0.70)
        tail = remaining - head
        if len(source) > remaining:
            source = (
                source[:head]
                + "\n\n[ORIGINAL SOURCE MIDDLE OMITTED FOR MODEL LIMITS. Do not infer omitted text.]\n\n"
                + source[-tail:]
            )
        return f"""You are the Artifact Review component of a legacy reverse-engineering Knowledge Engineering Agent.

Review ONLY the supplied artifact metadata, deterministic profile, and source excerpt.
Do not invent dependencies, business rules, or evidence. Put uncertain items in semantic_gaps.

SOURCE INTEGRITY RULE:
You are READ-ONLY. Never modify, rewrite, or claim to have modified the original source artifact.
If you identify a possible code defect or code change opportunity, report it only as a developer-owned code_fix finding.
Set source_modified to false. The developer is responsible for deciding and implementing any source-code change.
If no code fix is supported by evidence, set code_fix_required to false and leave the other code_fix fields empty/default.

Return JSON with: purpose, summary, key_findings, dependencies, business_rules, semantic_gaps,
evidence_candidates, confidence (0.0-1.0), needs_deeper_analysis, reason, and code_fix.
The code_fix object MUST contain exactly these conceptual fields:
code_fix_required (boolean), source_file (string), location (string), issue (string),
evidence (array of strings), recommended_action (string), action_owner (string, use DEVELOPER),
source_modified (boolean, MUST be false).

ARTIFACT METADATA:
{artifact_text}

DETERMINISTIC PROFILE:
{profile_text}

SOURCE EXCERPT:
{source}
"""

    def review(self, artifact: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_bounded_prompt(artifact, profile)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON matching the requested review fields, including code_fix."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Groq returned an empty response")
                result = ArtifactReview.model_validate_json(content).model_dump()
                print(f"{self.provider} review SUCCESS: {self.model}", flush=True)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Groq review failed for {artifact.get('file_name', 'unknown')}: {last_error}")
