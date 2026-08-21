"""Groq SDK adapter with bounded context and no provider fallback."""
from __future__ import annotations

import time
from typing import Any

from groq import Groq

from .reviewer import ArtifactReview, LLMReviewer, build_prompt, load_source


class GroqReviewer(LLMReviewer):
    provider = "groq"

    # Groq free/on-demand limits can be reached by large prompts. Keep a
    # conservative character budget for the combined metadata/profile/source.
    # This is configurable later if a higher service tier is used.
    max_prompt_chars = 26000

    def __init__(self, api_key: str, model: str, max_retries: int = 1) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = Groq(api_key=api_key)

    def _bounded_prompt(self, artifact: dict[str, Any], profile: dict[str, Any]) -> str:
        source = load_source(artifact)
        full_prompt = build_prompt(artifact, profile, source)
        if len(full_prompt) <= self.max_prompt_chars:
            return full_prompt

        # Preserve the deterministic metadata/profile and the beginning/end of
        # the source so joins, CASE logic, filters, and query structure are not
        # discarded wholesale. Explicitly tell the model that source was
        # bounded so it does not infer missing sections.
        marker = (
            "\n\n[ORIGINAL SOURCE CONTEXT BOUNDED FOR MODEL LIMITS. "
            "The omitted middle section must not be inferred.]\n\n"
        )
        remaining = self.max_prompt_chars - len(marker)
        head_chars = int(remaining * 0.65)
        tail_chars = remaining - head_chars
        bounded_source = source[:head_chars] + marker + source[-tail_chars:]
        return build_prompt(artifact, profile, bounded_source)

    def review(self, artifact: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        prompt = self._bounded_prompt(artifact, profile)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only valid JSON matching the requested review fields.",
                        },
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
        raise RuntimeError(
            f"Groq review failed for {artifact.get('file_name', 'unknown')}: {last_error}"
        )
