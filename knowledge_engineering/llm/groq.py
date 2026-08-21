"""Groq SDK adapter with no provider fallback."""
from __future__ import annotations

import time
from typing import Any

from groq import Groq

from .reviewer import ArtifactReview, LLMReviewer, build_prompt, load_source


class GroqReviewer(LLMReviewer):
    provider = "groq"

    def __init__(self, api_key: str, model: str, max_retries: int = 1) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = Groq(api_key=api_key)

    def review(self, artifact: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(artifact, profile, load_source(artifact))
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
