"""Gemini adapter selected explicitly by configuration."""
from __future__ import annotations

import json
import time
from typing import Any

from google import genai
from google.genai import errors, types

from .reviewer import ArtifactReview, LLMReviewer, build_prompt, load_source


class GeminiReviewer(LLMReviewer):
    provider = "gemini"

    def __init__(self, api_key: str, model: str, max_retries: int = 1) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)

    def review(self, artifact: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(artifact, profile, load_source(artifact))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            print(f"Gemini review: {artifact.get('file_name', 'unknown')} using {self.model} (attempt {attempt + 1}/{self.max_retries + 1})", flush=True)
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ArtifactReview,
                    ),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned an empty response")
                print(f"Gemini review SUCCESS: {self.model}", flush=True)
                return ArtifactReview.model_validate_json(response.text).model_dump()
            except errors.APIError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini review failed for {artifact.get('file_name', 'unknown')}: {last_error}")
