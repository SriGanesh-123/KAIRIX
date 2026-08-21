"""Groq OpenAI-compatible HTTP adapter with no provider fallback."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .reviewer import ArtifactReview, LLMReviewer, build_prompt, load_source


class GroqReviewer(LLMReviewer):
    provider = "groq"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str, max_retries: int = 1) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries

    def review(self, artifact: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(artifact, profile, load_source(artifact))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON matching the requested review fields."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                result = ArtifactReview.model_validate_json(content).model_dump()
                print(f"{self.provider} review SUCCESS: {self.model}", flush=True)
                return result
            except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Groq review failed for {artifact.get('file_name', 'unknown')}: {last_error}")
