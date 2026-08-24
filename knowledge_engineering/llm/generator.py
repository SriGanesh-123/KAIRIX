"""Provider-neutral text generation for grounded RAG answers."""
from __future__ import annotations

import json
import time
from typing import Any, Protocol

from google import genai
from google.genai import errors as gemini_errors, types
from groq import Groq


class LLMGenerator(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> str:
        """Generate a text response from a grounded prompt."""


_STRUCTURED_OUTPUT_SYSTEM_PROMPT = (
    "Return valid JSON only. Follow the exact JSON schema requested by the current user prompt. "
    "Do not add markdown fences or explanatory text outside the JSON object. "
    "The schema can differ between investigation stages, so use the schema explicitly requested."
)


def _generate_with_retry(request, max_retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = request()
            content = getattr(response, "content", None)
            if not content:
                raise RuntimeError("LLM returned an empty response")
            return content
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM generation failed: {last_error}")


class GroqGenerator:
    provider = "groq"

    def __init__(self, api_key: str, model: str, max_retries: int = 1) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = Groq(api_key=api_key)

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _STRUCTURED_OUTPUT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Groq returned an empty response")
                return content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Groq generation failed: {last_error}")


class GeminiGenerator:
    provider = "gemini"

    def __init__(self, api_key: str, model: str, max_retries: int = 1) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                if not response.text:
                    raise RuntimeError("Gemini returned an empty response")
                return response.text
            except gemini_errors.APIError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Gemini generation failed: {last_error}")


def create_generator(config: Any) -> LLMGenerator:
    provider = config.provider.lower()
    if provider == "groq":
        return GroqGenerator(config.api_key, config.model, config.max_retries)
    if provider == "gemini":
        return GeminiGenerator(config.api_key, config.model, config.max_retries)
    raise ValueError(f"Unsupported LLM_PROVIDER={config.provider!r}. Supported providers: gemini, groq")


def parse_generation(content: str) -> dict[str, Any]:
    """Parse the final answer envelope without weakening its required fields."""
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    answer = value.get("answer")
    evidence_ids = value.get("evidence_ids", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM response must contain a non-empty answer")
    if evidence_ids is None:
        evidence_ids = []
    if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
        raise ValueError("evidence_ids must be a list of strings")
    return {"answer": answer.strip(), "evidence_ids": evidence_ids}
