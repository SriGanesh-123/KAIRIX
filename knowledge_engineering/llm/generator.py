"""Provider-neutral text generation for grounded RAG answers and investigation stages."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Protocol, TypeVar

from google import genai
from google.genai import errors as gemini_errors, types
from groq import Groq

T = TypeVar("T")


class LLMGenerator(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> str:
        """Generate a text response from a grounded prompt."""


_STRUCTURED_OUTPUT_SYSTEM_PROMPT = (
    "Return valid JSON only. Follow the exact JSON schema requested by the current user prompt. "
    "Do not add markdown fences, preamble, or explanatory text outside the JSON object. "
    "The schema differs between investigation stages, so use the schema explicitly requested."
)


def _build_repair_prompt(prompt: str, error_message: str) -> str:
    return (
        f"{prompt}\n\n"
        "Your previous response failed schema validation. Please fix the error and return ONLY "
        "a valid JSON object matching the requested schema.\n"
        f"Validation error: {error_message}"
    )


class GroqGenerator:
    provider = "groq"

    def __init__(self, api_key: str, model: str, max_retries: int = 3) -> None:
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
        raise RuntimeError(f"Groq generation failed ({self.model}): {last_error}") from last_error


class GeminiGenerator:
    provider = "gemini"

    def __init__(self, api_key: str, model: str, max_retries: int = 3) -> None:
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
        raise RuntimeError(f"Gemini generation failed ({self.model}): {last_error}") from last_error


def create_generator(config: Any) -> LLMGenerator:
    provider = config.provider.lower()
    if provider == "groq":
        return GroqGenerator(config.api_key, config.model, config.max_retries)
    if provider == "gemini":
        return GeminiGenerator(config.api_key, config.model, config.max_retries)
    raise ValueError(f"Unsupported LLM_PROVIDER={config.provider!r}. Supported providers: gemini, groq")


def generate_structured(
    generator: LLMGenerator,
    prompt: str,
    parser_func: Callable[[str], T],
    max_repair_retries: int = 1,
) -> T:
    """Generate and parse structured output with schema-aware repair retries."""
    current_prompt = prompt
    last_validation_error: str = ""

    for attempt in range(max_repair_retries + 1):
        raw_output = generator.generate(current_prompt)
        try:
            return parser_func(raw_output)
        except Exception as exc:
            last_validation_error = str(exc)
            if attempt < max_repair_retries:
                current_prompt = _build_repair_prompt(prompt, last_validation_error)
                time.sleep(1)

    raise ValueError(
        f"Failed to produce valid structured output after {max_repair_retries + 1} attempts. "
        f"Last validation error: {last_validation_error}"
    )


def parse_generation(content: str) -> dict[str, Any]:
    """Parse legacy RAG generation envelope without weakening required fields."""
    from ..investigation_agent.contracts import extract_json_payload
    value = extract_json_payload(content)
    answer = value.get("answer")
    evidence_ids = value.get("evidence_ids", [])
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("LLM response must contain a non-empty answer")
    if evidence_ids is None:
        evidence_ids = []
    if not isinstance(evidence_ids, list) or not all(isinstance(item, str) for item in evidence_ids):
        raise ValueError("evidence_ids must be a list of strings")
    return {"answer": answer.strip(), "evidence_ids": evidence_ids}
