"""Provider-neutral text generation for grounded RAG answers and investigation stages."""
from __future__ import annotations

import json
import random
import time
from typing import Any, Callable, Protocol, TypeVar

from google import genai
from google.genai import errors as gemini_errors, types
from groq import Groq

from .errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    classify_exception,
)

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

    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.client = Groq(api_key=api_key, timeout=timeout)

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
                    raise LLMEmptyResponseError("Groq returned an empty response", provider=self.provider, model=self.model)
                return content
            except Exception as exc:
                classified = classify_exception(exc, provider=self.provider, model=self.model)
                last_error = classified

                # Non-retryable errors fail immediately (401 Auth, 400 Bad Request)
                if not classified.retryable:
                    raise classified from exc

                if attempt < self.max_retries:
                    if isinstance(classified, LLMRateLimitError) and classified.retry_after is not None:
                        delay = min(self.max_delay, max(classified.retry_after, self.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
                    else:
                        delay = min(self.max_delay, (self.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
                    time.sleep(delay)

        if isinstance(last_error, LLMError):
            raise last_error
        raise RuntimeError(f"Groq generation failed ({self.model}): {last_error}") from last_error


class GeminiGenerator:
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
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
                    raise LLMEmptyResponseError("Gemini returned an empty response", provider=self.provider, model=self.model)
                return response.text
            except Exception as exc:
                classified = classify_exception(exc, provider=self.provider, model=self.model)
                last_error = classified

                if not classified.retryable:
                    raise classified from exc

                if attempt < self.max_retries:
                    if isinstance(classified, LLMRateLimitError) and classified.retry_after is not None:
                        delay = min(self.max_delay, max(classified.retry_after, self.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
                    else:
                        delay = min(self.max_delay, (self.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
                    time.sleep(delay)

        if isinstance(last_error, LLMError):
            raise last_error
        raise RuntimeError(f"Gemini generation failed ({self.model}): {last_error}") from last_error


def create_generator(config: Any) -> LLMGenerator:
    provider = config.provider.lower()
    max_retries = getattr(config, "max_retries", 3)
    if provider == "groq":
        return GroqGenerator(config.api_key, config.model, max_retries)
    if provider == "gemini":
        return GeminiGenerator(config.api_key, config.model, max_retries)
    raise ValueError(f"Unsupported LLM_PROVIDER={config.provider!r}. Supported providers: gemini, groq")


def generate_structured(
    generator: LLMGenerator,
    prompt: str,
    parser_func: Callable[[str], T],
    max_repair_retries: int = 1,
) -> T:
    """Generate and parse structured output with schema-aware repair retries.

    If the generator itself raises an LLM infrastructure error (RateLimit, Auth, Timeout, Server),
    it is propagated immediately rather than formatted as an LLM prompt repair.
    """
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
