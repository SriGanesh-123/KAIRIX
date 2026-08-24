"""Provider-neutral text generation for grounded RAG answers and investigation stages."""
from __future__ import annotations

import json
import random
import time
from typing import Any, Callable, Protocol, TypeVar

from google import genai
from google.genai import errors as gemini_errors, types
from groq import Groq

from .config import LLMConfig
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
from .gateway import InvestigationBudget, LLMGateway, _STRUCTURED_OUTPUT_SYSTEM_PROMPT

T = TypeVar("T")


class LLMGenerator(Protocol):
    provider: str
    model: str

    def generate(self, prompt: str) -> str:
        """Generate a text response from a grounded prompt."""


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
        max_delay: float = 60.0,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        # max_retries=0 ensures the centralized LLMGateway owns all transport retry policies
        self.client = Groq(api_key=api_key, timeout=timeout, max_retries=0)

    def call(self, prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Raw API call returning content, headers, and usage."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt or _STRUCTURED_OUTPUT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
            timeout=self.timeout,
        )
        content = response.choices[0].message.content or ""
        raw_headers = getattr(response, "_headers", None)
        headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            if hasattr(raw_usage, "model_dump"):
                usage = raw_usage.model_dump()
            elif isinstance(raw_usage, dict):
                usage = dict(raw_usage)
            else:
                usage = {
                    "prompt_tokens": getattr(raw_usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(raw_usage, "completion_tokens", 0),
                    "total_tokens": getattr(raw_usage, "total_tokens", 0),
                }
        else:
            usage = {}
        return content, headers, usage

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content, _, _ = self.call(prompt)
                if not content:
                    raise LLMEmptyResponseError("Groq returned an empty response", provider=self.provider, model=self.model)
                return content
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
        raise RuntimeError(f"Groq generation failed ({self.model}): {last_error}") from last_error


class GeminiGenerator:
    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.client = genai.Client(api_key=api_key)

    def call(self, prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any], dict[str, Any]]:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        content = response.text or ""
        return content, {}, {}

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content, _, _ = self.call(prompt)
                if not content:
                    raise LLMEmptyResponseError("Gemini returned an empty response", provider=self.provider, model=self.model)
                return content
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


def create_generator(config: Any) -> LLMGateway:
    """Create provider adapter and wrap with centralized LLMGateway."""
    provider = config.provider.lower()
    max_retries = getattr(config, "max_retries", 3)
    base_delay = getattr(config, "base_delay", 1.0)
    max_delay = getattr(config, "max_delay", 60.0)
    timeout = getattr(config, "timeout", 30.0)

    if provider == "groq":
        adapter = GroqGenerator(
            api_key=config.api_key,
            model=config.model,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            timeout=timeout,
        )
    elif provider == "gemini":
        adapter = GeminiGenerator(
            api_key=config.api_key,
            model=config.model,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER={config.provider!r}. Supported providers: gemini, groq")

    return LLMGateway(adapter=adapter, config=config if isinstance(config, LLMConfig) else None)


def generate_structured(
    generator: LLMGenerator,
    prompt: str,
    parser_func: Callable[[str], T],
    max_repair_retries: int = 1,
    stage: str = "general",
    budget: InvestigationBudget | None = None,
) -> T:
    """Generate and parse structured output with schema-aware repair retries.

    If the generator itself raises an LLM infrastructure error (RateLimit, Auth, Timeout, Server),
    it is propagated immediately rather than formatted as an LLM prompt repair.
    """
    current_prompt = prompt
    last_validation_error: str = ""

    for attempt in range(max_repair_retries + 1):
        if hasattr(generator, "generate") and "stage" in getattr(generator.generate, "__code__", object()).co_varnames:
            raw_output = generator.generate(current_prompt, stage=stage, budget=budget)
        else:
            raw_output = generator.generate(current_prompt)

        try:
            return parser_func(raw_output)
        except Exception as exc:
            last_validation_error = str(exc)
            if attempt < max_repair_retries:
                if budget is not None and not budget.record_repair():
                    break
                current_prompt = _build_repair_prompt(prompt, last_validation_error)
                time.sleep(0.5)

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
