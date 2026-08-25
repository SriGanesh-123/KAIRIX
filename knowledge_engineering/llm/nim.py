"""NVIDIA NIM provider implementation using OpenAI-compatible API."""
from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any

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
from .gateway import _STRUCTURED_OUTPUT_SYSTEM_PROMPT
from .reviewer import ArtifactReview, LLMReviewer, build_prompt, load_source

logger = logging.getLogger(__name__)

DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _sanitize_secrets(text: str, *secrets: str) -> str:
    """Mask secret strings and authorization tokens to prevent leakage."""
    sanitized = text
    for secret in secrets:
        if secret and len(secret) > 3:
            sanitized = sanitized.replace(secret, "***")
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\-\.]+", "Bearer ***", sanitized, flags=re.IGNORECASE)
    return sanitized


def _normalize_endpoint(base_url: str) -> str:
    """Normalize NIM base URL into full chat/completions endpoint."""
    url = (base_url or DEFAULT_NIM_BASE_URL).strip().rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    return url


class NIMHTTPError(Exception):
    """Encapsulates HTTP errors from NIM REST endpoint with status code and headers."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        headers: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = status_code
        self.headers = headers or {}
        self.response = type("Response", (), {"headers": self.headers, "status_code": status_code})()


class NIMClient:
    """Lightweight, secret-safe OpenAI-compatible HTTP client for NVIDIA NIM."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_NIM_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty for NIM provider")
        self._api_key = api_key
        self.model = model
        self.base_url = (base_url or DEFAULT_NIM_BASE_URL).strip().rstrip("/")
        self.endpoint = _normalize_endpoint(self.base_url)
        self.timeout = max(1.0, float(timeout))

    def __repr__(self) -> str:
        return f"NIMClient(model={self.model!r}, endpoint={self.endpoint!r})"

    def chat_completion(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Send chat completion request to NVIDIA NIM endpoint.

        Returns:
            (content_text, response_headers_dict, usage_dict)
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt or _STRUCTURED_OUTPUT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "KAIRIX-NIM-Adapter/1.0",
        }

        req = urllib.request.Request(
            url=self.endpoint,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
                body_bytes = resp.read()
                body = body_bytes.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raw_headers = {str(k).lower(): str(v) for k, v in exc.headers.items()} if exc.headers else {}
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
                err_json = json.loads(err_body)
                err_msg = (
                    err_json.get("error", {}).get("message")
                    or err_json.get("message")
                    or err_body
                )
            except Exception:
                err_msg = f"HTTP {exc.code} {exc.reason}"

            sanitized_msg = _sanitize_secrets(str(err_msg), self._api_key)
            raise NIMHTTPError(
                f"NIM API error {exc.code}: {sanitized_msg}",
                status_code=exc.code,
                headers=raw_headers,
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                raise TimeoutError(f"NIM request to {self.endpoint} timed out") from exc
            raise RuntimeError(_sanitize_secrets(f"NIM connection error: {exc}", self._api_key)) from exc
        except Exception as exc:
            raise RuntimeError(_sanitize_secrets(f"NIM transport error: {exc}", self._api_key)) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"NIM returned invalid JSON body: {exc}") from exc

        choices = parsed.get("choices") or []
        if not choices:
            content = ""
        else:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                msg = first_choice.get("message") or {}
                content = msg.get("content") or ""
            else:
                content = ""

        raw_usage = parsed.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
            "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
            "total_tokens": int(raw_usage.get("total_tokens") or 0),
        }

        return content, raw_headers, usage


class NIMGenerator:
    """NVIDIA NIM text generator implementing LLMGenerator and ProviderAdapter protocols."""
    provider = "nim"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_NIM_BASE_URL,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or DEFAULT_NIM_BASE_URL).strip().rstrip("/")
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.client = NIMClient(
            api_key=api_key,
            model=model,
            base_url=self.base_url,
            timeout=timeout,
        )

    def __repr__(self) -> str:
        return f"NIMGenerator(model={self.model!r}, base_url={self.base_url!r})"

    def call(self, prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Raw API call returning content, headers, and token usage."""
        return self.client.chat_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=2048,
        )

    def generate(self, prompt: str) -> str:
        """Generate text with self-contained retries and typed error handling."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content, _, _ = self.call(prompt)
                if not content:
                    raise LLMEmptyResponseError(
                        "NIM returned an empty response",
                        provider=self.provider,
                        model=self.model,
                    )
                return content
            except Exception as exc:
                classified = classify_exception(exc, provider=self.provider, model=self.model)
                last_error = classified

                if not classified.retryable:
                    raise classified from exc

                if attempt < self.max_retries:
                    if isinstance(classified, LLMRateLimitError) and classified.retry_after is not None:
                        delay = min(
                            self.max_delay,
                            max(classified.retry_after, self.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4),
                        )
                    else:
                        delay = min(
                            self.max_delay,
                            (self.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4),
                        )
                    time.sleep(delay)

        if isinstance(last_error, LLMError):
            raise last_error
        raise RuntimeError(f"NIM generation failed ({self.model}): {last_error}") from last_error


class NIMReviewer(LLMReviewer):
    """NVIDIA NIM artifact reviewer implementing the LLMReviewer protocol."""
    provider = "nim"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_NIM_BASE_URL,
        max_retries: int = 1,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or DEFAULT_NIM_BASE_URL).strip().rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = NIMClient(
            api_key=api_key,
            model=model,
            base_url=self.base_url,
            timeout=timeout,
        )

    def review(self, artifact: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(artifact, profile, load_source(artifact))
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                content, _, _ = self.client.chat_completion(
                    prompt=prompt,
                    system_prompt="Return only valid JSON matching the requested review fields, including code_fix.",
                    temperature=0.1,
                )
                if not content:
                    raise ValueError("NIM returned an empty response")
                result = ArtifactReview.model_validate_json(content).model_dump()
                print(f"{self.provider} review SUCCESS: {self.model}", flush=True)
                return result
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"NIM review failed for {artifact.get('file_name', 'unknown')}: {last_error}")
