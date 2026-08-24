"""Typed error hierarchy and exception classification for LLM providers."""
from __future__ import annotations

import re
from typing import Any


class LLMError(Exception):
    """Base exception for all LLM provider failures."""
    code: str = "LLM_ERROR"
    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model


class LLMRateLimitError(LLMError):
    """Provider rate limit reached (HTTP 429 / RPM / TPM)."""
    code: str = "LLM_RATE_LIMIT"
    retryable: bool = True

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """Provider request timed out."""
    code: str = "LLM_TIMEOUT"
    retryable: bool = True


class LLMServerError(LLMError):
    """Provider server error (HTTP 500 / 502 / 503 / 504)."""
    code: str = "LLM_SERVER_ERROR"
    retryable: bool = True


class LLMAuthError(LLMError):
    """Authentication or authorization failure (HTTP 401 / 403). Non-retryable."""
    code: str = "LLM_AUTH_ERROR"
    retryable: bool = False


class LLMBadRequestError(LLMError):
    """Invalid request or context size exceeded (HTTP 400 / 413). Non-retryable."""
    code: str = "LLM_BAD_REQUEST"
    retryable: bool = False


class LLMEmptyResponseError(LLMError):
    """Provider returned an empty response body."""
    code: str = "LLM_EMPTY_RESPONSE"
    retryable: bool = True


class LLMSchemaError(LLMError):
    """Response failed structured JSON schema validation."""
    code: str = "LLM_SCHEMA_ERROR"
    retryable: bool = True


class LLMConfigurationError(LLMError):
    """Missing or invalid LLM provider configuration. Non-retryable."""
    code: str = "LLM_CONFIGURATION_ERROR"
    retryable: bool = False


def extract_retry_after(exc: Exception) -> float | None:
    """Extract retry delay seconds from HTTP headers or error message body.

    Handles:
    - HTTP response header 'Retry-After'
    - Text patterns like 'try again in 2.34s', 'retry after 3s', 'retry in 4.5 seconds'
    """
    # 1. Try response header if present
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers and hasattr(headers, "get"):
            header_val = headers.get("retry-after") or headers.get("Retry-After")
            if header_val:
                try:
                    val = float(header_val)
                    if val > 0:
                        return val
                except (ValueError, TypeError):
                    pass

    # 2. Try regex extraction from string representation
    text = str(exc)
    patterns = [
        r"(?:try again in|retry after|retry in|wait)\s+([\d\.]+)\s*(?:s|sec|seconds)?",
        r"([\d\.]+)\s*s(?:econds?)?\s+(?:remaining|before)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                if val > 0:
                    return val
            except (ValueError, TypeError):
                continue

    return None


def classify_exception(exc: Exception, *, provider: str = "", model: str = "") -> LLMError:
    """Classify a provider SDK exception into our typed LLM error hierarchy."""
    if isinstance(exc, LLMError):
        return exc

    msg = str(exc)
    lowered = msg.lower()
    retry_after = extract_retry_after(exc)

    # Rate limits (429)
    if "429" in msg or "rate_limit" in lowered or "rate limit" in lowered or "quota" in lowered or "tokens per minute" in lowered:
        return LLMRateLimitError(msg, retry_after=retry_after, provider=provider, model=model)

    # Auth failures (401 / 403 / invalid api key)
    if "401" in msg or "403" in msg or "invalid api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return LLMAuthError(msg, provider=provider, model=model)

    # Bad request / Prompt too large (400 / 413)
    if "400" in msg or "413" in msg or "too large" in lowered or "context length" in lowered or "maximum context" in lowered:
        return LLMBadRequestError(msg, provider=provider, model=model)

    # Timeouts
    if isinstance(exc, TimeoutError) or "timeout" in lowered or "timed out" in lowered or "deadline exceeded" in lowered:
        return LLMTimeoutError(msg, provider=provider, model=model)

    # Server errors (500 / 502 / 503 / 504)
    if any(code in msg for code in ("500", "502", "503", "504")) or "internal server error" in lowered or "service unavailable" in lowered or "bad gateway" in lowered:
        return LLMServerError(msg, provider=provider, model=model)

    # Empty response
    if "empty response" in lowered:
        return LLMEmptyResponseError(msg, provider=provider, model=model)

    return LLMError(msg, provider=provider, model=model)
