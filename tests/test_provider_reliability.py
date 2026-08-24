"""Comprehensive tests for LLM provider reliability, error classification, and retry backoff."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from knowledge_engineering.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    classify_exception,
    extract_retry_after,
)
from knowledge_engineering.llm.generator import (
    GeminiGenerator,
    GroqGenerator,
    generate_structured,
)


def test_extract_retry_after_from_header() -> None:
    class MockResponse:
        headers = {"retry-after": "3.5"}

    class MockHTTPError(Exception):
        response = MockResponse()

    exc = MockHTTPError("Rate limit exceeded")
    delay = extract_retry_after(exc)
    assert delay == 3.5


def test_extract_retry_after_from_text_patterns() -> None:
    patterns = [
        ("Error 429: Please try again in 2.34s.", 2.34),
        ("Rate limit reached: retry after 5.0 seconds", 5.0),
        ("Rate limit: wait 4s before next request", 4.0),
        ("Quota exhausted: 1.5s remaining before reset", 1.5),
    ]
    for text, expected in patterns:
        exc = Exception(text)
        assert extract_retry_after(exc) == expected


def test_classify_exception_rate_limit() -> None:
    exc = Exception("Error code: 429 - {'error': {'message': 'Rate limit reached. Please try again in 2.0s.'}}")
    classified = classify_exception(exc, provider="groq", model="test-model")
    assert isinstance(classified, LLMRateLimitError)
    assert classified.code == "LLM_RATE_LIMIT"
    assert classified.retryable is True
    assert classified.retry_after == 2.0
    assert classified.provider == "groq"
    assert classified.model == "test-model"


def test_classify_exception_auth_error() -> None:
    exc = Exception("Error code: 401 - {'error': {'message': 'Invalid API Key provided'}}")
    classified = classify_exception(exc, provider="groq", model="test-model")
    assert isinstance(classified, LLMAuthError)
    assert classified.code == "LLM_AUTH_ERROR"
    assert classified.retryable is False


def test_classify_exception_bad_request_context_length() -> None:
    exc = Exception("Error code: 400 - Context length exceeded maximum tokens")
    classified = classify_exception(exc)
    assert isinstance(classified, LLMBadRequestError)
    assert classified.retryable is False


def test_classify_exception_server_error() -> None:
    exc = Exception("Error code: 503 - Service Unavailable: Backend overloaded")
    classified = classify_exception(exc)
    assert isinstance(classified, LLMServerError)
    assert classified.retryable is True


def test_classify_exception_timeout() -> None:
    exc = TimeoutError("Connection to api.groq.com timed out")
    classified = classify_exception(exc)
    assert isinstance(classified, LLMTimeoutError)
    assert classified.retryable is True


def test_groq_generator_fast_fail_on_auth_error() -> None:
    gen = GroqGenerator(api_key="invalid_key", model="test_model", max_retries=3)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("401 Unauthorized: Invalid API Key")
    gen.client = mock_client

    with pytest.raises(LLMAuthError):
        gen.generate("Test prompt")

    # Fast-fail must NOT retry
    assert mock_client.chat.completions.create.call_count == 1


def test_groq_generator_retries_rate_limit_and_succeeds() -> None:
    gen = GroqGenerator(api_key="mock_key", model="test_model", max_retries=2, base_delay=0.01)
    mock_client = MagicMock()
    success_resp = MagicMock()
    success_resp.choices = [MagicMock(message=MagicMock(content='{"result": "ok"}'))]

    mock_client.chat.completions.create.side_effect = [
        Exception("Error 429: Please try again in 0.05s"),
        success_resp,
    ]
    gen.client = mock_client

    with patch("time.sleep") as mock_sleep:
        res = gen.generate("Test prompt")
        assert res == '{"result": "ok"}'
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1


def test_groq_generator_exhausts_retries_and_raises_rate_limit_error() -> None:
    gen = GroqGenerator(api_key="mock_key", model="test_model", max_retries=2, base_delay=0.01)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Error 429: Rate limit reached")
    gen.client = mock_client

    with patch("time.sleep"):
        with pytest.raises(LLMRateLimitError) as exc_info:
            gen.generate("Test prompt")
        assert exc_info.value.code == "LLM_RATE_LIMIT"
        assert mock_client.chat.completions.create.call_count == 3


def test_generate_structured_propagates_llm_error_without_schema_repair() -> None:
    class FailingGenerator:
        provider = "groq"
        model = "test"

        def generate(self, prompt: str) -> str:
            raise LLMRateLimitError("Rate limit reached")

    gen = FailingGenerator()
    with pytest.raises(LLMRateLimitError):
        generate_structured(
            gen,  # type: ignore
            "prompt",
            lambda c: json.loads(c),
            max_repair_retries=2,
        )
