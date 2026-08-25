"""Comprehensive unit tests for NVIDIA NIM provider integration."""
from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from knowledge_engineering.investigation_agent.agent import InvestigationAgent
from knowledge_engineering.llm.config import LLMConfig
from knowledge_engineering.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMEmptyResponseError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    classify_exception,
)
from knowledge_engineering.llm.factory import create_reviewer
from knowledge_engineering.llm.gateway import LLMGateway
from knowledge_engineering.llm.generator import create_generator, generate_structured
from knowledge_engineering.llm.nim import (
    DEFAULT_NIM_BASE_URL,
    NIMClient,
    NIMGenerator,
    NIMHTTPError,
    NIMReviewer,
    _normalize_endpoint,
    _sanitize_secrets,
)


def _make_mock_http_response(
    body_dict: dict[str, Any] | str,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock urllib response context manager."""
    if isinstance(body_dict, dict):
        raw_bytes = json.dumps(body_dict).encode("utf-8")
    else:
        raw_bytes = body_dict.encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = status_code
    mock_resp.headers = headers or {"content-type": "application/json"}
    mock_resp.read.return_value = raw_bytes
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def _make_mock_http_error(
    status_code: int,
    body_dict: dict[str, Any] | str,
    headers: dict[str, str] | None = None,
) -> urllib.error.HTTPError:
    """Create a mock urllib HTTPError."""
    if isinstance(body_dict, dict):
        raw_bytes = json.dumps(body_dict).encode("utf-8")
    else:
        raw_bytes = body_dict.encode("utf-8")

    fp = io.BytesIO(raw_bytes)
    hdrs = MagicMock()
    header_dict = headers or {"content-type": "application/json"}
    hdrs.items.return_value = list(header_dict.items())
    hdrs.get = lambda k, default=None: header_dict.get(k.lower(), header_dict.get(k, default))

    return urllib.error.HTTPError(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        code=status_code,
        msg=f"HTTP {status_code}",
        hdrs=hdrs,
        fp=fp,
    )


# ==============================================================================
# 1. Configuration & Provider Resolution Tests
# ==============================================================================

def test_nim_config_from_env_defaults() -> None:
    env = {
        "LLM_PROVIDER": "nim",
        "LLM_MODEL": "nvidia/nemotron-3-ultra-550b-a55b",
        "LLM_API_KEY": "nvapi-secret-key-12345",
    }
    with patch.dict(os.environ, env, clear=True):
        config = LLMConfig.from_env()
        assert config is not None
        assert config.provider == "nim"
        assert config.model == "nvidia/nemotron-3-ultra-550b-a55b"
        assert config.api_key == "nvapi-secret-key-12345"
        assert config.base_url == DEFAULT_NIM_BASE_URL


def test_nim_config_custom_base_url() -> None:
    env = {
        "LLM_PROVIDER": "nim",
        "LLM_MODEL": "nvidia/nemotron-3-ultra-550b-a55b",
        "LLM_API_KEY": "nvapi-secret-key-12345",
        "NIM_BASE_URL": "https://custom.nim.host/v1",
    }
    with patch.dict(os.environ, env, clear=True):
        config = LLMConfig.from_env()
        assert config is not None
        assert config.base_url == "https://custom.nim.host/v1"


def test_nim_config_fallback_to_nim_api_key() -> None:
    env = {
        "LLM_PROVIDER": "nim",
        "LLM_MODEL": "nvidia/nemotron-3-ultra-550b-a55b",
        "NIM_API_KEY": "nvapi-fallback-key-999",
    }
    with patch.dict(os.environ, env, clear=True):
        config = LLMConfig.from_env()
        assert config is not None
        assert config.api_key == "nvapi-fallback-key-999"


def test_nim_config_missing_api_key_returns_none() -> None:
    env = {
        "LLM_PROVIDER": "nim",
        "LLM_MODEL": "nvidia/nemotron-3-ultra-550b-a55b",
    }
    with patch.dict(os.environ, env, clear=True):
        config = LLMConfig.from_env()
        assert config is None


def test_create_generator_with_nim_provider() -> None:
    config = LLMConfig(
        provider="nim",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        api_key="nvapi-test-key",
        base_url="https://integrate.api.nvidia.com/v1",
    )
    gw = create_generator(config)
    assert isinstance(gw, LLMGateway)
    assert gw.provider == "nim"
    assert gw.model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert isinstance(gw.adapter, NIMGenerator)
    assert gw.adapter.base_url == "https://integrate.api.nvidia.com/v1"


def test_create_reviewer_with_nim_provider() -> None:
    config = LLMConfig(
        provider="nim",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        api_key="nvapi-test-key",
    )
    rev = create_reviewer(config)
    assert isinstance(rev, NIMReviewer)
    assert rev.provider == "nim"
    assert rev.model == "nvidia/nemotron-3-ultra-550b-a55b"


# ==============================================================================
# 2. Secret Sanitization & Endpoint Normalization Tests
# ==============================================================================

def test_sanitize_secrets_masks_keys_and_authorization() -> None:
    secret = "nvapi-very-secret-token-xyz"
    raw = f"Error calling endpoint with Authorization: Bearer {secret} and key={secret}"
    sanitized = _sanitize_secrets(raw, secret)
    assert secret not in sanitized
    assert "Bearer ***" in sanitized
    assert "***" in sanitized


def test_normalize_endpoint() -> None:
    assert _normalize_endpoint("https://integrate.api.nvidia.com/v1") == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert _normalize_endpoint("https://integrate.api.nvidia.com/v1/") == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert _normalize_endpoint("https://integrate.api.nvidia.com/v1/chat/completions") == "https://integrate.api.nvidia.com/v1/chat/completions"


def test_nim_generator_and_client_repr_masks_secrets() -> None:
    gen = NIMGenerator(api_key="nvapi-super-secret-key-12345", model="nvidia/nemotron-3-ultra-550b-a55b")
    rep = repr(gen)
    assert "nvapi-super-secret" not in rep
    assert "nvidia/nemotron-3-ultra-550b-a55b" in rep

    client_rep = repr(gen.client)
    assert "nvapi-super-secret" not in client_rep


# ==============================================================================
# 3. Generation & Parsing Tests (Mocked HTTP)
# ==============================================================================

def test_nim_generator_successful_call() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b")
    mock_payload = {
        "id": "chatcmpl-test-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"answer": "Grounded answer from NIM"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
    }
    mock_headers = {
        "x-ratelimit-remaining-requests": "45",
        "x-ratelimit-remaining-tokens": "9800",
    }
    mock_resp = _make_mock_http_response(mock_payload, headers=mock_headers)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        content, headers, usage = gen.call("What is PolicyCenter?")
        assert content == '{"answer": "Grounded answer from NIM"}'
        assert headers.get("x-ratelimit-remaining-requests") == "45"
        assert usage["prompt_tokens"] == 25
        assert usage["completion_tokens"] == 15
        assert usage["total_tokens"] == 40
        mock_urlopen.assert_called_once()

        # Verify request structure
        req = mock_urlopen.call_args[0][0]
        assert isinstance(req, urllib.request.Request)
        assert req.get_full_url() == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert req.get_header("Authorization") == "Bearer mock_key"
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
        assert body["messages"][1]["content"] == "What is PolicyCenter?"
        assert body["response_format"] == {"type": "json_object"}


def test_nim_generator_generate_success() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b")
    mock_payload = {
        "choices": [{"message": {"content": '{"result": "ok"}'}}],
    }
    mock_resp = _make_mock_http_response(mock_payload)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = gen.generate("Test prompt")
        assert res == '{"result": "ok"}'


def test_nim_structured_generation_and_repair() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b")
    mock_resp_invalid = _make_mock_http_response({"choices": [{"message": {"content": 'invalid json string'}}]})
    mock_resp_valid = _make_mock_http_response({"choices": [{"message": {"content": '{"parsed": 42}'}}]})

    with patch("urllib.request.urlopen", side_effect=[mock_resp_invalid, mock_resp_valid]):
        result = generate_structured(
            gen,
            "Give me parsed number",
            parser_func=lambda text: json.loads(text),
            max_repair_retries=2,
        )
        assert result == {"parsed": 42}


# ==============================================================================
# 4. Error Mapping & Classification Tests
# ==============================================================================

def test_nim_error_mapping_401_auth_error_fast_fails() -> None:
    gen = NIMGenerator(api_key="bad_key", model="nvidia/nemotron-3-ultra-550b-a55b", max_retries=3)
    http_err = _make_mock_http_error(401, {"error": {"message": "Invalid API key"}})

    with patch("urllib.request.urlopen", side_effect=http_err) as mock_urlopen:
        with pytest.raises(LLMAuthError) as exc_info:
            gen.generate("Test prompt")

        assert exc_info.value.code == "LLM_AUTH_ERROR"
        assert exc_info.value.retryable is False
        assert exc_info.value.provider == "nim"
        # Non-retryable error must fast fail on 1st attempt
        assert mock_urlopen.call_count == 1


def test_nim_error_mapping_400_bad_request() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b", max_retries=2)
    http_err = _make_mock_http_error(400, {"error": {"message": "Context length exceeded maximum tokens"}})

    with patch("urllib.request.urlopen", side_effect=http_err) as mock_urlopen:
        with pytest.raises(LLMBadRequestError) as exc_info:
            gen.generate("Test prompt")

        assert exc_info.value.code == "LLM_BAD_REQUEST"
        assert exc_info.value.retryable is False
        assert mock_urlopen.call_count == 1


def test_nim_error_mapping_429_rate_limit_with_retry_after() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b", max_retries=1, base_delay=0.01)
    http_err = _make_mock_http_error(
        429,
        {"error": {"message": "Rate limit exceeded"}},
        headers={"retry-after": "2.5"},
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(LLMRateLimitError) as exc_info:
                gen.generate("Test prompt")

            assert exc_info.value.code == "LLM_RATE_LIMIT"
            assert exc_info.value.retryable is True
            assert exc_info.value.retry_after == 2.5
            assert mock_sleep.call_count == 1


def test_nim_error_mapping_503_server_error_retries_and_recovers() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b", max_retries=2, base_delay=0.01)
    http_err = _make_mock_http_error(503, {"error": {"message": "Service Unavailable"}})
    success_resp = _make_mock_http_response({"choices": [{"message": {"content": '{"recovered": true}'}}]})

    with patch("urllib.request.urlopen", side_effect=[http_err, success_resp]) as mock_urlopen:
        with patch("time.sleep") as mock_sleep:
            res = gen.generate("Test prompt")
            assert res == '{"recovered": true}'
            assert mock_urlopen.call_count == 2
            assert mock_sleep.call_count == 1


def test_nim_error_mapping_timeout() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b", max_retries=1, base_delay=0.01)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")) as mock_urlopen:
        with patch("time.sleep"):
            with pytest.raises(LLMTimeoutError) as exc_info:
                gen.generate("Test prompt")

            assert exc_info.value.code == "LLM_TIMEOUT"
            assert exc_info.value.retryable is True


def test_nim_empty_response_handling() -> None:
    gen = NIMGenerator(api_key="mock_key", model="nvidia/nemotron-3-ultra-550b-a55b", max_retries=1, base_delay=0.01)
    empty_resp = _make_mock_http_response({"choices": [{"message": {"content": ""}}]})

    with patch("urllib.request.urlopen", return_value=empty_resp):
        with patch("time.sleep"):
            with pytest.raises(LLMEmptyResponseError) as exc_info:
                gen.generate("Test prompt")

            assert exc_info.value.code == "LLM_EMPTY_RESPONSE"
            assert exc_info.value.retryable is True


# ==============================================================================
# 5. Gateway Integration & InvestigationAgent with NIM
# ==============================================================================

def test_nim_gateway_integration_pacing_and_telemetry() -> None:
    config = LLMConfig(
        provider="nim",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        api_key="nvapi-test-key",
        min_request_interval=0.0,
    )
    gen = NIMGenerator(api_key="nvapi-test-key", model="nvidia/nemotron-3-ultra-550b-a55b")
    gateway = LLMGateway(adapter=gen, config=config)

    mock_payload = {
        "choices": [{"message": {"content": '{"answer": "Integrated response"}'}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    mock_resp = _make_mock_http_response(
        mock_payload,
        headers={"x-ratelimit-remaining-requests": "30", "x-ratelimit-remaining-tokens": "50000"},
    )

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = gateway.generate("Sample prompt")
        assert result == '{"answer": "Integrated response"}'
        assert gateway.successful_calls == 1
        assert gateway.total_input_tokens == 100
        assert gateway.total_output_tokens == 50
        assert gateway.total_tokens == 150
        assert gateway.quota_state.remaining_requests == 30


def test_nim_investigation_agent_end_to_end_mocked() -> None:
    config = LLMConfig(
        provider="nim",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        api_key="nvapi-test-key",
        min_request_interval=0.0,
    )
    gen = NIMGenerator(api_key="nvapi-test-key", model="nvidia/nemotron-3-ultra-550b-a55b")
    gateway = LLMGateway(adapter=gen, config=config)

    # 4 stages of investigation: Planning, Sufficiency, Answer, Verification
    responses = [
        _make_mock_http_response({"choices": [{"message": {"content": '{"intent": "Investigate entity", "objectives": ["find"], "retrieval_queries": ["entity:default_1"]}'}}]}),
        _make_mock_http_response({"choices": [{"message": {"content": '{"sufficient": true, "knowledge_gaps": [], "follow_up_queries": []}'}}]}),
        _make_mock_http_response({"choices": [{"message": {"content": '{"answer": "Grounded answer from NIM Nemotron.", "evidence_ids": ["entity:default_1"], "confidence": 0.95, "knowledge_gaps": []}'}}]}),
        _make_mock_http_response({"choices": [{"message": {"content": '{"verified": true, "answer": "Grounded answer from NIM Nemotron.", "evidence_ids": ["entity:default_1"], "confidence": 0.95, "knowledge_gaps": [], "claims": [{"claim": "fact", "supported": "YES", "evidence_id": "entity:default_1", "reason": "match"}]}'}}]}),
    ]

    class SimpleRetriever:
        def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]:
            return [{"id": "entity:default_1", "text": "PolicyCenter claims entity."}]

    agent = InvestigationAgent(retriever=SimpleRetriever(), generator=gateway)

    with patch("urllib.request.urlopen", side_effect=responses):
        res = agent.investigate("Explain PolicyCenter claims.")
        assert res["status"] == "SUCCESS"
        assert res["confidence"] >= 0.9
        assert res["evidence_ids"] == ["entity:default_1"]
        assert "telemetry" in res
        assert res["telemetry"]["logical_calls"] >= 4
