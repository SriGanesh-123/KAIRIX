"""Comprehensive unit and integration test suite for LLMGateway and Quota Management."""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from knowledge_engineering.investigation_agent.agent import InvestigationAgent
from knowledge_engineering.investigation_agent.config import InvestigationConfig
from knowledge_engineering.llm.config import LLMConfig
from knowledge_engineering.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMEmptyResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from knowledge_engineering.llm.gateway import (
    InvestigationBudget,
    LLMGateway,
    QuotaState,
    TokenBucketPacer,
)
from knowledge_engineering.llm.generator import generate_structured


class MockAdapter:
    """Configurable mock provider adapter."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        provider: str = "mock_provider",
        model: str = "mock_model",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.responses = list(responses or [])
        self.headers = headers or {}
        self.call_count = 0
        self.call_history: list[str] = []

    def call(self, prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any], dict[str, Any]]:
        self.call_count += 1
        self.call_history.append(prompt)
        if not self.responses:
            return '{"status": "ok"}', self.headers, {}
        next_resp = self.responses.pop(0)
        if isinstance(next_resp, Exception):
            raise next_resp
        return str(next_resp), self.headers, {}

    def generate(self, prompt: str) -> str:
        content, _, _ = self.call(prompt)
        return content


class MockRetriever:
    """Mock retriever returning synthetic evidence."""

    def __init__(self, evidence_map: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.evidence_map = evidence_map or {}

    def search(self, query: str, *, limit: int = 5, graph_hops: int = 1) -> list[dict[str, Any]]:
        for k, v in self.evidence_map.items():
            if k in query:
                return v
        return [
            {
                "id": "entity:default_1",
                "entity_name": "TestEntity",
                "text": "Entity definition text.",
            }
        ]


# ==============================================================================
# 1. QuotaState & Header Parsing Tests
# ==============================================================================

def test_gateway_quota_metadata_parsing_from_headers() -> None:
    state = QuotaState()
    headers = {
        "x-ratelimit-remaining-requests": "25",
        "x-ratelimit-remaining-tokens": "15000",
        "x-ratelimit-reset-requests": "2.5s",
        "x-ratelimit-reset-tokens": "500ms",
    }
    state.update_from_headers(headers)

    assert state.remaining_requests == 25
    assert state.remaining_tokens == 15000
    assert state.reset_requests_seconds == 2.5
    assert state.reset_tokens_seconds == 0.5


def test_gateway_missing_quota_headers_graceful_fallback() -> None:
    state = QuotaState()
    state.update_from_headers({})
    assert state.remaining_requests is None
    assert state.remaining_tokens is None
    assert state.reset_requests_seconds is None


def test_gateway_retry_after_header_parsed() -> None:
    state = QuotaState()
    state.set_cooldown(3.0)
    assert state.get_cooldown_remaining() > 2.0
    assert state.get_cooldown_remaining() <= 3.0


# ==============================================================================
# 2. TokenBucketPacer Tests
# ==============================================================================

def test_token_bucket_pacer_enforces_spacing() -> None:
    pacer = TokenBucketPacer(min_request_interval=0.05)
    t0 = time.time()
    pacer.acquire()
    pacer.acquire()
    elapsed = time.time() - t0
    assert elapsed >= 0.04


def test_token_bucket_pacer_respects_active_cooldown() -> None:
    pacer = TokenBucketPacer(min_request_interval=0.0)
    state = QuotaState()
    state.set_cooldown(0.08)

    t0 = time.time()
    waited = pacer.acquire(state)
    elapsed = time.time() - t0

    assert waited >= 0.05
    assert elapsed >= 0.05


# ==============================================================================
# 3. InvestigationBudget Tests
# ==============================================================================

def test_investigation_call_budget_enforcement() -> None:
    budget = InvestigationBudget(max_calls=3)
    assert budget.record_call("planning") is True
    assert budget.record_call("sufficiency") is True
    assert budget.record_call("answer") is True
    assert budget.record_call("verifier") is False  # Budget exhausted!
    assert budget.remaining() == 0


def test_investigation_repair_budget_accounting() -> None:
    budget = InvestigationBudget(max_calls=5)
    budget.record_call("planning")
    budget.record_call("planning")  # Schema repair call
    budget.record_call("answer")

    data = budget.to_dict()
    assert data["calls_made"] == 3
    assert data["remaining"] == 2
    assert data["stage_breakdown"]["planning"] == 2
    assert data["stage_breakdown"]["answer"] == 1


# ==============================================================================
# 4. Gateway Retry, Backoff, and Jitter Tests
# ==============================================================================

def test_gateway_429_rate_limit_handled() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=1,
        base_delay=0.01,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        RuntimeError("429 rate limit exceeded"),
        '{"answer": "recovered"}',
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    res = gateway.generate("test prompt")
    assert res == '{"answer": "recovered"}'
    assert adapter.call_count == 2
    assert gateway.rate_limits_encountered == 1


def test_gateway_retry_after_absent_uses_exponential_backoff() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=2,
        base_delay=0.02,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        RuntimeError("429 rate limit exceeded"),
        RuntimeError("429 rate limit exceeded"),
        '{"status": "ok"}',
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    t0 = time.time()
    res = gateway.generate("test")
    elapsed = time.time() - t0

    assert res == '{"status": "ok"}'
    assert adapter.call_count == 3
    # 0.02 + 0.04 = ~0.06s minimum
    assert elapsed >= 0.04


def test_gateway_exponential_backoff_respects_max_delay() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=1,
        base_delay=100.0,
        max_delay=0.05,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        RuntimeError("500 internal server error"),
        '{"status": "ok"}',
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    t0 = time.time()
    res = gateway.generate("test")
    elapsed = time.time() - t0

    assert res == '{"status": "ok"}'
    # Should not wait 100s, but clamp at max_delay ~0.05s
    assert elapsed < 1.0


def test_gateway_fast_fail_on_auth_error() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="bad_key",
        max_retries=3,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        RuntimeError("401 unauthorized: invalid api key"),
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    with pytest.raises(LLMAuthError) as exc_info:
        gateway.generate("test")

    assert exc_info.value.retryable is False
    assert adapter.call_count == 1  # Fast-failed without retries


def test_gateway_timeout_retry_and_recovery() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=2,
        base_delay=0.01,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        TimeoutError("Request timed out"),
        '{"answer": "recovered after timeout"}',
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    res = gateway.generate("test")
    assert res == '{"answer": "recovered after timeout"}'
    assert adapter.call_count == 2


def test_gateway_server_error_retry_and_exhaustion() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=2,
        base_delay=0.01,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        RuntimeError("503 Service Unavailable"),
        RuntimeError("503 Service Unavailable"),
        RuntimeError("503 Service Unavailable"),
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    with pytest.raises(LLMServerError):
        gateway.generate("test")

    assert adapter.call_count == 3


def test_gateway_retry_exhaustion_raises_typed_error() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=1,
        base_delay=0.01,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        RuntimeError("429 rate limit exceeded"),
        RuntimeError("429 rate limit exceeded"),
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    with pytest.raises(LLMRateLimitError):
        gateway.generate("test")


def test_gateway_budget_exhaustion_raises_bad_request() -> None:
    config = LLMConfig(provider="mock", model="mock", api_key="key", min_request_interval=0.0)
    adapter = MockAdapter(responses=['{"ok": true}'])
    gateway = LLMGateway(adapter=adapter, config=config)

    budget = InvestigationBudget(max_calls=1)
    gateway.generate("call 1", budget=budget)

    with pytest.raises(LLMBadRequestError) as exc_info:
        gateway.generate("call 2", budget=budget)

    assert "budget exceeded" in str(exc_info.value).lower()


# ==============================================================================
# 5. Concurrency & Thread-Safety Tests
# ==============================================================================

def test_gateway_concurrency_limit_with_semaphore() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_concurrent_requests=1,
        min_request_interval=0.0,
    )
    active_concurrent = 0
    max_observed_concurrent = 0
    lock = threading.Lock()

    class ConcurrencyTrackingAdapter:
        provider = "mock"
        model = "mock"

        def call(self, prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any], dict[str, Any]]:
            nonlocal active_concurrent, max_observed_concurrent
            with lock:
                active_concurrent += 1
                if active_concurrent > max_observed_concurrent:
                    max_observed_concurrent = active_concurrent
            time.sleep(0.02)
            with lock:
                active_concurrent -= 1
            return '{"status": "ok"}', {}, {}

    gateway = LLMGateway(adapter=ConcurrencyTrackingAdapter(), config=config)

    threads = [threading.Thread(target=gateway.generate, args=(f"prompt {i}",)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_observed_concurrent == 1


def test_gateway_multiple_simultaneous_investigations() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_concurrent_requests=2,
        min_request_interval=0.0,
    )
    adapter = MockAdapter()
    gateway = LLMGateway(adapter=adapter, config=config)

    budgets = [InvestigationBudget(max_calls=3) for _ in range(4)]
    errors: list[Exception] = []

    def run_investigation(b: InvestigationBudget) -> None:
        try:
            for _ in range(3):
                gateway.generate("test prompt", budget=b)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_investigation, args=(b,)) for b in budgets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert all(b.remaining() == 0 for b in budgets)
    assert gateway.total_calls == 12


# ==============================================================================
# 6. End-to-End Investigation with Gateway Resilience Tests
# ==============================================================================

def test_gateway_successful_investigation_after_transient_429() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=2,
        base_delay=0.01,
        min_request_interval=0.0,
    )
    # Plan, Sufficiency, Transient 429 on Answer, Answer recovered, Verifier
    adapter = MockAdapter(responses=[
        '{"intent": "Investigate entity", "objectives": ["find"], "retrieval_queries": ["entity:default_1"]}',
        '{"sufficient": true, "knowledge_gaps": [], "follow_up_queries": []}',
        RuntimeError("429 Too Many Requests: Rate limit reached. Try again in 0.05s"),
        '{"answer": "Grounded answer text.", "evidence_ids": ["entity:default_1"], "confidence": 0.9, "knowledge_gaps": []}',
        '{"verified": true, "answer": "Grounded answer text.", "evidence_ids": ["entity:default_1"], "confidence": 0.9, "knowledge_gaps": [], "claims": [{"claim": "fact", "supported": "YES", "evidence_id": "entity:default_1", "reason": "match"}]}',
    ])
    gateway = LLMGateway(adapter=adapter, config=config)
    retriever = MockRetriever()

    agent = InvestigationAgent(retriever=retriever, generator=gateway)
    res = agent.investigate("Explain entity connections.")

    assert res["status"] == "SUCCESS"
    assert res["confidence"] >= 0.8
    assert res["evidence_ids"] == ["entity:default_1"]
    assert gateway.rate_limits_encountered == 1


def test_gateway_controlled_failure_after_exhausted_quota() -> None:
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=1,
        base_delay=0.01,
        min_request_interval=0.0,
    )
    # Plan, Sufficiency, Rate limit on Answer (attempt 1 and attempt 2 exhausted)
    adapter = MockAdapter(responses=[
        '{"intent": "Investigate entity", "objectives": ["find"], "retrieval_queries": ["entity:default_1"]}',
        '{"sufficient": true, "knowledge_gaps": [], "follow_up_queries": []}',
        RuntimeError("429 Rate limit exhausted"),
        RuntimeError("429 Rate limit exhausted"),
    ])
    gateway = LLMGateway(adapter=adapter, config=config)
    retriever = MockRetriever()

    agent = InvestigationAgent(retriever=retriever, generator=gateway)
    res = agent.investigate("Explain entity connections.")

    assert res["status"] == "RATE_LIMITED"
    assert res["confidence"] == 0.0
    assert "rate limit" in res["answer"].lower()


# ==============================================================================
# 7. Mathematical Sliding Window RPM Limiter & Dual Budget Tests
# ==============================================================================

def test_token_bucket_pacer_sliding_window_rpm_limit() -> None:
    """Ensure rolling 60-second window enforces RPM limit strictly."""
    pacer = TokenBucketPacer(min_request_interval=0.0, rpm_limit=3, max_delay=60.0)
    now = time.time()
    # Inject 3 timestamps within the last 10 seconds
    pacer._request_timestamps = [now - 10.0, now - 5.0, now - 1.0]

    # The 4th request must wait for the oldest timestamp (now - 10.0) to exit the 60s window:
    # 60 - 10 = ~50 seconds wait
    with patch("time.sleep") as mock_sleep:
        waited = pacer.acquire()
        assert waited >= 49.0
        assert waited <= 51.0
        mock_sleep.assert_called_once()


def test_token_bucket_pacer_sliding_window_eviction() -> None:
    """Ensure timestamps older than 60s are evicted, allowing immediate execution."""
    pacer = TokenBucketPacer(min_request_interval=0.0, rpm_limit=3, max_delay=60.0)
    now = time.time()
    # Inject 3 timestamps older than 60 seconds
    pacer._request_timestamps = [now - 65.0, now - 62.0, now - 61.0]

    with patch("time.sleep") as mock_sleep:
        waited = pacer.acquire()
        assert waited == 0.0
        mock_sleep.assert_not_called()
        # Old timestamps evicted, only the new one remains
        assert len(pacer._request_timestamps) == 1


def test_dual_budget_logical_vs_provider_http_attempts() -> None:
    """Verify separate tracking and enforcement of logical calls vs physical HTTP attempts."""
    budget = InvestigationBudget(
        max_calls=5,
        max_provider_attempts=8,
        max_repair_retries=2,
    )

    # 1 logical call with 3 transport attempts (1 initial + 2 retries)
    assert budget.record_call("planning") is True
    assert budget.record_provider_attempt() is True
    budget.record_failure()
    budget.record_retry()

    assert budget.record_provider_attempt() is True
    budget.record_failure()
    budget.record_retry()

    assert budget.record_provider_attempt() is True
    budget.record_success()

    assert budget.calls_made == 1
    assert budget.provider_http_attempts == 3
    assert budget.retry_attempts == 2
    assert budget.successful_provider_requests == 1
    assert budget.failed_provider_requests == 2
    assert budget.remaining() == 4
    assert budget.remaining_provider_attempts() == 5


def test_provider_attempt_budget_exhaustion_in_gateway() -> None:
    """Ensure exceeding max_provider_attempts raises LLMRateLimitError."""
    config = LLMConfig(
        provider="mock",
        model="mock",
        api_key="key",
        max_retries=3,
        base_delay=0.01,
        min_request_interval=0.0,
    )
    adapter = MockAdapter(responses=[
        RuntimeError("500 server error"),
        RuntimeError("500 server error"),
        RuntimeError("500 server error"),
    ])
    gateway = LLMGateway(adapter=adapter, config=config)

    # Budget allows only 2 physical HTTP attempts
    budget = InvestigationBudget(max_calls=5, max_provider_attempts=2)

    with pytest.raises(LLMRateLimitError) as exc_info:
        gateway.generate("test prompt", budget=budget)

    assert "attempt budget exceeded" in str(exc_info.value).lower()
    assert budget.provider_http_attempts == 2


def test_gateway_token_usage_telemetry_aggregation() -> None:
    """Ensure provider token usage is aggregated into budget and telemetry."""
    config = LLMConfig(provider="mock", model="mock", api_key="key", min_request_interval=0.0)

    class UsageAdapter:
        provider = "mock"
        model = "mock"

        def call(self, prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any], dict[str, Any]]:
            return (
                '{"answer": "ok"}',
                {"x-ratelimit-remaining-requests": "20"},
                {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
            )

    gateway = LLMGateway(adapter=UsageAdapter(), config=config)
    budget = InvestigationBudget(max_calls=5)

    gateway.generate("call 1", budget=budget)
    gateway.generate("call 2", budget=budget)

    telemetry = budget.to_telemetry(gateway.quota_state)
    assert telemetry["logical_calls"] == 2
    assert telemetry["provider_http_attempts"] == 2
    assert telemetry["input_tokens"] == 300
    assert telemetry["output_tokens"] == 100
    assert telemetry["total_tokens"] == 400
    assert telemetry["quota_remaining_requests"] == 20


def test_investigation_result_contains_complete_telemetry() -> None:
    """Ensure agent.investigate() returns structured, sanitized telemetry."""
    config = LLMConfig(provider="mock", model="mock", api_key="key", min_request_interval=0.0)
    adapter = MockAdapter(responses=[
        '{"intent": "Investigate entity", "objectives": ["find"], "retrieval_queries": ["entity:default_1"]}',
        '{"sufficient": true, "knowledge_gaps": [], "follow_up_queries": []}',
        '{"answer": "Grounded answer text.", "evidence_ids": ["entity:default_1"], "confidence": 0.9, "knowledge_gaps": []}',
        '{"verified": true, "answer": "Grounded answer text.", "evidence_ids": ["entity:default_1"], "confidence": 0.9, "knowledge_gaps": [], "claims": [{"claim": "fact", "supported": "YES", "evidence_id": "entity:default_1", "reason": "match"}]}',
    ])
    gateway = LLMGateway(adapter=adapter, config=config)
    retriever = MockRetriever()

    agent = InvestigationAgent(retriever=retriever, generator=gateway)
    res = agent.investigate("Explain entity.")

    assert "telemetry" in res
    tel = res["telemetry"]
    assert tel["logical_calls"] >= 4
    assert tel["provider_http_attempts"] >= 4
    assert tel["successful_provider_requests"] >= 4
    assert tel["failed_provider_requests"] == 0
    assert tel["repair_attempts"] == 0
    assert "planning_calls" in tel
    assert "sufficiency_calls" in tel
    assert "answer_calls" in tel
    assert "verification_calls" in tel


def test_llm_config_validation_rules() -> None:
    """Ensure LLMConfig validates parameter boundaries."""
    with pytest.raises(ValueError, match="rpm_limit"):
        LLMConfig(rpm_limit=0)

    with pytest.raises(ValueError, match="max_retries"):
        LLMConfig(max_retries=-1)

    with pytest.raises(ValueError, match="base_delay"):
        LLMConfig(base_delay=0.0)

    with pytest.raises(ValueError, match="max_delay"):
        LLMConfig(max_delay=-1.0)

    with pytest.raises(ValueError, match="max_concurrent_requests"):
        LLMConfig(max_concurrent_requests=0)

    with pytest.raises(ValueError, match="max_calls_per_investigation"):
        LLMConfig(max_calls_per_investigation=0)

    with pytest.raises(ValueError, match="max_provider_attempts_per_investigation"):
        LLMConfig(max_provider_attempts_per_investigation=0)

