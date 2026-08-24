"""Centralized, provider-neutral LLM Gateway with quota tracking, burst pacing, and concurrency control."""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

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
    extract_retry_after,
)

_STRUCTURED_OUTPUT_SYSTEM_PROMPT = (
    "Return valid JSON only. Follow the exact JSON schema requested by the current user prompt. "
    "Do not add markdown fences, preamble, or explanatory text outside the JSON object. "
    "The schema differs between investigation stages, so use the schema explicitly requested."
)


@dataclass
class QuotaState:
    """Live snapshot of provider quota and rate-limit headers."""
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_requests_seconds: float | None = None
    reset_tokens_seconds: float | None = None
    last_updated: float = 0.0
    cooldown_until: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def update_from_headers(self, headers: dict[str, Any] | None) -> None:
        if not headers:
            return
        with self._lock:
            self.last_updated = time.time()
            # Normalize keys to lowercase
            norm = {str(k).lower(): str(v) for k, v in headers.items()}

            # Parse remaining requests
            rem_req = norm.get("x-ratelimit-remaining-requests") or norm.get("ratelimit-remaining-requests")
            if rem_req is not None:
                try:
                    self.remaining_requests = int(rem_req)
                except ValueError:
                    pass

            # Parse remaining tokens
            rem_tok = norm.get("x-ratelimit-remaining-tokens") or norm.get("ratelimit-remaining-tokens")
            if rem_tok is not None:
                try:
                    self.remaining_tokens = int(rem_tok)
                except ValueError:
                    pass

            # Parse reset requests duration
            res_req = norm.get("x-ratelimit-reset-requests") or norm.get("ratelimit-reset-requests")
            if res_req is not None:
                parsed = self._parse_duration(res_req)
                if parsed is not None:
                    self.reset_requests_seconds = parsed

            # Parse reset tokens duration
            res_tok = norm.get("x-ratelimit-reset-tokens") or norm.get("ratelimit-reset-tokens")
            if res_tok is not None:
                parsed = self._parse_duration(res_tok)
                if parsed is not None:
                    self.reset_tokens_seconds = parsed

    @staticmethod
    def _parse_duration(raw: str) -> float | None:
        text = str(raw).strip()
        if not text:
            return None
        if text.endswith("ms"):
            try:
                return float(text[:-2]) / 1000.0
            except ValueError:
                return None
        if "m" in text and text.endswith("s"):
            parts = text[:-1].split("m")
            if len(parts) == 2:
                try:
                    mins = float(parts[0])
                    secs = float(parts[1]) if parts[1] else 0.0
                    return mins * 60.0 + secs
                except ValueError:
                    return None
        if text.endswith("s"):
            text = text[:-1]
        try:
            return float(text)
        except ValueError:
            pass
        return None

    def set_cooldown(self, seconds: float, max_seconds: float = 60.0) -> None:
        with self._lock:
            now = time.time()
            capped = min(max_seconds, max(0.0, seconds))
            self.cooldown_until = max(self.cooldown_until, now + capped)

    def get_cooldown_remaining(self) -> float:
        with self._lock:
            now = time.time()
            return max(0.0, self.cooldown_until - now)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            return {
                "remaining_requests": self.remaining_requests,
                "remaining_tokens": self.remaining_tokens,
                "reset_requests_seconds": self.reset_requests_seconds,
                "reset_tokens_seconds": self.reset_tokens_seconds,
                "cooldown_remaining": max(0.0, self.cooldown_until - now),
            }


class TokenBucketPacer:
    """Thread-safe rate limiter enforcing rolling 60s RPM limits, spacing, and cooldowns."""

    def __init__(
        self,
        min_request_interval: float = 2.0,
        rpm_limit: int = 30,
        max_delay: float = 60.0,
    ) -> None:
        self.min_request_interval = max(0.0, min_request_interval)
        self.rpm_limit = max(1, rpm_limit)
        self.max_delay = max(0.01, max_delay)
        self._request_timestamps: list[float] = []
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def acquire(self, quota_state: QuotaState | None = None) -> float:
        """Wait if necessary before allowing the next request through. Returns wait time in seconds."""
        wait_time = 0.0
        with self._lock:
            now = time.time()

            # Clean expired timestamps older than 60 seconds
            self._request_timestamps = [t for t in self._request_timestamps if now - t < 60.0]

            # 1. Enforce strict rolling 60-second RPM limit
            if len(self._request_timestamps) >= self.rpm_limit:
                oldest_timestamp = self._request_timestamps[0]
                rpm_wait = max(0.0, (oldest_timestamp + 60.0) - now)
                wait_time = max(wait_time, rpm_wait)

            # 2. Enforce inter-request spacing (burst smoothing)
            if self.min_request_interval > 0:
                elapsed = now - self._last_request_time
                if elapsed < self.min_request_interval:
                    spacing_wait = self.min_request_interval - elapsed
                    wait_time = max(wait_time, spacing_wait)

            # 3. Respect active cooldown (e.g. from prior 429), capped at self.max_delay
            if quota_state is not None:
                cooldown = quota_state.get_cooldown_remaining()
                if cooldown > 0.0:
                    wait_time = max(wait_time, min(self.max_delay, cooldown))

                # If remaining tokens is completely exhausted (0), pause briefly for token reset
                if quota_state.remaining_tokens is not None and quota_state.remaining_tokens <= 0:
                    if quota_state.reset_tokens_seconds and quota_state.reset_tokens_seconds > 0:
                        wait_time = max(wait_time, min(3.0, quota_state.reset_tokens_seconds))

            # Final cap on scheduled wait time
            wait_time = min(self.max_delay, wait_time)
            scheduled_time = now + wait_time
            self._request_timestamps.append(scheduled_time)
            self._last_request_time = scheduled_time

        if wait_time > 0.0:
            time.sleep(wait_time)

        return wait_time

    def reset(self) -> None:
        with self._lock:
            self._request_timestamps.clear()
            self._last_request_time = 0.0


@dataclass
class InvestigationBudget:
    """Tracks and limits logical LLM calls, provider HTTP attempts, tokens, and repairs."""
    # Logical LLM Call budget
    max_calls: int = 12
    calls_made: int = 0

    # Provider HTTP Attempt budget
    max_provider_attempts: int = 20
    provider_http_attempts: int = 0
    successful_provider_requests: int = 0
    failed_provider_requests: int = 0
    retry_attempts: int = 0

    # Schema Repair budget
    max_repair_retries: int = 2
    repairs_made: int = 0

    # Events and Telemetry
    rate_limit_events: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_wait_seconds: float = 0.0
    stage_breakdown: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record_call(self, stage: str = "general") -> bool:
        """Record a logical LLM call. Returns True if within budget, False if budget exceeded."""
        with self._lock:
            if self.calls_made >= self.max_calls:
                return False
            self.calls_made += 1
            self.stage_breakdown[stage] = self.stage_breakdown.get(stage, 0) + 1
            return True

    def record_provider_attempt(self) -> bool:
        """Record a physical provider HTTP attempt. Returns True if within budget, False if exceeded."""
        with self._lock:
            if self.provider_http_attempts >= self.max_provider_attempts:
                return False
            self.provider_http_attempts += 1
            return True

    def record_success(self) -> None:
        with self._lock:
            self.successful_provider_requests += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failed_provider_requests += 1

    def record_retry(self) -> None:
        with self._lock:
            self.retry_attempts += 1

    def record_repair(self) -> bool:
        """Record a schema repair retry attempt. Returns True if within budget, False if exceeded."""
        with self._lock:
            if self.repairs_made >= self.max_repair_retries:
                return False
            self.repairs_made += 1
            return True

    def record_rate_limit(self) -> None:
        with self._lock:
            self.rate_limit_events += 1

    def record_tokens(self, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        with self._lock:
            self.input_tokens += max(0, input_tokens)
            self.output_tokens += max(0, output_tokens)
            self.total_tokens += max(0, total_tokens)

    def record_wait(self, seconds: float) -> None:
        with self._lock:
            self.total_wait_seconds += max(0.0, seconds)

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_calls - self.calls_made)

    def remaining_provider_attempts(self) -> int:
        with self._lock:
            return max(0, self.max_provider_attempts - self.provider_http_attempts)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "calls_made": self.calls_made,
                "max_provider_attempts": self.max_provider_attempts,
                "provider_http_attempts": self.provider_http_attempts,
                "successful_provider_requests": self.successful_provider_requests,
                "failed_provider_requests": self.failed_provider_requests,
                "retry_attempts": self.retry_attempts,
                "max_repair_retries": self.max_repair_retries,
                "repairs_made": self.repairs_made,
                "rate_limit_events": self.rate_limit_events,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "total_wait_seconds": round(self.total_wait_seconds, 3),
                "remaining": max(0, self.max_calls - self.calls_made),
                "remaining_calls": max(0, self.max_calls - self.calls_made),
                "remaining_provider_attempts": max(0, self.max_provider_attempts - self.provider_http_attempts),
                "stage_breakdown": dict(self.stage_breakdown),
            }

    def to_telemetry(self, quota_state: QuotaState | None = None) -> dict[str, Any]:
        """Produce standardized investigation telemetry with zero secret leakage."""
        with self._lock:
            q_dict = quota_state.to_dict() if quota_state is not None else {}
            return {
                "logical_calls": self.calls_made,
                "provider_http_attempts": self.provider_http_attempts,
                "successful_provider_requests": self.successful_provider_requests,
                "failed_provider_requests": self.failed_provider_requests,
                "retry_attempts": self.retry_attempts,
                "repair_attempts": self.repairs_made,
                "rate_limit_events": self.rate_limit_events,
                "planning_calls": self.stage_breakdown.get("planning", 0),
                "sufficiency_calls": self.stage_breakdown.get("sufficiency", 0),
                "answer_calls": self.stage_breakdown.get("answer", 0),
                "verification_calls": self.stage_breakdown.get("verification", 0),
                "total_wait_seconds": round(self.total_wait_seconds, 3),
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "quota_remaining_requests": q_dict.get("remaining_requests"),
                "quota_remaining_tokens": q_dict.get("remaining_tokens"),
                "reset_requests_seconds": q_dict.get("reset_requests_seconds"),
                "reset_tokens_seconds": q_dict.get("reset_tokens_seconds"),
            }


class ProviderAdapter(Protocol):
    """Protocol for provider-specific LLM adapters."""
    provider: str
    model: str

    def call(self, prompt: str, system_prompt: str = "") -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Execute the raw API call and return (content_text, headers_dict, usage_dict)."""


class LLMGateway:
    """Central provider-neutral gateway orchestrating pacing, quota tracking, and retries."""

    def __init__(
        self,
        adapter: ProviderAdapter | Any,
        config: LLMConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.provider = getattr(adapter, "provider", "unknown")
        self.model = getattr(adapter, "model", "unknown")
        self.config = config or LLMConfig(
            provider=self.provider,
            model=self.model,
            api_key="gateway_managed",
        )

        self.quota_state = QuotaState()
        self.pacer = TokenBucketPacer(
            min_request_interval=self.config.min_request_interval,
            rpm_limit=self.config.rpm_limit,
            max_delay=self.config.max_delay,
        )
        self.semaphore = threading.Semaphore(self.config.max_concurrent_requests)

        self._stats_lock = threading.Lock()
        self.total_calls: int = 0
        self.successful_calls: int = 0
        self.failed_calls: int = 0
        self.rate_limits_encountered: int = 0
        self.total_wait_seconds: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_tokens: int = 0

    def generate(
        self,
        prompt: str,
        stage: str = "general",
        budget: InvestigationBudget | None = None,
    ) -> str:
        """Execute text generation through the gateway with dual budgeting, pacing, and retries."""
        if budget is not None and not budget.record_call(stage):
            raise LLMBadRequestError(
                f"Investigation LLM call budget exceeded (limit: {budget.max_calls} calls)",
                provider=self.provider,
                model=self.model,
            )

        with self._stats_lock:
            self.total_calls += 1

        last_error: Exception | None = None

        with self.semaphore:
            for attempt in range(self.config.max_retries + 1):
                # Check provider HTTP attempt budget
                if budget is not None and not budget.record_provider_attempt():
                    raise LLMRateLimitError(
                        f"Investigation provider HTTP attempt budget exceeded (limit: {budget.max_provider_attempts} attempts)",
                        provider=self.provider,
                        model=self.model,
                    )

                # Apply token-bucket pacing and cooldown check
                waited = self.pacer.acquire(self.quota_state if self.config.enable_quota_tracking else None)
                if budget is not None:
                    budget.record_wait(waited)
                with self._stats_lock:
                    self.total_wait_seconds += waited

                try:
                    # Execute adapter call
                    headers: dict[str, Any] = {}
                    usage: dict[str, Any] = {}
                    if hasattr(self.adapter, "call"):
                        content, headers, usage = self.adapter.call(prompt, _STRUCTURED_OUTPUT_SYSTEM_PROMPT)
                        if self.config.enable_quota_tracking and headers:
                            self.quota_state.update_from_headers(headers)
                    else:
                        # Fallback for plain generator objects
                        if hasattr(self.adapter, "generate") and "stage" in getattr(self.adapter.generate, "__code__", object()).co_varnames:
                            content = self.adapter.generate(prompt, stage=stage, budget=budget)
                        else:
                            content = self.adapter.generate(prompt)

                    if not content:
                        raise LLMEmptyResponseError(
                            f"{self.provider} returned an empty response",
                            provider=self.provider,
                            model=self.model,
                        )

                    # Ingest token usage if present
                    if usage:
                        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                        tot = int(usage.get("total_tokens") or (inp + out))
                        if budget is not None:
                            budget.record_tokens(inp, out, tot)
                        with self._stats_lock:
                            self.total_input_tokens += inp
                            self.total_output_tokens += out
                            self.total_tokens += tot

                    if budget is not None:
                        budget.record_success()
                    with self._stats_lock:
                        self.successful_calls += 1
                    return content

                except Exception as exc:
                    classified = classify_exception(exc, provider=self.provider, model=self.model)
                    last_error = classified

                    if budget is not None:
                        budget.record_failure()
                    with self._stats_lock:
                        self.failed_calls += 1

                    # Fast-fail on non-retryable errors (401 Auth, 400 Bad Request)
                    if not classified.retryable:
                        raise classified from exc

                    if isinstance(classified, LLMRateLimitError):
                        if budget is not None:
                            budget.record_rate_limit()
                        with self._stats_lock:
                            self.rate_limits_encountered += 1
                        cooldown_delay = classified.retry_after or (self.config.base_delay * (2 ** attempt))
                        self.quota_state.set_cooldown(cooldown_delay, max_seconds=self.config.max_delay)

                    if attempt < self.config.max_retries:
                        if budget is not None:
                            budget.record_retry()
                        if isinstance(classified, LLMRateLimitError) and classified.retry_after is not None:
                            delay = min(self.config.max_delay, max(classified.retry_after, self.config.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
                        else:
                            delay = min(self.config.max_delay, (self.config.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
                        logger.warning(f"LLMGateway: {self.provider}/{self.model} [{stage}] attempt {attempt+1} failed ({classified.code}: {classified.message}). Backing off for {delay:.2f}s...")
                        time.sleep(delay)

        if isinstance(last_error, LLMError):
            raise last_error
        raise RuntimeError(f"{self.provider} generation failed ({self.model}): {last_error}") from last_error

    def get_telemetry(self) -> dict[str, Any]:
        """Return a live telemetry snapshot with secrets safely masked."""
        with self._stats_lock:
            return {
                "provider": self.provider,
                "model": self.model,
                "total_calls": self.total_calls,
                "successful_calls": self.successful_calls,
                "failed_calls": self.failed_calls,
                "rate_limits_encountered": self.rate_limits_encountered,
                "total_wait_seconds": round(self.total_wait_seconds, 3),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "quota_state": self.quota_state.to_dict(),
            }

