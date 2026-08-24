"""Centralized, provider-neutral LLM Gateway with quota tracking, burst pacing, and concurrency control."""
from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

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

    def set_cooldown(self, seconds: float) -> None:
        with self._lock:
            now = time.time()
            self.cooldown_until = max(self.cooldown_until, now + max(0.0, seconds))

    def get_cooldown_remaining(self) -> float:
        with self._lock:
            now = time.time()
            return max(0.0, self.cooldown_until - now)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "remaining_requests": self.remaining_requests,
                "remaining_tokens": self.remaining_tokens,
                "reset_requests_seconds": self.reset_requests_seconds,
                "reset_tokens_seconds": self.reset_tokens_seconds,
                "cooldown_remaining": self.get_cooldown_remaining(),
            }


class TokenBucketPacer:
    """Thread-safe pacer to smooth outbound request bursts and honor quotas."""

    def __init__(self, min_request_interval: float = 1.5, rpm_limit: int = 30) -> None:
        self.min_request_interval = max(0.0, min_request_interval)
        self.rpm_limit = max(1, rpm_limit)
        self._last_request_time: float = 0.0
        self._lock = threading.Lock()

    def acquire(self, quota_state: QuotaState | None = None) -> float:
        """Wait if necessary before allowing the next request through. Returns wait time in seconds."""
        wait_time = 0.0
        with self._lock:
            now = time.time()

            # 1. Respect active cooldown (e.g. from prior 429)
            if quota_state is not None:
                cooldown = quota_state.get_cooldown_remaining()
                if cooldown > 0.0:
                    wait_time = max(wait_time, cooldown)

                # If remaining tokens is completely exhausted (0), pause briefly for token reset
                if quota_state.remaining_tokens is not None and quota_state.remaining_tokens <= 0:
                    if quota_state.reset_tokens_seconds and quota_state.reset_tokens_seconds > 0:
                        wait_time = max(wait_time, min(3.0, quota_state.reset_tokens_seconds))

            # 2. Enforce minimum request interval (burst smoothing)
            if self.min_request_interval > 0:
                elapsed = now - self._last_request_time
                if elapsed < self.min_request_interval:
                    spacing_wait = self.min_request_interval - elapsed
                    wait_time = max(wait_time, spacing_wait)

            self._last_request_time = now + wait_time

        if wait_time > 0.0:
            time.sleep(wait_time)

        return wait_time


@dataclass
class InvestigationBudget:
    """Tracks and limits LLM calls during an investigation lifecycle."""
    max_calls: int = 10
    calls_made: int = 0
    stage_breakdown: dict[str, int] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record_call(self, stage: str = "general") -> bool:
        """Record an LLM call. Returns True if within budget, False if budget exceeded."""
        with self._lock:
            if self.calls_made >= self.max_calls:
                return False
            self.calls_made += 1
            self.stage_breakdown[stage] = self.stage_breakdown.get(stage, 0) + 1
            return True

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_calls - self.calls_made)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_calls": self.max_calls,
                "calls_made": self.calls_made,
                "remaining": max(0, self.max_calls - self.calls_made),
                "stage_breakdown": dict(self.stage_breakdown),
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
        )
        self.semaphore = threading.Semaphore(self.config.max_concurrent_requests)

        self._stats_lock = threading.Lock()
        self.total_calls: int = 0
        self.successful_calls: int = 0
        self.failed_calls: int = 0
        self.rate_limits_encountered: int = 0
        self.total_wait_seconds: float = 0.0

    def generate(
        self,
        prompt: str,
        stage: str = "general",
        budget: InvestigationBudget | None = None,
    ) -> str:
        """Execute text generation through the gateway with budget, pacing, and retries."""
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
                # Apply token-bucket pacing and cooldown check
                waited = self.pacer.acquire(self.quota_state if self.config.enable_quota_tracking else None)
                with self._stats_lock:
                    self.total_wait_seconds += waited

                try:
                    # Execute adapter call
                    if hasattr(self.adapter, "call"):
                        content, headers, usage = self.adapter.call(prompt, _STRUCTURED_OUTPUT_SYSTEM_PROMPT)
                        if self.config.enable_quota_tracking and headers:
                            self.quota_state.update_from_headers(headers)
                    else:
                        # Fallback for plain generator objects
                        content = self.adapter.generate(prompt)

                    if not content:
                        raise LLMEmptyResponseError(
                            f"{self.provider} returned an empty response",
                            provider=self.provider,
                            model=self.model,
                        )

                    with self._stats_lock:
                        self.successful_calls += 1
                    return content

                except Exception as exc:
                    classified = classify_exception(exc, provider=self.provider, model=self.model)
                    last_error = classified

                    with self._stats_lock:
                        self.failed_calls += 1

                    # Fast-fail on non-retryable errors (401 Auth, 400 Bad Request)
                    if not classified.retryable:
                        raise classified from exc

                    if isinstance(classified, LLMRateLimitError):
                        with self._stats_lock:
                            self.rate_limits_encountered += 1
                        cooldown_delay = classified.retry_after or (self.config.base_delay * (2 ** attempt))
                        self.quota_state.set_cooldown(cooldown_delay)

                    if attempt < self.config.max_retries:
                        if isinstance(classified, LLMRateLimitError) and classified.retry_after is not None:
                            delay = min(self.config.max_delay, max(classified.retry_after, self.config.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
                        else:
                            delay = min(self.config.max_delay, (self.config.base_delay * (2 ** attempt)) + random.uniform(0.1, 0.4))
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
                "quota_state": self.quota_state.to_dict(),
            }
