"""Central LLM configuration shared by all agents."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "groq"
    model: str = "default-model"
    api_key: str = "default-key"
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    timeout: float = 30.0
    rpm_limit: int = 30
    min_request_interval: float = 2.0
    max_concurrent_requests: int = 2
    max_calls_per_investigation: int = 12
    max_provider_attempts_per_investigation: int = 20
    max_provider_attempts_per_generation: int = 4
    max_repair_retries: int = 2
    enable_quota_tracking: bool = True

    def __post_init__(self) -> None:
        if self.rpm_limit < 1:
            raise ValueError(f"rpm_limit must be >= 1, got {self.rpm_limit}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.base_delay <= 0.0:
            raise ValueError(f"base_delay must be > 0.0, got {self.base_delay}")
        if self.max_delay < 0.0:
            raise ValueError(f"max_delay must be >= 0.0, got {self.max_delay}")
        if self.max_concurrent_requests < 1:
            raise ValueError(f"max_concurrent_requests must be >= 1, got {self.max_concurrent_requests}")
        if self.max_calls_per_investigation < 1:
            raise ValueError(f"max_calls_per_investigation must be >= 1, got {self.max_calls_per_investigation}")
        if self.max_provider_attempts_per_investigation < 1:
            raise ValueError(f"max_provider_attempts_per_investigation must be >= 1, got {self.max_provider_attempts_per_investigation}")
        if self.max_repair_retries < 0:
            raise ValueError(f"max_repair_retries must be >= 0, got {self.max_repair_retries}")

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key and provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
        elif not api_key and provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not provider or not model or not api_key:
            return None

        def _get_int(key: str, default: int) -> int:
            val = os.getenv(key)
            if val is not None:
                try:
                    return int(val.strip())
                except ValueError:
                    pass
            return default

        def _get_float(key: str, default: float) -> float:
            val = os.getenv(key)
            if val is not None:
                try:
                    return float(val.strip())
                except ValueError:
                    pass
            return default

        def _get_bool(key: str, default: bool) -> bool:
            val = os.getenv(key)
            if val is not None:
                return val.strip().lower() in ("1", "true", "yes")
            return default

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            max_retries=_get_int("LLM_MAX_RETRIES", 3),
            base_delay=_get_float("LLM_BASE_DELAY", 1.0),
            max_delay=_get_float("LLM_MAX_DELAY", 60.0),
            timeout=_get_float("LLM_TIMEOUT", 30.0),
            rpm_limit=_get_int("LLM_RPM_LIMIT", 30),
            min_request_interval=_get_float("LLM_MIN_REQUEST_INTERVAL", 2.0),
            max_concurrent_requests=_get_int("LLM_MAX_CONCURRENT_REQUESTS", 2),
            max_calls_per_investigation=_get_int("LLM_MAX_CALLS_PER_INVESTIGATION", 12),
            max_provider_attempts_per_investigation=_get_int("LLM_MAX_PROVIDER_ATTEMPTS_PER_INVESTIGATION", 20),
            max_provider_attempts_per_generation=_get_int("LLM_MAX_PROVIDER_ATTEMPTS_PER_GENERATION", 4),
            max_repair_retries=_get_int("LLM_MAX_REPAIR_RETRIES", 2),
            enable_quota_tracking=_get_bool("LLM_ENABLE_QUOTA_TRACKING", True),
        )
