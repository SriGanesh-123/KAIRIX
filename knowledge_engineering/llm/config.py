"""Central LLM configuration shared by all agents."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    timeout: float = 30.0
    rpm_limit: int = 30
    min_request_interval: float = 1.5
    max_concurrent_requests: int = 2
    enable_quota_tracking: bool = True

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
            max_delay=_get_float("LLM_MAX_DELAY", 10.0),
            timeout=_get_float("LLM_TIMEOUT", 30.0),
            rpm_limit=_get_int("LLM_RPM_LIMIT", 30),
            min_request_interval=_get_float("LLM_MIN_REQUEST_INTERVAL", 1.5),
            max_concurrent_requests=_get_int("LLM_MAX_CONCURRENT_REQUESTS", 2),
            enable_quota_tracking=_get_bool("LLM_ENABLE_QUOTA_TRACKING", True),
        )
